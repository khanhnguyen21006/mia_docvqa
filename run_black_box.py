import argparse

from run_white_box import seed_everything, load_data, load_model
from run_white_box import run_inputgradient, run_finetunelayer, run_baseline
from utils import *

BLACKBOX_OUTPUT = './save/blackbox/'

def parse_args():
	parser = argparse.ArgumentParser(description='script to train proxy')
	parser.add_argument('--model', type=str, help='model to attack.')
	parser.add_argument('--ckpt', type=str, help='blackbox checkpoint.')
	parser.add_argument('--proxy', type=str, help='model to attack.')
	parser.add_argument('--proxy_ckpt', type=str, help='proxy checkpoint.')

	parser.add_argument('--data_dir', type=str, help='path to data.')
	parser.add_argument('--pilot', type=int, default=300, help='pilot.')

	parser.add_argument('--batch_size', type=int, default=8, help='train batch size.')
	parser.add_argument('--lr', type=float, default=3e-5, help='train lr.')
	parser.add_argument('--num_epoch', type=int, default=5, help='train epochs.')

	parser.add_argument('--seed', type=int, default=1026, help='random seed.')
	parser.add_argument('--save_dir', type=str, help='path to save checkpoint.')
	parser.add_argument('--save_name', type=str, help='checkpoint name.')

	args = parser.parse_args()
	return args

def query_blackbox(model, ckpt, data_dir, pilot, seed, question='v0'):
	dset = os.path.basename(os.path.normpath(data_dir))
	assert dset in ['docvqa', 'docvqav0', 'pfl'], ValueError(f"Invalid dataset: {dset}.")

	output_dir = os.path.join(BLACKBOX_OUTPUT, model, dset, f'pilot/seed{seed}' if pilot > 0 else '')
	os.makedirs(output_dir, exist_ok=True)

	output_path = os.path.join(output_dir, f'blackbox_output_question={question}.json')
	if os.path.exists(output_path):
		blackbox_output = json.load(open(output_path, 'r'))
	else:
		if model == 'vt5':
			blackbox_output = vt5_inference(ckpt, data_dir, seed if pilot > 0 else None, dset, attack='mia') # , question_type=question
		elif model == 'donut':
			blackbox_output = donut_inference(ckpt, data_dir, seed if pilot > 0 else None, dset, attack='mia')
		elif model in ['pix2struct', 'pix2struct-large']:
			blackbox_output = pix2struct_inference(ckpt, data_dir, seed if pilot > 0 else None, dset, attack='mia')
		elif model in ['layoutlmv3', 'layoutlmv3-large']:
			blackbox_output = layoutlmv3_inference(ckpt, data_dir, seed if pilot > 0 else None, dset, attack='mia')
		elif model == 'udop':
			blackbox_output = udop_inference(ckpt, data_dir, seed if pilot > 0 else None, dset, attack='mia')
		else:
			raise ValueError(f"Invalid model: {model}.")
		json.dump(blackbox_output, open(output_path, 'w'))
	return blackbox_output
	
def train_proxy(data_dict, data_npy, blackbox_output, args):
	os.makedirs(args.save_dir, exist_ok=True)

	start_time = datetime.datetime.now()

	proxy_model, prep_fn, forward_fn, eval_fn, processor = load_model(args.proxy, args.proxy_ckpt, run_eval=True)
	proxy_model = cuda_and_maybe_copy(proxy_model, args.proxy)
	optimizer, _ = get_optimizer(args.proxy, proxy_model, args.lr)

	train_dl = get_dataloader(args.data_dir, data_npy, data_dict, args.proxy, args.proxy_ckpt,
									outputs=blackbox_output, batch_size=args.batch_size, shuffle=True)
	print('>'*5 + f" Num. train steps: {len(train_dl)*args.num_epoch}.")

	all_info, max_score = dict({'batch_size': args.batch_size, 'iteration':len(train_dl), 'loss':[], 'accuracy':[], 'anls':[]}), -np.inf
	for _epoch in range(args.num_epoch):
		ts = datetime.datetime.now()
		epoch_loss, epoch_acc, epoch_anls, eval_count, best = [], [], [], 0, False
		for _bind, _batch in enumerate(train_dl):
			for _k in _batch:
				if _k == 'images':
					for __k in _batch[_k]:
						_batch[_k][__k] = _batch[_k][__k].cuda()
				elif isinstance(_batch[_k], torch.Tensor):
					_batch[_k] = _batch[_k].cuda()
				elif all([isinstance(_t, torch.Tensor) for _t in _batch[_k]]):
					_batch[_k] = [_t.cuda() for _t in _batch[_k]]

			out = forward_fn(proxy_model, _batch)
			loss = out.loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss += [loss.mean().item()]
			mess = ""
			if _bind % 50 == 0 or _bind == len(train_dl)-1:
				mess = f"[TRAIN] Epoch[{_epoch}][{_bind}] batch loss: {loss.mean().item():.6f}, " +\
														f"learning rate: {optimizer.param_groups[0]['lr']}"
			if _bind % 100 == 0 or _bind == len(train_dl)-1:
				_batch_preds = eval_fn(proxy_model, _batch, processor=processor)
				_batch_acc, _batch_anls = [], []
				for _a,_p in zip(_batch['answers'], _batch_preds):
					_batch_acc += [accuracy(_a,_p)]; _batch_anls += [anls(_a,_p)]
					print(f"answer={_a}, predict={_p}")
				epoch_acc += _batch_acc; epoch_anls += _batch_anls
				mess += f", ACC: {np.mean(_batch_acc):.4f}, ANLS: {np.mean(_batch_anls):.4f}"
			if mess != "": print(mess)

		all_info['epoch'] = _epoch
		all_info['loss'] += epoch_loss; all_info['accuracy'] += epoch_acc; all_info['anls'] += epoch_anls;
		trainscore, trainloss = (sum(epoch_acc)+sum(epoch_anls))/(2*len(epoch_anls)), sum(epoch_loss)/len(train_dl)
		if trainscore > max_score: 
			max_score = trainscore; best = True
		print('>'*5 + f" Epoch: {_epoch}, train loss: {trainloss:.4f}, train metric: {trainscore:.4f}, " +\
			f"learning rate: {optimizer.param_groups[0]['lr']:.6f}, time: {(datetime.datetime.now() - ts).total_seconds():.2f}s." +\
			(" BEST!!!" if best else "")
		)
		save_model(args.save_dir, args.proxy, proxy_model, processor, _epoch, best=best)
		json.dump(all_info, open(os.path.join(args.save_dir, 'train_info.json'), 'w'))

	elapsed = datetime.datetime.now() - start_time
	print(f">>>>>>>> FINISHED!!! Time elapsed: {elapsed.seconds//3600} hours, {elapsed.seconds//60%60} mins.")

def main():
	args = parse_args()
	print(args)

	seed_everything(args.seed)

	blackbox_output = query_blackbox(args.model, args.ckpt, args.data_dir, args.pilot, args.seed)

	data_dict, data_npy = load_data(args.data_dir, args.pilot, args.seed)

	train_proxy(data_dict, data_npy, blackbox_output, args)

if __name__ == '__main__':
	main()