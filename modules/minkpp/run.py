import argparse

from run_white_box import sample_indices
from utils import *

RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6]

def main(args):
	random.seed(args.seed)
	sampler = random.Random(args.seed)

	data_dict, all_data = load_data(args.data_dir, args.pilot, args.seed)
	data_logs = {}
	for _d in data_dict:
		data_dict[_d]['sampled_indices'] = sample_indices(all_data, data_dict[_d]['indices'], args.max_query, sampler=sampler)
		if args.model == 'udop':
			data_dict[_d]['sampled_indices'] = [
				_i for _i in data_dict[_d]['sampled_indices']
				if len(all_data[_i]['ocr_tokens']) > 0
			]  # workaround
			if len(data_dict[_d]['sampled_indices']) == 0:
				continue
		data_logs[_d] = {
			"loss": [],  # len(inds)
			'time': [], # len(inds)
		}
		for ratio in RATIOS:
			data_logs[_d].update({
				f'mink_{ratio}': [],
				f"mink++_{ratio}": [],
			})
	if args.model == 'udop':
		data_dict = {_k:_v for _k,_v in data_dict.items() if len(_v['sampled_indices']) > 0}

	start_expt = datetime.datetime.now()
	expt = args.expt if args.expt else f"exp_{start_expt.strftime('%d-%m-%Y-%H-%M-%S')}"

	model, prep_fn, forward_fn, processor = load_model(args.model, args.ckpt, freeze=True)
	model = cuda_and_maybe_copy(model, args.model)

	dloader = get_dataloader(args.data_dir, all_data, data_dict,args.model, args.ckpt, batch_size=1)

	for _full_batch in tqdm(dloader):
		for _k in _full_batch:
			if _k == 'images':
				for __k in _full_batch[_k]:
					_full_batch[_k][__k] = _full_batch[_k][__k].cuda()
			elif isinstance(_full_batch[_k], torch.Tensor):
				_full_batch[_k] = _full_batch[_k].cuda()
			elif all([isinstance(_t, torch.Tensor) for _t in _full_batch[_k]]):
				_full_batch[_k] = [_t.cuda() for _t in _full_batch[_k]]

		for _i in range(len(_full_batch["document_ids"])):
			start = datetime.datetime.now()
			_d = _full_batch["document_ids"][_i]
			_batch = prep_fn(_full_batch, _i)

			out = forward_fn(model, _batch, mode='eval')
			loss, logits = out.loss, out.logits  # logits.size() == (1, len, vocab)

			# _batch['labels'].size() == (1, len+1)
			if args.model == 'donut':
				mask = _batch['labels'][0, :-1] != -100
				pred_ids = _batch['labels'][0, :-1][mask].unsqueeze(-1)
				pred_logits = logits[0, mask.squeeze(0), :]
			elif args.model in ['pix2struct', 'udop']:
				pred_ids = _batch['labels'][0].unsqueeze(-1)
				pred_logits = logits[0]
			elif args.model == 'vt5':
				input_ids = processor(_batch['answers'][0], return_tensors='pt').input_ids
				pred_ids = input_ids[0].unsqueeze(-1).cuda()
				pred_logits = logits[0]
			assert pred_ids.size(0) == pred_logits.size(0)
			# pred_ids.size() == (len, 1); pred_logits.size() == (len, vocab)

			probs = torch.nn.functional.softmax(pred_logits, dim=-1)
			log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
			token_log_probs = log_probs.gather(dim=-1, index=pred_ids).squeeze(-1)
			mu = (probs * log_probs).sum(-1)
			sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

			## minK
			for ratio in RATIOS:
				k_length = int(len(token_log_probs) * ratio)
				topk = np.sort(token_log_probs.cpu())[:k_length]
				data_logs[_d][f'mink_{ratio}'].append(np.mean(topk).item())

			## minK++
			mink_plus = (token_log_probs - mu) / sigma.sqrt()
			for ratio in RATIOS:
				k_length = int(len(mink_plus) * ratio)
				topk = np.sort(mink_plus.cpu())[:k_length]
				data_logs[_d][f'mink++_{ratio}'].append(np.mean(topk).item())

			data_logs[_d]['loss'].append(out.loss.detach().cpu().item())
			data_logs[_d]['time'].append((datetime.datetime.now()-start).seconds)

	for _it, (_d, _ddict) in enumerate(data_dict.items()):
		data_logs[_d].update({
			"label":_ddict['label'],
			"indices": _ddict['sampled_indices'],
		})

	mink_tpr001fpr_lst, mink_tpr003fpr_lst, mink_X_lst, y_lst = [], [], [], []
	minkpp_tpr001fpr_lst, minkpp_tpr003fpr_lst, minkpp_X_lst = [], [], []
	for ratio in RATIOS:
		X_mink, X_minkpp, y = [], [], []
		for _d in data_logs:
			y += [data_logs[_d]["label"]]
			X_mink += [np.mean(data_logs[_d][f'mink_{ratio}'])]
			X_minkpp += [np.mean(data_logs[_d][f'mink++_{ratio}'])]
		try:
			mink_ranks = stats.rankdata(X_mink, method='ordinal')
			fpr, tpr, _ = metrics.roc_curve(y, mink_ranks)
			tpr001fpr = tpr[np.where(fpr<.01)[0][-1]]*100
			tpr003fpr = tpr[np.where(fpr<.03)[0][-1]]*100

			mink_tpr001fpr_lst.append(tpr001fpr)
			mink_tpr003fpr_lst.append(tpr003fpr)
			mink_X_lst.append(X_mink); y_lst.append(y)

			minkpp_ranks = stats.rankdata(X_minkpp, method='ordinal')
			fpr, tpr, _ = metrics.roc_curve(y, minkpp_ranks)
			tpr001fpr = tpr[np.where(fpr<.01)[0][-1]]*100
			tpr003fpr = tpr[np.where(fpr<.03)[0][-1]]*100

			minkpp_tpr001fpr_lst.append(tpr001fpr)
			minkpp_tpr003fpr_lst.append(tpr003fpr)
			minkpp_X_lst.append(X_minkpp)
		except ValueError as e:
			print(f"ratio={ratio}, error={e}")
			continue

	print(f">>>>>>>> Best:")
	max_ind = np.argmax(minkpp_tpr003fpr_lst)

	mink_y_pred = (np.array(mink_X_lst[max_ind]) > np.mean(mink_X_lst[max_ind])).astype(int)
	minkpp_y_pred = (np.array(minkpp_X_lst[max_ind]) > np.mean(minkpp_X_lst[max_ind])).astype(int)
	print(f"Min-K%: "
		+f"TPR@1%FPR={mink_tpr001fpr_lst[max_ind]:.2f}, "
		+f"TPR@3%FPR={mink_tpr003fpr_lst[max_ind]:.2f}, "
		+f"ACC={metrics.accuracy_score(y_lst[max_ind], mink_y_pred)*100:.2f}, "
		+f"F1={metrics.f1_score(y_lst[max_ind], mink_y_pred)*100:.2f}, "
		+f"K={RATIOS[max_ind]*100}%")
	print(f"Min-K%++: "
		+f"TPR@1%FPR={minkpp_tpr001fpr_lst[max_ind]:.2f}, "
		+f"TPR@3%FPR={minkpp_tpr003fpr_lst[max_ind]:.2f}, "
		+f"ACC={metrics.accuracy_score(y_lst[max_ind], minkpp_y_pred)*100:.2f}, "
		+f"F1={metrics.f1_score(y_lst[max_ind], minkpp_y_pred)*100:.2f}, "
		+f"K={RATIOS[max_ind]*100}%")
	save_dir = os.path.join(args.result_dir, expt)
	os.makedirs(save_dir, exist_ok=True)
	with open(os.path.join(save_dir, f"seed{args.seed}.pkl"), 'wb') as f:
		pickle.dump({'result': data_logs}, f)
	print("="*80)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='script to train')
	parser.add_argument('--model', type=str, help='model to finetune.')
	parser.add_argument('--ckpt', type=str, help='init checkpoint.')
	parser.add_argument('--data_dir', type=str, help='path to data.')
	parser.add_argument('--pilot', type=str, default=300, help='pilot.')
	parser.add_argument('--max_query', type=int, default=10, help='maximum number of queries.')
	parser.add_argument('--seed', type=int, default=1026, help='random seed.')
	parser.add_argument('--expt', type=str, default=None, help='experiment name.')
	parser.add_argument('--result_dir', type=str, default='save/result/', help='path to save result.')

	args = parser.parse_args()
	main(args)
