import argparse

from utils import *

class DonutClass(torch.nn.Module):
	def __init__(self, ckpt):
		super(DonutClass, self).__init__()

		# Lower resolution for stable training
		max_length = 128
		image_size = [1280, 960]

		# during pre-training, a LARGER image size was used [2560, 1920]
		self.config = VisionEncoderDecoderConfig.from_pretrained(ckpt)
		self.config.encoder.image_size = image_size
		self.config.decoder.max_length = max_length
		self.model = VisionEncoderDecoderModel.from_pretrained(ckpt, config=self.config)

		# TODO: we should update max_position_embeddings and interpolate the pre-trained ones:
		# https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602
		self.processor = DonutProcessor.from_pretrained(ckpt)
		self.processor.feature_extractor.size = image_size[::-1] # REVERSED (width, height)
		self.processor.feature_extractor.do_align_long_axis = False

		self.processor.tokenizer.add_tokens([
			"<s_docvqa>", "<yes/>", "<no/>", "<s_question>", "<s_answer>", "</s_answer>", "</s_question>"
		])
		self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))

	def forward(self, pixel_values, decoder_input_ids, labels):
		out = self.model(
			pixel_values=pixel_values,
			decoder_input_ids=decoder_input_ids,
			labels=labels
		)
		return out

def save_dp_model(save_dir, model_name, model):
	if model_name == 'donut':
		model.model.save_pretrained(os.path.join(save_dir, "last.ckpt"))
		model.processor.save_pretrained(os.path.join(save_dir, "last.ckpt"))
	else:
		raise ValueError(f"Invalid model: {model_name}.")

def finetune(args):
	save_dir = os.path.join('checkpoints/', args.model, args.expt+f'_seed{args.seed}')
	os.makedirs(save_dir, exist_ok=True)

	# Model/Optimizer
	model, _, forward_fn, eval_fn, processor = load_model(args.model, args.ckpt, run_eval=True)
	model = cuda_and_maybe_copy(model, args.model)
	optimizer, scheduler = get_optimizer(
		args.model, model, args.learning_rate, finetune=True,
		weight_decay=args.weight_decay, warmup_steps=args.warmup_steps, max_steps=args.max_steps
	)

	# Dataset/Dataloader
	train_ds, val_ds = FTDataset(args.data_dir, 'train'), FTDataset(args.data_dir, 'val')
	if args.model == 'vt5':
		collate_fn = partial(train_ds.collate_fn, model_name=args.model, model=model)
	elif args.model in ['donut', 'pix2struct', 'pix2struct-large', 'layoutlmv3', 'layoutlmv3-large', 'udop']:
		collate_fn = partial(train_ds.collate_fn, model_name=args.model, processor=processor)
	else:
		raise ValueError(f"Invalid model: {args.model}.")
	train_dl = DataLoader(
		dataset=train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		collate_fn=collate_fn,
		num_workers=8, # 8
		pin_memory=True,
	)
	val_dl = DataLoader(
		dataset=val_ds,
		batch_size=16,
		shuffle=False,
		collate_fn=collate_fn,
		num_workers=8,
		pin_memory=True,
	)
	print(
		'>'*5 + f"TRAIN/VAL dataset: {len(train_ds)}/{len(val_ds)}. "\
		+ f"Num. TRAIN/VAL steps: {len(train_dl)*args.num_epoch}/{len(val_dl)*args.num_epoch}."
	)
	# Train/Evaluate
	all_info, max_score = dict({
		'batch_size': args.batch_size, 'iteration': 0, 'loss':[],
		'accuracy':[], 'anls':[], 'val_accuracy':[], 'val_anls':[]
	}), -np.inf
	for _epoch in range(args.num_epoch):
		train_loss, train_acc, train_anls = [], [], []
		for _i, _batch in enumerate(train_dl):
			ts = datetime.datetime.now()
			for _k in _batch:
				if _k == 'images':
					for __k in _batch[_k]:
						_batch[_k][__k] = _batch[_k][__k].cuda()
				elif isinstance(_batch[_k], torch.Tensor):
					_batch[_k] = _batch[_k].cuda()
				elif isinstance(_batch[_k], list) and all([isinstance(__k, torch.Tensor) for __k in _batch[_k]]):
					for __k in range(len(_batch[_k])):
						_batch[_k][__k] = _batch[_k][__k].cuda()

			out = forward_fn(model, _batch)
			loss = out.loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if scheduler is not None:
				scheduler.step()

			train_loss += [loss.mean().item()]
			mess = ""
			if _i % 20 == 0 or _i == len(train_dl)-1:
				mess = f"[TRAIN] Epoch[{_epoch}][{_i}] batch loss: {loss.mean().item():.6f}, " +\
				f"learning rate: {optimizer.param_groups[0]['lr']:.6f}, time: {(datetime.datetime.now() - ts).total_seconds():.2f}s"
			if _i % 100 == 0:
				_batch_preds = eval_fn(model, _batch, processor=processor)
				_batch_acc, _batch_anls = [], []
				for _a,_p in zip(_batch['answers'], _batch_preds):
					_batch_acc += [accuracy(_a,_p)]; _batch_anls += [anls(_a,_p)]
				train_acc += _batch_acc; train_anls += _batch_anls
				mess += f", ACC: {np.mean(_batch_acc):.4f}, ANLS: {np.mean(_batch_anls):.4f}"
			if mess != "": print(mess)

		trainscore, trainloss = (sum(train_acc)+sum(train_anls))/(2*len(train_anls)), sum(train_loss)/len(train_dl)
		print('>'*5 + f" Epoch: {_epoch}, train loss: {trainloss:.4f}, train score: {trainscore:.4f}, learning rate: {optimizer.param_groups[0]['lr']:.6f}.")

		eval_acc, eval_anls, is_best = [], [], False
		for _batch in tqdm(val_dl, desc=f"eval epoch {_epoch}"):
			for _k in _batch:
				if _k == 'images':
					for __k in _batch[_k]:
						_batch[_k][__k] = _batch[_k][__k].cuda()
				elif isinstance(_batch[_k], torch.Tensor):
					_batch[_k] = _batch[_k].cuda()
				elif isinstance(_batch[_k], list) and all([isinstance(__k, torch.Tensor) for __k in _batch[_k]]):
					for __k in range(len(_batch[_k])):
						_batch[_k][__k] = _batch[_k][__k].cuda()
			_batch_preds = eval_fn(model, _batch, processor=processor)
			_batch_acc, _batch_anls = [], []
			for _a,_p in zip(_batch['answers'], _batch_preds):
				_batch_acc += [accuracy(_a,_p)]; _batch_anls += [anls(_a,_p)]
			eval_acc += _batch_acc; eval_anls += _batch_anls

		all_info['epoch'] = _epoch; all_info['iteration'] += len(train_dl)
		all_info['loss'] += train_loss; all_info['accuracy'] += train_acc; all_info['anls'] += train_anls
		all_info['val_accuracy'] += eval_acc; all_info['val_anls'] += eval_anls

		val_anls = np.mean(eval_anls)
		if val_anls > max_score:
			max_score = val_anls; is_best = True
		print('>'*5 + f"[VALIDATION] Epoch[{_epoch}] ACC {np.mean(eval_acc):.4f}, ANLS {val_anls:.4f}.\t"\
			+ ("\tBEST Performance!" if is_best else ""))
		save_model(save_dir, args.model, model, processor, _epoch, best=is_best)
		json.dump(all_info, open(os.path.join(save_dir, 'train_info.json'), 'w'))  # save every epoch

def dp_donut(args):
	from torch.func import functional_call, vmap, grad

	SAVE_DIR = os.path.join('checkpoints/dp', args.model, args.expt+f'_seed={args.seed}')
	os.makedirs(SAVE_DIR, exist_ok=True)

	# Model/Optimizer
	model = DonutClass(args.ckpt)
	processor = model.processor
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

	model.cuda()

	def compute_loss(params, buffers, pixel_values, decoder_input_ids, label):
		pixel_values = pixel_values.unsqueeze(0)
		decoder_input_ids = decoder_input_ids.unsqueeze(0)
		label = label.unsqueeze(0)
		out = functional_call(
			model, (params, buffers), (pixel_values, decoder_input_ids, label,)
		)
		loss = out.loss
		return loss, (loss)

	ft_grad = grad(compute_loss, has_aux=True)
	ft_vmap_grad = vmap(ft_grad, in_dims=(None, None, 0, 0, 0), randomness="different")

	param_dict = {n: p.requires_grad for n, p in model.named_parameters()}

	# Dataset/Dataloader
	train_ds = FTDataset(args.data_dir, 'train')
	collate_fn = partial(train_ds.collate_fn, model_name='donut', processor=processor)

	train_document_dict = dict()  # Dict of document to its question indices in the dataset
	for _i, _record in tqdm(enumerate(train_ds.all_data), desc='reading dataset'):
		_document_id = _record['image_name']
		if _document_id in train_document_dict:
			train_document_dict[_document_id].append(_i)
		else:
			train_document_dict[_document_id] = [_i]
	train_document_lst = list(train_document_dict.items())

	S = args.sensitivity
	c = args.noise_multiplier
	batch_size = args.batch_size
	T_gd = args.num_epoch * len(train_document_lst) // batch_size
	sampling_prob = np.divide(batch_size, len(train_document_lst))
	print('>'*5 + f" TRAIN dataset: {len(train_document_lst)}, Num. TRAIN steps: {T_gd}.")

	# Train
	assert batch_size > args.mini_batch_size
	all_info = dict({
		'batch_size': batch_size, 'mini_batch_size': args.mini_batch_size,
		'iteration': 0, 'loss':[]
	})
	for _i in range(T_gd):
		ts = datetime.datetime.now()

		doc_inds = np.arange(len(train_document_lst))[
			np.random.choice(
				a=[False, True], size=len(train_document_lst),
				p=[1-sampling_prob, sampling_prob]
			)
		]  # Poisson subsampling over documents

		document_accum_grad, train_loss = OrderedDict(), []  # global batch of documents
		for _j, _doc_i in enumerate(doc_inds):  # for each document
			_document_id, _inds = train_document_lst[_doc_i]
			_train_dl = DataLoader(
				dataset=Subset(train_ds, _inds),
				batch_size=args.mini_batch_size,
				shuffle=True,
				collate_fn=collate_fn,
				num_workers=8,
				pin_memory=True
			)

			optimizer.zero_grad()

			_question_grads, _question_train_losses = [], []
			for _batch in _train_dl:  # for each batch of questions from this document
				for _k in _batch:
					if _k == 'images':
						for __k in _batch[_k]:
							_batch[_k][__k] = _batch[_k][__k].cuda()
					elif isinstance(_batch[_k], torch.Tensor):
						_batch[_k] = _batch[_k].cuda()
					elif isinstance(_batch[_k], list) and all([isinstance(__k, torch.Tensor) for __k in _batch[_k]]):
						for __k in range(len(_batch[_k])):
							_batch[_k][__k] = _batch[_k][__k].cuda()

				images = _batch['images']['pixel_values']
				text_inputs = _batch['text_input_ids'][:, :-1] # training [<start>,q,u,e,s,t,i,o,n,<sep>,a,n,s,w,e,r]
				prompts = _batch['prompt_input_ids'] # evaluation [<start>q,u,e,s,t,i,o,n,<sep>]
				labels = _batch['labels'][:, 1:] # [q,u,e,s,t,i,o,n,<sep>,a,n,s,w,e,r,<end>]

				# params to be optimized, while model's params are deactivated
				_params = {n: p.detach() for n, p in model.named_parameters()}
				_buffers = {n: b.detach() for n, b in model.named_buffers()}

				# 1. compute per-question grads
				_grads, _losses = ft_vmap_grad(_params, _buffers, images, text_inputs, labels)
				_grad_shapes = get_shape(_grads)

				# 2. flatten per-question grads
				_grad_keys, _flatten_grads = flatten_grads(_grads)

				del _grads # delete ft_tmgrads after flattened

				_question_grads.extend(_flatten_grads)
				_question_train_losses += _losses.tolist()

			# 3. aggregate (avg) question grads of this document => this document grad
			_question_agg_grad = torch.mean(torch.stack(_question_grads, dim=0), dim=0)

			# 4. clip this document grad
			_clipped_grad = clip_grad(_question_agg_grad, clip_norm=S)
			# assert torch.linalg.vector_norm(_clipped_grad, ord=2) <= S

			# 5. reconstruct and zero out frozen params grads
			_agg_grad = reconstruct_shape(_clipped_grad, _grad_keys, _grad_shapes)
			_agg_grad = {_k: _agg_grad[_k] if param_dict[_k] else None for _k in _agg_grad}

			# 6. aggregate this document grad
			if _j == 0:
				for _k in _agg_grad:
					document_accum_grad[_k] = _agg_grad[_k]
			else:
				for _k in _agg_grad:
					document_accum_grad[_k] += _agg_grad[_k]
			train_loss += [np.mean(_question_train_losses)]

			del _agg_grad, _clipped_grad, _question_agg_grad, _question_grads

			if args.empty_cache:
				torch.cuda.empty_cache()

		# 7. add noise
		document_accum_grad = add_noise(document_accum_grad, batch_size, noise_multiplier=c, sensitivity=S)

		# 8. update grads to model
		for _n, _p in model.named_parameters():
			_p.grad = document_accum_grad[_n]

		del document_accum_grad

		optimizer.step()

		print(
			f"[TRAIN] Iter[{_i}] Num.Docs={len(doc_inds)}, " +\
			f"Num.questions={sum([len(train_document_lst[_di][1]) for _di in doc_inds])}, " +\
			f"global batch loss: {np.mean(train_loss):.6f}, " +\
			f"learning rate: {optimizer.param_groups[0]['lr']:.6f}, " +\
			f"time: {(datetime.datetime.now() - ts).total_seconds():.2f}s"
		)

		all_info['iteration'] += len(train_loss); all_info['loss'] += train_loss

	json.dump(all_info, open(os.path.join(SAVE_DIR, 'train_info.json'), 'w'))

	save_dp_model(SAVE_DIR, args.model, model)
	print(f">>>>> Saved model to: {os.path.join(SAVE_DIR, 'last.ckpt')}")

	donut_inference(os.path.join(SAVE_DIR, "last.ckpt"), data_dir='data/docvqa/', dset='docvqa', lowreso=True)

def dp_vt5(args):
	raise NotImplementedError

def parse_args():
	parser = argparse.ArgumentParser(description='script to train')
	parser.add_argument('--model', type=str, help='model to finetune.')
	parser.add_argument('--ckpt', type=str, help='init checkpoint.')
	parser.add_argument('--data_dir', type=str, help='path to data.')

	parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate.')
	parser.add_argument('--batch_size', type=int, default=8, help='batch size.')
	parser.add_argument('--num_epoch', type=int, default=10, help='num. epoch.')
	parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay.')
	parser.add_argument('--warmup_steps', type=int, default=1000, help='num. warmup steps.')
	parser.add_argument('--max_steps', type=int, default=20000, help='max steps.')

	parser.add_argument('--dp', default=False, action='store_true', help='group-level DP.')
	parser.add_argument('--sensitivity', type=float, default=1.5, help='sensitivity.')
	parser.add_argument('--noise_multiplier', type=float, default=1.0, help='noise multiplier.')
	parser.add_argument('--mini_batch_size', type=int, default=2, help='batch size.')

	parser.add_argument('--empty_cache', default=False, action='store_true', help='empty cuda cache.')
	parser.add_argument('--seed', type=int, default=1027, help='random seed.')
	parser.add_argument('--expt', type=str, default=None, help='experiment name.')

	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	print(args)

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.seed)

	if args.dp:
		assert args.model in ['donut', 'vt5'], f"Invalid model: {args.model}"
		if args.model == 'donut':
			dp_donut(args)
		elif args.model == 'vt5':
			dp_vt5(args)
	else:
		finetune(args)

if __name__ == '__main__':
	main()
