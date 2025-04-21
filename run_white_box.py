import argparse

from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	transformers.set_seed(seed)

def sample_indices(alldata, inds, max_query, sampler=None):
	dinds = []
	_ind = 0 # Prioritize documents we can find the answer
	while len(dinds) < max_query and _ind < len(inds):
		record = alldata[inds[_ind]]
		context = ' '.join([_t.lower() for _t in record['ocr_tokens']])
		answer = [_an.lower() for _an in record['answers']] if isinstance(record['answers'], list) else [record['answers'].lower()]
		if any([_an in context or re.sub(r'\s+', ' ', _an) in context for _an in answer]):
			dinds.append(inds[_ind])
		_ind += 1
	l = max_query - len(dinds)
	if l > 0:
		leftover = [_i for _i in inds if _i not in dinds]
		dinds += sampler.sample(leftover, l) if sampler and (len(leftover) > l) else leftover[:l]
	return dinds

def select_finetune_layer(model, model_name, layer_name, verbose=True):
	if model_name == 'vt5':
		_model = model.model
	else:
		_model = model
	all_modules = [n for n, _ in _model.named_modules()]
	if layer_name in all_modules:
		pass
	elif layer_name == 'random_decoder_layer': # SELECT FROM "decoder."+["fc1", "fc2", "block_ln", "self_attn_ln", "encdec_attn_ln"]
		assert model_name in ['vt5', 'donut', 'pix2struct', 'pix2struct-large', 'udop']
		layer_name = random.sample([
			n for n in all_modules
			if ('language_backbone.decoder' in n and 'DenseReluDense.wi' in n) or ('decoder.model.decoder' in n and 'fc1' in n) or ('decoder.layer' in n and 'wi_0' in n) or ('decoder.block' in n and 'DenseReluDense.wi' in n)
			or ('language_backbone.decoder' in n and 'DenseReluDense.wo' in n) or ('decoder.model.decoder' in n and 'fc2' in n) or ('decoder.layer' in n and 'wo' in n) or ('decoder.block' in n and 'DenseReluDense.wo' in n)
			or ('language_backbone.decoder' in n and 'layer_norm' in n) or ('decoder.model.decoder' in n and 'final_layer_norm' in n) or ('decoder.layer' in n and 'mlp.layer_norm' in n) or ('decoder.block' in n and 'layer.2.layer_norm' in n)
			or ('language_backbone.decoder' in n and 'layer.0.layer_norm' in n) or ('decoder.model.decoder' in n and 'self_attn_layer_norm' in n) or ('decoder.layer' in n and 'self_attention.layer_norm' in n) or ('decoder.block' in n and 'layer.0.layer_norm' in n)
			or ('language_backbone.decoder' in n and 'layer.1.layer_norm' in n) or ('decoder.model.decoder' in n and 'encoder_attn_layer_norm' in n) or ('decoder.layer' in n and 'encoder_decoder_attention.layer_norm' in n) or ('decoder.block' in n and 'layer.1.layer_norm' in n)
		], 1)[0]
	elif layer_name == 'random_decoder_fc1':
		assert model_name in ['vt5', 'donut', 'pix2struct', 'pix2struct-large', 'udop']
		layer_name = random.sample([
			n for n in all_modules
			if ('language_backbone.decoder' in n and 'DenseReluDense.wi' in n)
			or ('decoder.model.decoder' in n and 'fc1' in n)
			or ('decoder.layer' in n and 'wi_0' in n)
			or ('decoder.block' in n and 'DenseReluDense.wi' in n)
		], 1)[0]
	elif layer_name == 'random_decoder_fc2':
		assert model_name in ['vt5', 'donut', 'pix2struct', 'pix2struct-large', 'udop']
		layer_name = random.sample([
			n for n in all_modules
			if ('language_backbone.decoder' in n and 'DenseReluDense.wo' in n)
			or ('decoder.model.decoder' in n and 'fc2' in n)
			or ('decoder.layer' in n and 'wo' in n)
			or ('decoder.block' in n and 'DenseReluDense.wo' in n)
		], 1)[0]
	elif layer_name == 'random_decoder_block_fln':
		assert model_name in ['vt5', 'donut', 'pix2struct', 'pix2struct-large', 'udop']
		layer_name = random.sample([
			n for n in all_modules
			if ('language_backbone.decoder' in n and 'layer.2.layer_norm' in n)
			or ('decoder.model.decoder' in n and 'final_layer_norm' in n)
			or ('decoder.layer' in n and 'mlp.layer_norm' in n)
			('decoder.block' in n and 'layer.2.layer_norm' in n)
		], 1)[0]
	elif layer_name == 'random_decoder_block_ln':
		assert model_name in ['vt5', 'donut', 'pix2struct', 'pix2struct-large', 'udop']
		layer_name = random.sample([
			n for n in all_modules
			if ('language_backbone.decoder' in n and 'layer.0.layer_norm' in n)
			or ('language_backbone.decoder' in n and 'layer.1.layer_norm' in n)
			or ('decoder.model.decoder' in n and 'self_attn_layer_norm' in n)
			or ('decoder.model.decoder' in n and 'encoder_attn_layer_norm' in n)
			or ('decoder.layer' in n and 'self_attention.layer_norm' in n)
			or ('decoder.layer' in n and 'encoder_decoder_attention.layer_norm' in n)
			or ('decoder.block' in n and 'layer.0.layer_norm' in n)
			or ('decoder.block' in n and 'layer.1.layer_norm' in n)
		], 1)[0]
	elif layer_name == 'random_encoder_layer': # SELECT FROM "decoder."+["fc1", "fc2", "block_fln"]
		assert model_name in ['layoutlmv3', 'layoutlmv3-large']
		layer_name = random.sample([
			n for n in all_modules
			if ('layoutlmv3.encoder.layer' in n and 'intermediate.dense' in n)
			or ('layoutlmv3.encoder.layer' in n and 'output.dense' in n)
			or ('layoutlmv3.encoder.layer' in n and 'output.LayerNorm' in n)
		], 1)[0]
	elif layer_name == 'random_encoder_fc1':
		assert model_name in ['layoutlmv3', 'layoutlmv3-large']
		layer_name = random.sample([
			n for n in all_modules
			if ('layoutlmv3.encoder.layer' in n and 'intermediate.dense' in n)
		], 1)[0]
	elif layer_name == 'random_encoder_fc2':
		assert model_name in ['layoutlmv3', 'layoutlmv3-large']
		layer_name = random.sample([
			n for n in all_modules
			if ('layoutlmv3.encoder.layer' in n and 'output.dense' in n)
		], 1)[0]
	elif layer_name == 'random_decoder_block_fln':
		assert model_name in ['layoutlmv3', 'layoutlmv3-large']
		layer_name = random.sample([
			n for n in all_modules
			if ('layoutlmv3.encoder.layer' in n and 'output.LayerNorm' in n)
		], 1)[0]
	else:
		raise ValueError(f"Invalid layer: {layer_name} in model: {model_name}.")
	print('>'*5 + f' Finetune layer: {layer_name}.')
	_ft_params = [n for n in all_modules if layer_name in n]

	freeze_params(_model, _ft_params, verbose)
	return model, layer_name

def yield_params(model, model_name, named=False):
	if model_name == 'vt5':
		if named:
			for _n,_p in model.model.named_parameters():
				yield (_n,_p)
		else:
			for _p in model.model.parameters():
				yield _p
	elif model_name in ['donut', 'pix2struct', 'pix2struct-large', 'layoutlmv3', 'layoutlmv3-large', 'udop']:
		if named:
			for _n,_p in model.named_parameters():
				yield (_n,_p)
		else:
			for _p in model.parameters():
				yield _p
	else:
		raise ValueError(f"Invalid model: {model_name}")

def get_state_dict(model, model_name):
	if model_name == 'vt5':
		return model.model.state_dict()
	elif model_name in ['donut', 'pix2struct', 'pix2struct-large', 'layoutlmv3', 'layoutlmv3-large', 'udop']:
		return model.state_dict()
	else:
		raise ValueError(f"Invalid model: {model_name}")

def run_finetunelayer(args, start_model=None, **kwargs):
	sampler = random.Random(args.seed)

	data_dict, all_data = load_data(args.data_dir, args.pilot, args.seed)
	data_logs = {}
	for _d in data_dict:
		data_dict[_d]['sampled_indices'] = sample_indices(all_data, data_dict[_d]['indices'], args.max_query, sampler=sampler)
		data_logs[_d] = {
			"loss": [],  # len(inds)*step
			"update": [],  # len(inds)
			"step": [],  # len(inds)
			"accuracy": [],  # len(inds)*step
			"anls": [],  # len(inds)*step
			"output": [],  # len(inds)*step
		}

	start_expt = datetime.datetime.now()
	expt = args.expt if args.expt else f"exp_{start_expt.strftime('%d-%m-%Y-%H-%M-%S')}"
	"""
		pix2struct: [
			'decoder.layer.11.mlp.DenseReluDense.wi_0',
			'decoder.layer.11.mlp.DenseReluDense.wo'
			'decoder.layer.11.self_attention.layer_norm', 'decoder.layer.11.encoder_decoder_attention.layer_norm'
			'decoder.layer.11.mlp.layer_norm'
			'decoder.final_layer_norm', 'decoder.embed_tokens'
		]
		donut: [
			'decoder.model.decoder.layers.3.fc1',
			'decoder.model.decoder.layers.3.fc2'
			'decoder.model.decoder.layers.3.self_attn_layer_norm', 'decoder.model.decoder.layers.3.encoder_attn_layer_norm'
			'decoder.model.decoder.layers.3.final_layer_norm'
			'decoder.model.decoder.layer_norm', 'decoder.model.decoder.embed_tokens'
		]
		VT5: [
			'language_backbone.decoder.block.11.layer.2.DenseReluDense.wi',
			'language_backbone.decoder.block.11.layer.2.DenseReluDense.wo',
			'language_backbone.decoder.block.11.layer.0.layer_norm', 'language_backbone.decoder.block.11.layer.1.layer_norm'
			'language_backbone.decoder.block.11.layer.2.layer_norm'
			'language_backbone.decoder.final_layer_norm', 'language_backbone.lm_head'
		]
		layoutlmv3: [
			'layoutlmv3.encoder.layer.11.intermediate.dense',
			'layoutlmv3.encoder.layer.11.output.dense',
			'layoutlmv3.encoder.layer.11.attention.output.LayerNorm',
			'layoutlmv3.encoder.layer.11.output.LayerNorm'
			'qa_outputs.dense', 'qa_outputs.out_proj'
		]
		udop: [
			'decoder.block.23.layer.2.DenseReluDense.wi',
			'decoder.block.23.layer.2.DenseReluDense.wo',
			'decoder.block.23.layer.0.layer_norm', 'decoder.block.23.layer.1.layer_norm'
			'decoder.block.23.layer.2.layer_norm',
			'decoder.final_layer_norm', 'shared'
		]
	"""
	if start_model is None:
		start_model, prep_fn, forward_fn, eval_fn, processor = load_model(args.model, args.ckpt, run_eval=True)
	else:
		prep_fn, forward_fn, eval_fn, processor = kwargs.get("prep_fn", None), kwargs.get("forward_fn", None), kwargs.get("eval_fn", None), kwargs.get("processor", None)
	start_model, layer = select_finetune_layer(start_model, args.model, args.layer)
	if args.lora:
		start_model = add_lora(start_model, args.model, layer)
	print('>'*5 + f' Finetune params: {[_n for _n,_p in yield_params(start_model, args.model, named=True) if _p.requires_grad]}.')

	dloader = get_dataloader(args.data_dir, all_data, data_dict, args.model, args.ckpt)

	# run attack
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

			finetune_model = cuda_and_maybe_copy(start_model, args.model, maybe_copy=True)
			optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, yield_params(finetune_model, args.model)), lr=args.step_size)

			flatten_before = torch.cat([
				get_state_dict(finetune_model, args.model)[_n].clone().detach().flatten()
				for _n,_p in yield_params(finetune_model, args.model, named=True) if _p.requires_grad
			])

			step, prev_loss = 0, 0.0
			loss_prog, acc_prog, anls_prog, pred_prog = [], [], [], []
			while step < args.max_step:
				preds = eval_fn(finetune_model, _batch, processor=processor)
				acc_val = accuracy(_batch['answers'][0], preds[0]); anls_val = anls(_batch['answers'][0], preds[0])

				out = forward_fn(finetune_model, _batch)
				loss = out.loss

				cur_loss = loss.clone().detach().item()
				if not args.no_stop_on_delta and np.abs(cur_loss-prev_loss) < args.threshold:
					loss_prog += [cur_loss]
					acc_prog += [acc_val]; anls_prog += [anls_val]; pred_prog += [preds[0]]
					break

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				prev_loss = cur_loss; loss_prog += [prev_loss]
				acc_prog += [acc_val]; anls_prog += [anls_val]; pred_prog += [preds[0]]
				step += 1

			flatten_after = torch.cat([
				_p.clone().detach().flatten()
				for _p in yield_params(finetune_model, args.model) if _p.requires_grad
			])
			updatenorm2 = (flatten_after-flatten_before).data.norm(2).item()

			data_logs[_d]['loss'].append(loss_prog); data_logs[_d]['update'].append(updatenorm2)
			data_logs[_d]['step'].append(step); data_logs[_d]['anls'].append(anls_prog)
			data_logs[_d]['accuracy'].append(acc_prog); data_logs[_d]['output'].append(pred_prog)

	for _it, (_d, _ddict) in enumerate(data_dict.items()):
		data_logs[_d].update({
			"label":_ddict['label'],
			"indices": _ddict['sampled_indices'],
		})

	expt_dir = os.path.join(args.result_dir, expt)
	expt_cfg = f"seed{args.seed}_{layer.replace('.', '_')}{f'_lora' if args.lora else ''}_alpha{str(args.step_size).replace('.','')}_tau{str(args.threshold).replace('.','')}_T{str(args.max_step)}"
	save_result(expt_dir, expt_cfg, data_logs)

	print("RESULT:")
	print(f"FL(layer={layer}{f'_lora' if args.lora else ''},alpha={args.step_size},tau={args.threshold},T={args.max_step}): "\
		+json.dumps(evaluate(expt_dir, expt_cfg, args.predictor, FEATURES, ['update'], AGG_FNS, args.seed), indent=4))
	elapsed = datetime.datetime.now() - start_expt
	print(f"Finished!!! time elapsed: {elapsed.seconds//3600} hours, {elapsed.seconds//60%60} mins.")
	print("="*20)

def run_inputgradient(args, start_model=None, **kwargs):
	sampler = random.Random(args.seed)

	data_dict, all_data = load_data(args.data_dir, args.pilot, args.seed)
	data_logs = {}
	for _d in data_dict:
		data_dict[_d]['sampled_indices'] = sample_indices(all_data, data_dict[_d]['indices'], args.max_query, sampler=sampler)
		data_logs[_d] = {
			"loss": [],  # len(inds)*step
			"update": [],  # len(inds)
			"step": [],  # len(inds)
			"accuracy": [],  # len(inds)*step
			"anls": [],  # len(inds)*step
			"output": [],  # len(inds)*step
		}

	if start_model is None:
		start_model, prep_fn, forward_fn, eval_fn, processor = load_model(args.model, args.ckpt, freeze=True, run_eval=True)
	else:
		prep_fn, forward_fn, eval_fn, processor = kwargs.get("prep_fn", None), kwargs.get("forward_fn", None), kwargs.get("eval_fn", None), kwargs.get("processor", None)
	start_model = cuda_and_maybe_copy(start_model, args.model)

	dloader = get_dataloader(args.data_dir, all_data, data_dict, args.model, args.ckpt)

	start_expt = datetime.datetime.now()
	expt = args.expt if args.expt else f"exp_{start_expt.strftime('%d-%m-%Y-%H-%M-%S')}"

	# run attack
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

			_batch = prep_fn(_full_batch, _i) # batch size must be 1
			_batch['images']['pixel_values'].requires_grad = True

			optimizer = torch.optim.Adam([_batch['images']['pixel_values']], lr=args.step_size)

			step, prev_loss = 0, 0.0
			loss_prog, _grads, acc_prog, anls_prog, pred_prog = [], [], [], [], []
			while step < args.max_step:
				preds = eval_fn(start_model, _batch, processor=processor)
				acc_val, anls_val = accuracy(_batch['answers'][0], preds[0]), anls(_batch['answers'][0], preds[0])

				out = forward_fn(start_model, _batch)
				loss = out.loss

				cur_loss = loss.clone().detach().item()
				if not args.no_stop_on_delta and np.abs(cur_loss-prev_loss) < args.threshold:
					loss_prog += [cur_loss]
					acc_prog += [acc_val]; anls_prog += [anls_val]; pred_prog += [preds[0]]
					break

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				flatten_current = _batch['images']['pixel_values'].grad.clone().detach().flatten()

				step += 1
				prev_loss = cur_loss; loss_prog += [prev_loss]; _grads += [flatten_current.data.norm(2).item()]
				acc_prog += [acc_val]; anls_prog += [anls_val]; pred_prog += [preds[0]]

			data_logs[_d]['loss'].append(loss_prog); data_logs[_d]['update'].append(sum(_grads) if step > 0 else 0.0)
			data_logs[_d]['step'].append(step); data_logs[_d]['anls'].append(anls_prog)
			data_logs[_d]['accuracy'].append(acc_prog); data_logs[_d]['output'].append(pred_prog)

	for _it, (_d, _ddict) in enumerate(data_dict.items()):
		data_logs[_d].update({
			"label":_ddict['label'],
			"indices": _ddict['sampled_indices'],
		})

	expt_dir = os.path.join(args.result_dir, expt)
	expt_cfg = f"seed{args.seed}_alpha{str(args.step_size).replace('.','')}_tau{str(args.threshold).replace('.','')}_T{str(args.max_step)}"
	save_result(expt_dir, expt_cfg, data_logs)

	print("RESULT:")
	print(f"IG(alpha={args.step_size},tau={args.threshold},T={args.max_step}): " \
		+json.dumps(evaluate(expt_dir, expt_cfg, args.predictor, FEATURES, ['update'], AGG_FNS, args.seed), indent=4))
	elapsed = datetime.datetime.now() - start_expt
	print(f"Finished!!! time elapsed: {elapsed.seconds//3600} hours, {elapsed.seconds//60%60} mins.")
	print("="*20)

def run_baseline(args, start_model=None, **kwargs):
	sampler = random.Random(args.seed)

	data_dict, all_data = load_data(args.data_dir, args.pilot, args.seed)
	data_logs = {}
	for _d in data_dict:
		data_dict[_d]['sampled_indices'] = sample_indices(all_data, data_dict[_d]['indices'], args.max_query, sampler=sampler)
		data_logs[_d] = {
			"loss": [],  # len(inds)
			"accuracy": [],  # len(inds)
			"anls": [],  # len(inds)
			"output": [],  # len(inds)
		}

	if start_model is None:
		model, prep_fn, forward_fn, eval_fn, processor = load_model(args.model, args.ckpt, freeze=True, run_eval=True)
	else:
		model, prep_fn, forward_fn, eval_fn, processor = start_model, kwargs.get("prep_fn", None), kwargs.get("forward_fn", None), kwargs.get("eval_fn", None), kwargs.get("processor", None)
	model = cuda_and_maybe_copy(model, args.model)

	dloader = get_dataloader(args.data_dir, all_data, data_dict, args.model, args.ckpt)

	start_expt = datetime.datetime.now()
	expt = args.expt if args.expt else f"exp_{start_expt.strftime('%d-%m-%Y-%H-%M-%S')}"

	# run attack
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

			preds, _ = eval_fn(model, _batch, processor=processor, return_probs=True)
			out = forward_fn(model, _batch)
			for _an, _pred in zip(_batch["answers"], preds):
				data_logs[_d]['accuracy'].append(accuracy(_an, _pred))
				data_logs[_d]['anls'].append(anls(_an, _pred))
				data_logs[_d]['output'].append(_pred)
			data_logs[_d]['loss'].append(out.loss.detach().cpu().item())

	for _it, (_d,_ddict) in enumerate(data_dict.items()):
		data_logs[_d].update({
			"label":_ddict['label'],
			"indices": _ddict['sampled_indices'],
		})

	expt_dir = os.path.join(args.result_dir, expt)
	expt_cfg = f"seed{args.seed}_{args.model}"
	save_result(expt_dir, expt_cfg, data_logs)

	print("RESULT:")
	print("SCORE-UA: "\
		+json.dumps(evaluate(expt_dir, expt_cfg, args.predictor, ['accuracy', 'anls'], ['accuracy', 'anls'], ['avg'], args.seed), indent=4))
	print("SCOREALL-UA: "\
		+json.dumps(evaluate(expt_dir, expt_cfg, args.predictor, ['accuracy', 'anls'], ['accuracy', 'anls'], AGG_FNS, args.seed), indent=4))
	print("SCORELOSSALL-UA: "\
		+json.dumps(evaluate(expt_dir, expt_cfg, args.predictor, BASES, BASES, AGG_FNS, args.seed), indent=4))
	elapsed = datetime.datetime.now() - start_expt
	print(f"Finished!!! time elapsed: {elapsed.seconds//3600} hours, {elapsed.seconds//60%60} mins.")
	print("="*20)

def parse_args():
	parser = argparse.ArgumentParser(description='script to run attack')
	parser.add_argument('--attack', type=str, choices=['ig','fl','bl'], help='ig|fl|bl.')
	parser.add_argument('--model', type=str, help='model to attack.')
	parser.add_argument('--ckpt', type=str, help='model checkpoint.')
	parser.add_argument('--data_dir', type=str, help='path to data.')
	parser.add_argument('--pilot', type=int, default=300, help='pilot.')
	parser.add_argument('--max_query', type=int, default=10, help='maximum number of queries.')

	parser.add_argument('--max_step', type=int, default=10, help='maximum number of denoising steps.')
	parser.add_argument('--step_size', type=float, default=1.0, help='step size.')
	parser.add_argument('--threshold', type=float, default=1e-5, help='difference threshold.')
	parser.add_argument('--no_stop_on_delta', default=False, action='store_true', help='NO stop condition.')

	parser.add_argument('--layer', type=str, default=None, help='layer to finetune.')
	parser.add_argument('--lora', default=False, action='store_true', help='low rank adaptation.')

	parser.add_argument('--predictor', type=str, default='kmeans', help='predictor.')

	parser.add_argument('--seed', type=int, default=1027, help='random seed.')
	parser.add_argument('--expt', type=str, default=None, help='experiment name.')
	parser.add_argument('--result_dir', type=str, default='./save/results/', help='path to save result.')

	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	print(args)

	seed_everything(args.seed)

	if args.attack == 'ig':
		run_inputgradient(args)
	elif args.attack == 'fl':
		run_finetunelayer(args)
	elif args.attack == 'bl':
		run_baseline(args)
	else:
		raise ValueError(f"Invalid attack: {args.attack}.")

if __name__ == '__main__':
	main()
