import torch
from torch.nn.utils.rnn import pad_sequence

import transformers
from transformers import DonutProcessor, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from transformers import UdopProcessor, UdopForConditionalGeneration
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from peft import LoraConfig, TaskType
from peft import get_peft_model

from modules.pfl.build_utils import build_model
from .misc import *

def add_lora(model, model_name, modules):
	lora_config = LoraConfig(
		target_modules=modules,
		r=8, lora_alpha=8, lora_dropout=0.05
		# init_lora_weights="gaussian",  # commented by default
	)
	print("With LoRA:")
	if model_name == 'vt5':
		model.model = get_peft_model(model.model, lora_config)
		model.model.print_trainable_parameters()
	elif model_name in ['donut', 'pix2struct', 'pix2struct-large', 'layoutlmv3', 'layoutlmv3-large', 'udop']:
		model = get_peft_model(model, lora_config)
		model.print_trainable_parameters()
	else:
		raise ValueError(f"Invalid model: {model_name}")
	return model

def get_optimizer(model_name, model, lr, finetune=False, **kwargs):
	"""
		finetune=True: finetune model on docvqa dataset (for utility)
		finetune=False: finetune model on blackbox data (for attack)
	"""
	if model_name == 'vt5':
		optimizer, scheduler = torch.optim.AdamW(model.model.parameters(), lr=lr), None
	elif model_name == 'donut':
		optimizer, scheduler = torch.optim.Adam(model.parameters(), lr=lr), None
	elif model_name in ['pix2struct', 'pix2struct-large']:
		if finetune:
			optimizer = transformers.Adafactor(
				model.parameters(), lr=lr, weight_decay=kwargs['weight_decay'], # weight_decay=1e-05,
				scale_parameter=False, relative_step=False
			)
			# num_warmup_steps = len(dset) // bs * num_epoch
			scheduler = get_cosine_schedule_with_warmup(
				optimizer, num_warmup_steps=kwargs['warmup_steps'], num_training_steps=kwargs['max_steps']
			)
			return optimizer, scheduler
		else:
			optimizer, scheduler = torch.optim.AdamW(model.parameters(), lr=lr), None
	elif model_name in ['layoutlmv3', 'layoutlmv3-large']:
		if finetune:
			# NO_DECAY = ["bias", "LayerNorm.weight"]
			# optimizer_grouped_parameters = [
			#	 {
			#		 "params": [
			#		 	p for n, p in model.named_parameters()
			#		 	if not any(nd in n for nd in NO_DECAY)
			#		 ],
			#		 "weight_decay": kwargs['weight_decay'],
			#	 },
			#	 {
			#		 "params": [
			#		 	p for n, p in model.named_parameters()
			#		 	if any(nd in n for nd in NO_DECAY)
			#		 ],
			#		 "weight_decay": 0.0,
			#	 },
			# ]
			optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
			scheduler = get_linear_schedule_with_warmup(
				optimizer, num_warmup_steps=kwargs['warmup_steps'], num_training_steps=kwargs['max_steps']
			)
			return optimizer, scheduler
		else:
			optimizer, scheduler = torch.optim.AdamW(model.parameters(), lr=lr), None
	elif model_name == 'udop':
		# if finetune:
		# 	optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=kwargs['weight_decay']) # , weight_decay=1e-2
		# 	scheduler = get_linear_schedule_with_warmup(
		# 		optimizer, num_warmup_steps=kwargs['warmup_steps'], num_training_steps=kwargs['max_steps']
		# 	)
		# 	return optimizer, scheduler
		if finetune:
			optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , weight_decay=1e-2
			return optimizer, None
	else:
		raise ValueError(f"Invalid model: {model_name}.")
	return optimizer, scheduler

def freeze_params(model, layer=None, verbose=False):
	"""
	Freeze all params except from layer:
		- model: expected (not required) at train mode
	"""
	for _n, _p in model.named_parameters():
		if not layer or not any([_l in _n for _l in layer]):
			_p.requires_grad = False
	if verbose:
		total, n_train = count_params(model)
		print('>'*5 + f" Model parameters: {total}, trainable: {n_train}({n_train*100/total:.2f}%)")

def count_params(model):
	num_param = sum(p.numel() for p in model.parameters())
	num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return num_param, num_trainable

def save_model(save_dir, model_name, model, processor, epoch, best=False):
	if model_name == 'vt5':
		model.model.save_pretrained(os.path.join(save_dir, "last.ckpt"))
		processor.save_pretrained(os.path.join(save_dir, "last.ckpt"))
		if best:
			model.model.save_pretrained(os.path.join(save_dir, f"best_epoch{epoch}.ckpt"))
			processor.save_pretrained(os.path.join(save_dir, f"best_epoch{epoch}.ckpt"))
			for _prev_epoch in range(epoch):
				_prev_epoch_dir = os.path.join(save_dir, f"best_epoch{_prev_epoch}.ckpt")
				if os.path.exists(_prev_epoch_dir):
					shutil.rmtree(_prev_epoch_dir)
	elif model_name in ['donut', 'pix2struct', 'pix2struct-large', 'layoutlmv3', 'layoutlmv3-large', 'udop']:
		model.save_pretrained(os.path.join(save_dir, "last.ckpt"))
		processor.save_pretrained(os.path.join(save_dir, "last.ckpt"))
		if best:
			model.save_pretrained(os.path.join(save_dir, f"best_epoch{epoch}.ckpt"))
			processor.save_pretrained(os.path.join(save_dir, f"best_epoch{epoch}.ckpt"))
			for _prev_epoch in range(epoch):
				_prev_epoch_dir = os.path.join(save_dir, f"best_epoch{_prev_epoch}.ckpt")
				if os.path.exists(_prev_epoch_dir):
					shutil.rmtree(_prev_epoch_dir)
	else:
		raise ValueError(f"Invalid model: {model_name}.")

def load_model(model, model_ckpt, freeze=False, run_eval=False):
	if model == 'vt5':
		model = init_vt5(model_ckpt)
		if freeze: freeze_params(model.model)
		if run_eval:
			return model, vt5_preprocess, vt5_forward, vt5_eval, model.tokenizer
		return model, vt5_preprocess, vt5_forward, model.tokenizer
	elif model == 'donut':
		model, processor = init_donut(model_ckpt,
			lowreso=(model_ckpt!='naver-clova-ix/donut-base-finetuned-docvqa')
		)
		if freeze: freeze_params(model)
		if run_eval:
			return model, donut_preprocess, donut_forward, donut_eval, processor
		return model, donut_preprocess, donut_forward, processor
	elif model in ['pix2struct', 'pix2struct-large']:
		model, processor = init_pix2struct(model_ckpt)
		if freeze: freeze_params(model)
		if run_eval:
			return model, pix2struct_preprocess, pix2struct_forward, pix2struct_eval, processor
		return model, pix2struct_preprocess, pix2struct_forward, processor
	elif model in ['layoutlmv3', 'layoutlmv3-large']:
		model, processor = init_layoutlmv3(model_ckpt)
		if freeze: freeze_params(model)
		if run_eval:
			return model, layoutlmv3_preprocess, layoutlmv3_forward, layoutlmv3_eval, processor
		return model, layoutlmv3_preprocess, layoutlmv3_forward, processor
	elif model == 'udop':
		model, processor = init_udop(model_ckpt)
		if freeze: freeze_params(model)
		if run_eval:
			return model, udop_preprocess, udop_forward, udop_eval, processor
		return model, udop_preprocess, udop_forward, processor
	else:
		raise ValueError(f"Invalid model: {model}")

def cuda_and_maybe_copy(model, model_name, maybe_copy=False):
	if maybe_copy: model = copy.deepcopy(model)
	if model_name == 'vt5':
		model.model.cuda()
	elif model_name in ['donut', 'pix2struct', 'pix2struct-large', 'layoutlmv3', 'layoutlmv3-large', 'udop']:
		model.cuda()
	else:
		raise ValueError(f"Invalid model: {model_name}")
	return model

########## VT5 ##########
def init_vt5(ckpt):
	visual_module_config = {'finetune': False, 'model': 'dit', 'model_weights': 'microsoft/dit-base-finetuned-rvlcdip'}
	experiment_config = {'model_weights': ckpt, 'max_input_tokens': 512, 'device': 'cuda', 'visual_module': visual_module_config}
	model = build_model('vt5', experiment_config)
	return model

def vt5_preprocess(batch, i):
	# To create a batch of 1 training data
	_batch = {_k:batch[_k][i:i+1] for _k in batch if _k not in ['images']}
	_batch['images'] = {
		'pixel_values': batch['images']['pixel_values'][i][None, :],
	}
	return _batch

def vt5_forward(model, batch, mode='train'):
	if mode == 'train':
		model.model.train()
	else:
		model.model.eval()
	out, _, _ = model.forward(batch)
	return out

def vt5_eval(model, batch, return_probs=False, **kwargs):
	model.model.eval()
	with torch.no_grad():
		_, preds, (_, probs) = model.forward(batch, return_pred_answer=True)
	if return_probs:
		return preds, probs
	return preds

def vt5_inference(ckpt, data_dir, seed=None, dset='pfl', attack=None, question_type='v0'):
	model = init_vt5(ckpt)
	model.model.cuda()

	if not attack:
		if 'pfl' in dset:
			if dset == 'pfl_v1.2':
				data_dir = '/data/shared/DocVQA_1.2/imdb/'
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, 'pfl_test.npy'), allow_pickle=True)[1:]
		elif dset == 'docvqa':
			data_dir, im_ext = '/data/users/vkhanh/due/docvqa', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)
		elif dset == 'docvqav0':
			data_dir, im_ext = '/data/shared/docvqav0', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)[1:]
	else:
		pilot_or_full = data_dir if seed is None  else os.path.join(data_dir, f'pilot/seed{seed}')
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, f'pfl_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))
		elif 'docvqa' in dset:
			im_ext = '.png'
			vqa_data = np.load(os.path.join(data_dir, f'{dset}_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))

	score_dict = dict()
	for _ind, _record in tqdm(enumerate(vqa_data)):
		doc_id = _record['image_name'].split('_')[0] if 'pfl' in dset else _record['image_name']
		if attack and doc_id not in data_dict: continue;
		if dset == 'pfl_v1.2':
			image = Image.open(os.path.join('/data/shared/PFL-DocVQA/images', _record['image_name'] + im_ext)).convert("RGB")
		else:
			image = Image.open(os.path.join(data_dir, 'images' , _record['image_name'] + im_ext)).convert("RGB")
		words = [word.lower() for word in _record['ocr_tokens']]
		boxes = np.array([bbox for bbox in _record['ocr_normalized_boxes']])

		if not attack:
			question = _record['question']
		else:
			if question_type == 'v0':
				if _ind not in data_dict[doc_id]['indices']: continue;
				else:
					question = _record['question']
			else:
				if str(_ind) not in data_dict[doc_id]['rephrased']: continue;
				else:
					question = data_dict[doc_id]['rephrased'][str(_ind)]

		_batch = [{
			'question_ids': _ind,
			'questions': question,
			'contexts': " ".join([_t.lower() for _t in _record['ocr_tokens']]),
			'answers': [_record['answers'].lower()] if dset == 'pfl' else _record['answers'],
			'image_names': _record['image_name'],
			'images': model.model.visual_embedding.feature_extractor(images=image, return_tensors="pt"),
			'words': words,
			'boxes': boxes,
		}]
		batch = {
			_k: [_d[_k] for _d in _batch] if _k != 'images'
			else {'pixel_values': torch.stack([_d[_k]['pixel_values'].squeeze(0) for _d in _batch])}
			for _k in _batch[0]
		}
		preds = vt5_eval(model, batch)
		out = vt5_forward(model, batch)
		score_dict[_ind] = {
			"doc_id": doc_id,
			"accuracy": accuracy(batch["answers"][0], preds[0]),
			"anls": anls(batch["answers"][0], preds[0]),
			"answers": batch["answers"][0],
			"predict": preds[0],
		}
		if attack:
			score_dict[_ind]["label"] = data_dict[doc_id]['label']
	if not attack:
		print(f"ACC={np.mean([_v['accuracy'] for _k,_v in score_dict.items()]):.4f}, "+\
			f"ANLS={np.mean([_v['anls'] for _k,_v in score_dict.items()]):.4f}")
	else:
		acc = np.mean([_v['accuracy'] for _k,_v in score_dict.items()])
		acc1 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 1])
		acc0 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 0])
		an = np.mean([_v['anls'] for _k,_v in score_dict.items()])
		an1 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 1])
		an0 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 0])
		print(f"ATTACK={attack}, question={question_type}: "+\
			f"ACC={acc:.4f}({acc1:.4f}/{acc0:.4f}), "+\
			f"ANLS={an:.4f}({an1:.4f}/{an0:.4f})")
	return score_dict

########## DONUT ##########
def init_donut(ckpt, lowreso=False):
	if lowreso:
		# Lower resolution for stable training
		max_length = 128
		image_size = [1280, 960]

		# during pre-training, a LARGER image size was used [2560, 1920]
		config = VisionEncoderDecoderConfig.from_pretrained(ckpt)
		config.encoder.image_size = image_size
		config.decoder.max_length = max_length
		model = VisionEncoderDecoderModel.from_pretrained(ckpt, config=config)

		# TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
		# https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602
		image_processor = DonutProcessor.from_pretrained(ckpt)
		image_processor.feature_extractor.size = image_size[::-1] # REVERSED (width, height)
		image_processor.feature_extractor.do_align_long_axis = False

		image_processor.tokenizer.add_tokens([
			"<s_docvqa>", "<yes/>", "<no/>", "<s_question>", "<s_answer>", "</s_answer>", "</s_question>"
		])
		model.decoder.resize_token_embeddings(len(image_processor.tokenizer))
	else:
		# Follow the exact model's config
		image_processor = DonutProcessor.from_pretrained(ckpt)
		model = VisionEncoderDecoderModel.from_pretrained(ckpt)
	return model, image_processor

def donut_preprocess(batch, i):
	# To create a batch of 1 training data
	_batch = {_k:batch[_k][i:i+1] for _k in batch if _k not in ['images', 'text_input_ids', 'prompt_input_ids', 'labels']}
	_batch['images'] = {
		'pixel_values': batch['images']['pixel_values'][i][None, :],
	}
	_batch['text_input_ids'] = batch['text_input_ids'][i][None, :]
	_batch['prompt_input_ids'] = batch['prompt_input_ids'][i][None, :]
	_batch['labels'] = batch['labels'][i][None, :]
	return _batch

def donut_forward(model, batch, mode='train'):
	if mode == 'train':
		model.train()
	else:
		model.eval()
	return model(
		pixel_values=batch['images']['pixel_values'],
		decoder_input_ids=batch['text_input_ids'][:, :-1],
		labels=batch['labels'][:, 1:]
	)

def donut_eval(model, batch, processor, return_probs=False):
	model.eval()
	if len(batch['questions']) == 1:
		with torch.no_grad():
			outs = model.generate(
				pixel_values=batch['images']['pixel_values'],
				decoder_input_ids=batch['prompt_input_ids'][0].unsqueeze(0),
				max_length=128,
				early_stopping=True,
				pad_token_id=processor.tokenizer.pad_token_id,
				eos_token_id=processor.tokenizer.eos_token_id,
				use_cache=True,
				num_beams=1,
				bad_words_ids=[[processor.tokenizer.unk_token_id]],
				output_scores=return_probs,
				return_dict_in_generate=True,
			)
		decoded = processor.batch_decode(outs.sequences)
		if return_probs:
			logits = torch.stack(outs.scores, dim=1)
			probs = torch.amax(logits.softmax(dim=-1), dim=-1).prod(dim=-1).tolist()
	else:
		decoded, probs = [], []
		for _i in range(len(batch['questions'])):
			with torch.no_grad():
				outs = model.generate(
					pixel_values=batch['images']['pixel_values'][_i].unsqueeze(0),
					decoder_input_ids=batch['prompt_input_ids'][_i].unsqueeze(0),
					max_length=128,
					early_stopping=True,
					pad_token_id=processor.tokenizer.pad_token_id,
					eos_token_id=processor.tokenizer.eos_token_id,
					use_cache=True,
					num_beams=1,
					bad_words_ids=[[processor.tokenizer.unk_token_id]],
					output_scores=return_probs,
					return_dict_in_generate=True,
				)
			decoded.append(processor.batch_decode(outs.sequences)[0])
			if return_probs:
				logits = torch.stack(outs.scores, dim=1)
				probs.append(
					torch.amax(logits.softmax(dim=-1), dim=-1).prod(dim=-1).tolist()[0]
				)
	preds = []
	for _dec in decoded:
		_no_eos_pad = _dec.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
		_no_task = re.sub(r"<.*?>", "", _no_eos_pad, count=1).strip()
		_json = processor.token2json(_no_task)
		preds.append(_json['answer'] if 'answer' in _json and isinstance(_json['answer'], str) else '')
	if return_probs:
		return preds, probs
	return preds

def donut_inference(ckpt, data_dir, seed=None, dset='docvqav0', attack=None, question_type='v0', lowreso=False):
	model, image_processor = init_donut(ckpt, lowreso=lowreso)
	model.cuda(); model.eval()

	if not attack:
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, 'pfl_test.npy'), allow_pickle=True)[1:]
		elif dset == 'docvqa':
			data_dir, im_ext = '/data/users/vkhanh/due/docvqa', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)
		elif dset == 'docvqav0':
			data_dir, im_ext = '/data/shared/docvqav0', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)[1:]
	else:
		pilot_or_full = data_dir if seed is None  else os.path.join(data_dir, f'pilot/seed{seed}')
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, f'pfl_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))
		elif 'docvqa' in dset:
			im_ext = '.png'
			vqa_data = np.load(os.path.join(data_dir, f'{dset}_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))

	score_dict = dict()
	for _ind, _record in tqdm(enumerate(vqa_data)):
		doc_id = _record['image_name'].split('_')[0] if 'pfl' in dset else _record['image_name']
		if attack and doc_id not in data_dict: continue;
		if dset == 'pfl_v1.2':
			image = Image.open(os.path.join('/data/shared/PFL-DocVQA/images', _record['image_name'] + im_ext)).convert("RGB")
		else:
			image = Image.open(os.path.join(data_dir, 'images', _record['image_name'] + im_ext)).convert("RGB")
		image_tensor = image_processor(image, return_tensors="pt")
		if not attack:
			question = _record['question']
		else:
			if question_type == 'v0':
				if _ind not in data_dict[doc_id]['indices']: continue;
				else:
					question = _record['question']
			else:
				if str(_ind) not in data_dict[doc_id]['rephrased']: continue;
				else:
					question = data_dict[doc_id]['rephrased'][str(_ind)]
		answers = [_record['answers']] if dset == 'pfl' else _record['answers']

		prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
		prompt_input_ids = image_processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

		_batch = {
			"images": image,
			"prompts": prompt,
			"questions": question,
			"answers": answers,
			'image_names': _record['image_name'],
		}
		with torch.no_grad():
			outs = model.generate(
				image_tensor['pixel_values'].cuda(),
				decoder_input_ids=prompt_input_ids.cuda(),
				max_length=model.decoder.config.max_position_embeddings,
				pad_token_id=image_processor.tokenizer.pad_token_id,
				eos_token_id=image_processor.tokenizer.eos_token_id,
				use_cache=True,
				bad_words_ids=[[image_processor.tokenizer.unk_token_id]],
				return_dict_in_generate=True,
			)
		decoded = image_processor.batch_decode(outs.sequences)
		_no_eos_pad = decoded[0].replace(image_processor.tokenizer.eos_token, "").replace(image_processor.tokenizer.pad_token, "")
		_no_task = re.sub(r"<.*?>", "", _no_eos_pad, count=1).strip()
		_json = image_processor.token2json(_no_task)
		score_dict[_ind] = {
			"accuracy": accuracy(_batch["answers"], _json["answer"]) if "answer" in _json and isinstance(_json['answer'], str) else 0.0,
			"anls": anls(_batch["answers"], _json["answer"]) if "answer" in _json and isinstance(_json['answer'], str) else 0.0,
			"answers": _batch["answers"],
			"predict": _json["answer"] if "answer" in _json and isinstance(_json['answer'], str) else '',
		}
		if attack:
			score_dict[_ind]["label"] = data_dict[doc_id]['label']
	if not attack:
		print(f"ACC={np.mean([_v['accuracy'] for _k,_v in score_dict.items()]):.4f}, "+\
			f"ANLS={np.mean([_v['anls'] for _k,_v in score_dict.items()]):.4f}")
	else:
		acc = np.mean([_v['accuracy'] for _k,_v in score_dict.items()])
		acc1 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 1])
		acc0 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 0])
		an = np.mean([_v['anls'] for _k,_v in score_dict.items()])
		an1 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 1])
		an0 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 0])
		print(f"ATTACK={attack}, question={question_type}: "+\
			f"ACC={acc:.4f}({acc1:.4f}/{acc0:.4f}), "+\
			f"ANLS={an:.4f}({an1:.4f}/{an0:.4f})")
	return score_dict

########## PIX2STRUCT ##########
def init_pix2struct(ckpt):
	model = Pix2StructForConditionalGeneration.from_pretrained(ckpt)
	image_processor = Pix2StructProcessor.from_pretrained(ckpt)
	return model, image_processor

def pix2struct_preprocess(batch, i):
	# To create a batch of 1 training data
	_batch = {_k:batch[_k][i:i+1] for _k in batch if _k not in ['images', 'labels']}
	_batch['images'] = {
		'pixel_values': batch['images']['pixel_values'][i][None, :],
		'attention_mask': batch['images']['attention_mask'][i][None, :],
	}
	_batch["labels"] = batch['labels'][i][None, :]
	return _batch

def pix2struct_forward(model, batch, mode='train'):
	if mode == 'train':
		model.train()
	else:
		model.eval()
	return model(
		flattened_patches=batch['images']['pixel_values'],
		attention_mask=batch['images']['attention_mask'],
		labels=batch['labels']
	)

def pix2struct_eval(model, batch, processor, return_probs=False):
	model.eval()
	with torch.no_grad():
		outs = model.generate(
			flattened_patches=batch['images']['pixel_values'],
			attention_mask=batch['images']['attention_mask'],
			max_new_tokens=256,
			output_scores=return_probs,
			return_dict_in_generate=True,
		)
	preds = processor.batch_decode(outs.sequences, skip_special_tokens=True)
	if return_probs:
		logits = torch.stack(outs.scores, dim=1)
		probs = torch.amax(logits.softmax(dim=-1), dim=-1).prod(dim=-1)
		return preds, probs.tolist()
	return preds

def pix2struct_inference(ckpt, data_dir, seed=None, dset='docvqav0', attack=None, question_type='v0'):
	model, processor = init_pix2struct(ckpt)
	model.cuda(); model.eval()

	if not attack:
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, 'pfl_test.npy'), allow_pickle=True)[1:]
		elif dset == 'docvqa':
			data_dir, im_ext = '/data/users/vkhanh/due/docvqa', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)
		elif dset == 'docvqav0':
			data_dir, im_ext = '/data/shared/docvqav0', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)[1:]
	else:
		pilot_or_full = data_dir if seed is None  else os.path.join(data_dir, f'pilot/seed{seed}')
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, f'pfl_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))
		elif 'docvqa' in dset:
			im_ext = '.png'
			vqa_data = np.load(os.path.join(data_dir, f'{dset}_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))

	score_dict = dict()
	for _ind, _record in tqdm(enumerate(vqa_data)):
		doc_id = _record['image_name'].split('_')[0] if 'pfl' in dset else _record['image_name']
		if attack and doc_id not in data_dict: continue;
		image = Image.open(os.path.join(data_dir, 'images', _record['image_name'] + im_ext)).convert("RGB")
		if not attack:
			question = _record['question']
		else:
			if question_type == 'v0':
				if _ind not in data_dict[doc_id]['indices']: continue;
				else:
					question = _record['question']
			else:
				if str(_ind) not in data_dict[doc_id]['rephrased']: continue;
				else:
					question = data_dict[doc_id]['rephrased'][str(_ind)]
		answers = [_record['answers']] if dset == 'pfl' else _record['answers']

		inputs = processor(images=image, text=question, return_tensors="pt").to("cuda")
		with torch.no_grad():
			outs = model.generate(**inputs, max_new_tokens=50)
		pred = processor.batch_decode(outs, skip_special_tokens=True)[0]
		score_dict[_ind] = {
			"accuracy": accuracy(answers, pred),
			"anls": anls(answers, pred),
			"answers": answers,
			"predict": pred,
		}
		if attack:
			score_dict[_ind]["label"] = data_dict[doc_id]['label']
	if not attack:
		print(f"ACC={np.mean([_v['accuracy'] for _k,_v in score_dict.items()]):.4f}, "+\
			f"ANLS={np.mean([_v['anls'] for _k,_v in score_dict.items()]):.4f}")
	else:
		acc = np.mean([_v['accuracy'] for _k,_v in score_dict.items()])
		acc1 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 1])
		acc0 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 0])
		an = np.mean([_v['anls'] for _k,_v in score_dict.items()])
		an1 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 1])
		an0 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 0])
		print(f"ATTACK={attack}, question={question_type}: "+\
			f"ACC={acc:.4f}({acc1:.4f}/{acc0:.4f}), "+\
			f"ANLS={an:.4f}({an1:.4f}/{an0:.4f})")
	return score_dict

########## UDOP ##########
def init_udop(ckpt):
	processor = UdopProcessor.from_pretrained(ckpt, apply_ocr=False)
	model = UdopForConditionalGeneration.from_pretrained(ckpt)
	return model, processor

def udop_preprocess(batch, i):
	_batch = {_k:batch[_k][i:i+1] for _k in batch if _k not in ['images', 'input_ids', 'attention_mask', 'bbox', 'labels']}
	_batch['images'] = {
		'pixel_values': batch['images']['pixel_values'][i][None, :],
	}
	_batch["input_ids"] = batch['input_ids'][i][None, :]
	_batch["attention_mask"] = batch['attention_mask'][i][None, :]
	_batch["bbox"] = batch['bbox'][i][None, :]
	_batch["labels"] = batch['labels'][i][None, :]
	return _batch

def udop_forward(model, batch, mode='train'):
	if mode == 'train':
		model.train()
	else:
		model.eval()
	return model(
		pixel_values=batch['images']['pixel_values'],
		input_ids=batch['input_ids'],
		attention_mask=batch['attention_mask'],
		bbox=batch['bbox'],
		labels=batch['labels']
	 )

def udop_eval(model, batch, processor, return_probs=False):
	model.eval()
	with torch.no_grad():
		outs = model.generate(
			pixel_values=batch['images']['pixel_values'],
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			bbox=batch['bbox'],
			max_new_tokens=256,
			output_scores=return_probs,
			return_dict_in_generate=True,
		)
	preds = processor.batch_decode(outs.sequences, skip_special_tokens=True)
	if return_probs:
		logits = torch.stack(outs.scores, dim=1)
		probs = torch.amax(logits.softmax(dim=-1), dim=-1).prod(dim=-1)
		return preds, probs.tolist()
	return preds

def udop_inference(ckpt, data_dir, seed=None, dset='docvqa', attack=None, question_type='v0'):
	model, processor = init_udop(ckpt)
	model.cuda(); model.eval()

	if not attack:
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, 'pfl_test.npy'), allow_pickle=True)[1:]
		elif dset == 'docvqa':
			data_dir, im_ext = '/data/users/vkhanh/due/docvqa', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)
		elif dset == 'docvqav0':
			data_dir, im_ext = '/data/shared/docvqav0', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)[1:]
	else:
		pilot_or_full = data_dir if seed is None  else os.path.join(data_dir, f'pilot/seed{seed}')
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, f'pfl_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))
		elif 'docvqa' in dset:
			im_ext = '.png'
			vqa_data = np.load(os.path.join(data_dir, f'{dset}_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))

	score_dict = dict(); skip = 0
	for _ind, _record in tqdm(enumerate(vqa_data)):
		doc_id = _record['image_name'].split('_')[0] if 'pfl' in dset else _record['image_name']
		if attack and doc_id not in data_dict: continue;
		if dset == 'pfl_v1.2':
			image = Image.open(os.path.join('/data/shared/PFL-DocVQA/images', _record['image_name'] + im_ext)).convert("RGB")
		else:
			image = Image.open(os.path.join(data_dir, 'images' , _record['image_name'] + im_ext)).convert("RGB")

		if not attack:
			question = _record['question']
		else:
			if question_type == 'v0':
				if _ind not in data_dict[doc_id]['indices']: continue;
				else:
					question = _record['question']
			else:
				if str(_ind) not in data_dict[doc_id]['rephrased']: continue;
				else:
					question = data_dict[doc_id]['rephrased'][str(_ind)]
		words = _record['ocr_tokens']
		boxes = np.array([bbox for bbox in _record['ocr_normalized_boxes']])
		answers = [_record['answers']] if dset == 'pfl' else _record['answers']
		try:
			inputs = processor(image, "Question Answering. "+question, words, boxes=boxes, return_tensors="pt").to("cuda")
		except ValueError as e:
			skip += 1
			continue

		with torch.no_grad():
			outs = model.generate(**inputs, max_new_tokens=256)

		pred = processor.batch_decode(outs, skip_special_tokens=True)[0]
		score_dict[_ind] = {
			"doc_id": doc_id,
			"accuracy": accuracy(answers, pred),
			"anls": anls(answers, pred),
			"answers": answers,
			"predict": pred,
		}
		if attack:
			score_dict[_ind]["label"] = data_dict[doc_id]['label']
	print(f"Skipped={skip}")
	if not attack:
		print(f"ACC={np.mean([_v['accuracy'] for _k,_v in score_dict.items()]):.4f}, "+\
			f"ANLS={np.mean([_v['anls'] for _k,_v in score_dict.items()]):.4f}")
	else:
		acc = np.mean([_v['accuracy'] for _k,_v in score_dict.items()])
		acc1 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 1])
		acc0 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 0])
		an = np.mean([_v['anls'] for _k,_v in score_dict.items()])
		an1 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 1])
		an0 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 0])
		print(f"ATTACK={attack}, question={question_type}: "+\
			f"ACC={acc:.4f}({acc1:.4f}/{acc0:.4f}), "+\
			f"ANLS={an:.4f}({an1:.4f}/{an0:.4f})")
	return score_dict

########## LayoutLMv3 ##########
def init_layoutlmv3(ckpt):
	processor = LayoutLMv3Processor.from_pretrained(ckpt, apply_ocr=False)
	model = LayoutLMv3ForQuestionAnswering.from_pretrained(ckpt)
	return model, processor

def layoutlmv3_preprocess(batch, i):
	# To create a batch of 1 training data
	_batch = {_k:batch[_k][i:i+1] for _k in batch if _k not in ['images', 'input_ids', 'attention_mask', 'boxes', 'start_positions', 'end_positions']}
	_batch['images'] = {'pixel_values': batch['images']['pixel_values'][i][None, :]}
	_batch['input_ids'] = batch['input_ids'][i][None, :]
	_batch['attention_mask'] = batch['attention_mask'][i][None, :]
	_batch['boxes'] = batch['boxes'][i][None, :]
	_batch['start_positions'] = batch['start_positions'][i][None, :]
	_batch['end_positions'] = batch['end_positions'][i][None, :]
	return _batch

def layoutlmv3_forward(model, batch, mode='train'):
	if mode == 'train':
		model.train()
	else:
		model.eval()
	return model(
		pixel_values=batch['images']['pixel_values'],
		input_ids=batch['input_ids'],
		attention_mask=batch['attention_mask'],
		bbox=batch['boxes'],
		start_positions=batch['start_positions'], end_positions=batch['end_positions']
	)

def layoutlmv3_eval(model, batch, processor):
	model.eval()
	with torch.no_grad():
		outs = model(
			pixel_values=batch['images']['pixel_values'],
			input_ids=batch['input_ids'],
			attention_mask=batch['attention_mask'],
			bbox=batch['boxes'],
			start_positions=None, end_positions=None
		)
	starts = torch.argmax(outs.start_logits, axis=1)
	ends = torch.argmax(outs.end_logits, axis=1)
	return [
		processor.tokenizer.decode(
			batch['input_ids'][_i][starts[_i]:ends[_i]], skip_special_tokens=True
		).strip()
		for _i in range(len(batch['input_ids']))
	]

def layoutlmv3_inference(ckpt, data_dir, seed=None, dset='docvqav0', attack=None, question_type='v0'):
	model, processor = init_layoutlmv3(ckpt)
	model.cuda(); model.eval()

	if not attack:
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, 'pfl_test.npy'), allow_pickle=True)[1:]
		elif dset == 'docvqa':
			data_dir, im_ext = '/data/users/vkhanh/due/docvqa', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)
		elif dset == 'docvqav0':
			data_dir, im_ext = '/data/shared/docvqav0', '.png'
			vqa_data = np.load(os.path.join(data_dir, 'test', 'vqa.npy'), allow_pickle=True)[1:]
	else:
		pilot_or_full = data_dir if seed is None  else os.path.join(data_dir, f'pilot/seed{seed}')
		if dset == 'pfl':
			im_ext = '.jpg'
			vqa_data = np.load(os.path.join(data_dir, f'pfl_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))
		elif 'docvqa' in dset:
			im_ext = '.png'
			vqa_data = np.load(os.path.join(data_dir, f'{dset}_{attack}.npy'), allow_pickle=True)
			data_dict = json.load(open(os.path.join(pilot_or_full, f'pilot300_{attack}_{question_type}.json'), 'r'))

	score_dict = dict()
	for _ind, _record in tqdm(enumerate(vqa_data)):
		doc_id = _record['image_name'].split('_')[0] if 'pfl' in dset else _record['image_name']
		if attack and doc_id not in data_dict: continue;
		image = Image.open(os.path.join(data_dir, 'images', _record['image_name'] + im_ext)).convert("RGB")
		if not attack:
			question = _record['question']
		else:
			if question_type == 'v0':
				if _ind not in data_dict[doc_id]['indices']: continue;
				else:
					question = _record['question']
			else:
				if str(_ind) not in data_dict[doc_id]['rephrased']: continue;
				else:
					question = data_dict[doc_id]['rephrased'][str(_ind)]

		question = question.lower()
		answers = [_record['answers'].lower] if dset == 'pfl' else [_an.lower() for _an in _record['answers']]
		words = [_w.lower() for _w in _record["ocr_tokens"]]
		context = ' '.join(words)
		boxes = [list((_b*1000).astype(int)) for _b in _record["ocr_normalized_boxes"]]

		try:
			encoding = processor(
				image, question, words, boxes=boxes,
				padding=True, truncation=True, return_tensors="pt"
			)
		except ValueError:
			continue
		_batch = {}
		_batch["images"] = {'pixel_values': encoding['pixel_values'].cuda()}
		_batch['input_ids'] = encoding['input_ids'].cuda()
		_batch['attention_mask'] = encoding['attention_mask'].cuda()
		_batch['boxes'] = encoding['bbox'].cuda()

		with torch.no_grad():
			outs = model(
				pixel_values=_batch['images']['pixel_values'],
				input_ids=_batch['input_ids'],
				attention_mask=_batch['attention_mask'],
				bbox=_batch['boxes'],
				start_positions=None, end_positions=None
			)
		starts = torch.argmax(outs.start_logits, axis=1)
		ends = torch.argmax(outs.end_logits, axis=1)
		prediction = processor.tokenizer.decode(
			_batch['input_ids'][0][starts[0]:ends[0]], skip_special_tokens=True
		).strip()

		score_dict[_ind] = {
			"accuracy": accuracy(answers, prediction),
			"anls": anls(answers, prediction),
			"answers": answers,
			"predict": prediction,
		}
		if attack:
			score_dict[_ind]["label"] = data_dict[doc_id]['label']
	if not attack:
		print(f"ACC={np.mean([_v['accuracy'] for _k,_v in score_dict.items()]):.4f}, "+\
			f"ANLS={np.mean([_v['anls'] for _k,_v in score_dict.items()]):.4f}")
	else:
		acc = np.mean([_v['accuracy'] for _k,_v in score_dict.items()])
		acc1 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 1])
		acc0 = np.mean([_v['accuracy'] for _k,_v in score_dict.items() if _v['label'] == 0])
		an = np.mean([_v['anls'] for _k,_v in score_dict.items()])
		an1 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 1])
		an0 = np.mean([_v['anls'] for _k,_v in score_dict.items() if _v['label'] == 0])
		print(f"ATTACK={attack}, question={question_type}: "+\
			f"ACC={acc:.4f}({acc1:.4f}/{acc0:.4f}), "+\
			f"ANLS={an:.4f}({an1:.4f}/{an0:.4f})")
	return score_dict
