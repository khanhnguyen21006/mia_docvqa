import torch
from torch.utils.data import Dataset, Subset, DataLoader

from .misc import *
from .model import init_vt5, init_donut, init_pix2struct, init_layoutlmv3, init_udop

def get_start_end(encoding, context, answers, processor):
	pos_idx = []
	for batch_idx in range(len(encoding.input_ids)):
		answer_pos = []
		for answer in answers[batch_idx]:
			encoded_answer = [token for token in processor.tokenizer.encode([answer], boxes=[0, 0, 0, 0]) if token not in processor.tokenizer.all_special_ids]
			answer_tokens_length = len(encoded_answer)

			for token_pos in range(len(encoding.input_ids[batch_idx])):
				if encoding.input_ids[batch_idx][token_pos: token_pos+answer_tokens_length].tolist() == encoded_answer:
					answer_pos.append([token_pos, token_pos + answer_tokens_length])

		if len(answer_pos) == 0:
			pos_idx.append([9999, 9999])
		else:
			answer_pos = random.choice(answer_pos)  # To add variability, pick a random correct span.
			pos_idx.append(answer_pos)

	start_idxs = torch.LongTensor([idx[0] for idx in pos_idx])
	end_idxs = torch.LongTensor([idx[1] for idx in pos_idx])
	return start_idxs, end_idxs

def load_data(data_dir, pilot, seed, question='v0'):
	dset = os.path.basename(os.path.normpath(data_dir))
	assert dset in ['docvqa', 'docvqav0', 'pfl'], ValueError(f"Invalid dataset: {dset}.")
	if'pfl' == dset:
		return load_data_pfl(data_dir, pilot, seed, question)
	elif 'docvqav0' == dset:
		# This is the offifcial DocVQA
		return load_data_docvqa(data_dir, pilot, seed, question, ver='v0')
	elif 'docvqa' == dset:
		# By default we use DocVQA data from DUE
		return load_data_docvqa(data_dir, pilot, seed, question)
	else:
		raise ValueError(f"Invalid data: {data_dir}")

def load_data_pfl(data_dir, pilot, seed, question='v0'):
	DOCVQA_DATA = np.load(os.path.join(data_dir, 'pfl_mia.npy'), allow_pickle=True)
	data_dict = json.load(open(os.path.join(data_dir, 'pfl_mia.json'), 'r'))

	if pilot:
		data_dict = json.load(open(os.path.join(data_dir, f'pilot/seed{seed}', 'pfl_mia.json'), 'r'))
		data_dict = create_pilot(data_dir, data_dict, seed, question)
	return data_dict, DOCVQA_DATA
	
def load_data_docvqa(data_dir, pilot, seed, question='v0', ver=''):
	DOCVQA_DATA = np.load(os.path.join(data_dir, f'docvqa{ver}_mia.npy'), allow_pickle=True)
	data_dict = json.load(open(os.path.join(data_dir, f'docvqa{ver}_mia.json'), 'r'))

	if pilot:
		DOCVQA_DATA = np.load(os.path.join(data_dir, f'pilot/seed{seed}', f'docvqa{ver}_mia.npy'), allow_pickle=True)
		data_dict = json.load(open(os.path.join(data_dir, f'pilot/seed{seed}', f'docvqa{ver}_mia.json'), 'r'))
		data_dict = create_pilot(data_dir, data_dict, seed, question)
	return data_dict, DOCVQA_DATA

def get_dataloader(data_dir, all_data, data_dict, model_name, model_ckpt, question='v0', outputs=None, batch_size=8, shuffle=False):
	dset = MIDataset(all_data, data_dict, data_dir, question=question, outputs=outputs)
	if model_name == 'vt5':
		model = init_vt5(model_ckpt)
		collate_fn = partial(dset.collate_fn, model_name=model_name, model=model)
	elif model_name == 'donut':
		_, processor = init_donut(model_ckpt, lowreso=(model_ckpt!='naver-clova-ix/donut-base-finetuned-docvqa'))
		collate_fn = partial(dset.collate_fn, model_name=model_name, processor=processor)
	elif model_name in ['pix2struct', 'pix2struct-large']:
		_, processor = init_pix2struct(model_ckpt)
		collate_fn = partial(dset.collate_fn, model_name=model_name, processor=processor)
	elif model_name in ['layoutlmv3', 'layoutlmv3-large']:
		_, processor = init_layoutlmv3(model_ckpt)
		collate_fn = partial(dset.collate_fn, model_name=model_name, processor=processor)
	elif model_name == 'udop':
		_, processor = init_udop(model_ckpt)
		collate_fn = partial(dset.collate_fn, model_name=model_name, processor=processor)
	else:
		raise ValueError(f"Invalid model: {model_name}.")
	dloader = DataLoader(
		dataset=dset,
		batch_size=batch_size,
		shuffle=shuffle,
		collate_fn=collate_fn,
		num_workers=8,
		pin_memory=True,
	)
	return dloader

class BaseDataset(Dataset):
	def __len__(self):
		raise NotImplementedError(f"__len__() method not implemented for {self.__class__.__name__}")

	def __getitem__(self, i):
		raise NotImplementedError(f"__getitem__() method not implemented for {self.__class__.__name__}")

	def collate_fn(self, batch, **kwargs):
		if kwargs['model_name'] == 'vt5':
			_batch = {_k: [_d[_k] for _d in batch] for _k in batch[0] if _k not in ["images"]}
			_batch["words"] = [[_w.lower() for _w in _ws] for _ws in _batch["words"]]
			_batch["contexts"] = [_c.lower() for _c in _batch["contexts"]]
			_batch["answers"] = [[_a.lower() for _a in _ans] for _ans in _batch["answers"]] # VT5 ANSWERS MUST be lowered, QUESTIONS not
			_batch["images"] = {
				'pixel_values': torch.stack([
					kwargs['model'].model.visual_embedding.feature_extractor(images=_d["images"], return_tensors="pt")['pixel_values'].squeeze(0)
					for _d in batch
				])
			}
		elif kwargs['model_name'] == 'donut':
			_batch = {_k: [_d[_k] for _d in batch] for _k in batch[0]}
			processor = kwargs['processor']

			texts = [
				f"<s_docvqa><s_question>{_q}</s_question><s_answer>{_a[0]}</s_answer></s>"
				for _q,_a in zip(_batch['questions'], _batch['answers'])
			]
			text_input_ids = processor.tokenizer(texts,
				add_special_tokens=False, max_length=128,
				padding="max_length", truncation=True,
				return_tensors="pt",
			).input_ids

			prompts = [f"<s_docvqa><s_question>{_q}</s_question><s_answer>" for _q in _batch['questions']]
			prompt_input_ids = [processor.tokenizer(prompt,
				add_special_tokens=False, max_length=128, truncation=True, return_tensors="pt"
			).input_ids.squeeze(0) for prompt in prompts]

			labels = text_input_ids.clone()
			labels[labels == processor.tokenizer.pad_token_id] = -100
			for _i in range(len(labels)):
				labels[_i][:torch.nonzero(labels[_i] == processor.tokenizer.convert_tokens_to_ids('<s_answer>')).sum() + 1] = -100

			_batch["images"] = processor(_batch["images"], return_tensors="pt")
			_batch["text_input_ids"] = text_input_ids
			_batch["prompt_input_ids"] = prompt_input_ids
			_batch["labels"] = labels
		elif kwargs['model_name'] in ['pix2struct', 'pix2struct-large']:
			_batch = {_k: [_d[_k] for _d in batch] for _k in batch[0]}
			processor = kwargs['processor']

			encodings = processor(images=_batch["images"], text=_batch['questions'],
				add_special_tokens=True, max_length=128,
				padding="longest", truncation=True, return_tensors="pt"
			)
			label_input_ids = processor.tokenizer([_a[0] for _a in _batch['answers']],
				add_special_tokens=True, max_length=128,
				padding="longest", truncation=True, return_tensors="pt",
			).input_ids
			labels = label_input_ids.clone()
			labels[labels == processor.tokenizer.pad_token_id] = -100

			_batch["images"] = {
				'pixel_values': encodings['flattened_patches'],
				'attention_mask': encodings['attention_mask']
			}
			_batch["labels"] = labels
		elif kwargs['model_name'] in ['layoutlmv3', 'layoutlmv3-large']:
			_batch = {_k: [_d[_k] for _d in batch] for _k in batch[0]}
			_batch["questions"] = [_q.lower() for _q in _batch["questions"]]
			_batch["answers"] = [[_an.lower() for _an in _ans] for _ans in _batch["answers"]]
			_batch["all_answers"] = [[_an.lower() for _an in _ans] for _ans in _batch["all_answers"]]
			_batch["contexts"] = [_c.lower() for _c in _batch["contexts"]]
			_batch["words"] = [[_w.lower() for _w in _ws] for _ws in _batch["words"]]
			processor = kwargs['processor']
			# https://huggingface.co/docs/transformers/en/model_doc/layoutlmv3#transformers.LayoutLMv3Tokenizer.__call__
			encodings = processor(
				_batch["images"], _batch['questions'], _batch['words'],
				boxes=[[list((_b*1000).astype(int)) for _b in _boxes] for _boxes in _batch['boxes']],
				padding=True, truncation=True, return_tensors="pt"
			)
			_batch["images"] = {'pixel_values': encodings['pixel_values']}
			_batch['input_ids'] = encodings['input_ids']
			_batch['attention_mask'] = encodings['attention_mask']
			_batch['boxes'] = encodings['bbox']
			_starts, _ends = get_start_end(encodings, _batch["contexts"], _batch["all_answers"], processor)
			_batch["start_positions"], _batch["end_positions"] = _starts[None,:].t(), _ends[None,:].t()
		elif kwargs['model_name'] == 'udop':
			# Some documents have no OCR texts and UDOP does not process those :/
			_batch = {_k: [_d[_k] for _d in [_b for _b in batch if len(_b["words"]) > 0]] for _k in batch[0]}
			processor = kwargs['processor']

			encodings = processor(
				_batch["images"],
				["Question Answering. "+_q for _q in _batch['questions']],
				_batch['words'], boxes=[[list(_b) for _b in _boxes] for _boxes in _batch['boxes']], 
				padding="max_length", truncation=True, return_tensors="pt"
			)
			label_input_ids = processor(
				text_target=[_a[0] for _a in _batch['answers']],
				padding="longest", truncation=True, max_length=128, return_tensors="pt"
			).input_ids
			labels = label_input_ids.clone()
			labels[labels == processor.tokenizer.pad_token_id] = -100

			_batch["images"] = {'pixel_values': encodings['pixel_values']}
			_batch["input_ids"] = encodings['input_ids']
			_batch["attention_mask"] = encodings['attention_mask']
			_batch["bbox"] = encodings['bbox']
			_batch["labels"] = labels
		else:
			raise ValueError(f"Invalid model: {kwargs['model_name']}.")
		return _batch

class MIDataset(BaseDataset):
	def __init__(self, all_data, data_dict, data_dir, question='v0', outputs=None):
		self.data_dir = data_dir
		dset = os.path.basename(os.path.normpath(data_dir))
		self.imext = '.png' if 'docvqa' in dset else '.jpg'
		self.all_data = all_data
		self.data_dict = data_dict
		self.docs, self.inds, self.answers, self.questions = [], [], [], []
		for _d in data_dict:
			in_key = 'sampled_indices' if 'sampled_indices' in data_dict[_d] else 'indices'
			for _ind in data_dict[_d][in_key]:
				doc_id = all_data[_ind]['image_name'] if 'docvqa' in dset else all_data[_ind]['image_name'].split('_')[0]
				assert doc_id == _d
				self.docs += [doc_id]; self.inds += [_ind]
				if outputs:
					self.answers += [outputs[_ind]['predict'] if _ind in outputs else outputs[str(_ind)]['predict']]
				else:
					self.answers += [random.sample(all_data[_ind]['answers'], 1)[0]] if isinstance(all_data[_ind]['answers'], list) else [all_data[_ind]['answers']]
				if question == 'v0':
					self.questions += [all_data[_ind]['question']]
				else:
					assert question is not None
					if str(_ind) not in data_dict[doc_id]['rephrased']: continue;
					self.questions += [data_dict[doc_id]['rephrased'][str(_ind)]]
		assert len(self.inds) == len(self.answers)

	def __len__(self):
		return len(self.inds)

	def __getitem__(self, i):
		record = self.all_data[self.inds[i]]

		image_path = os.path.join(self.data_dir, "images", record['image_name'] + self.imext)
		image = Image.open(image_path).convert("RGB")
		words = record['ocr_tokens']
		boxes = np.array([bbox for bbox in record['ocr_normalized_boxes']])
		question = self.questions[i]
		answer = self.answers[i]
		ret = {
			"images": image,
			'image_names': record['image_name'],
			'question_ids': self.inds[i],
			'questions': question,
			'contexts': " ".join([_t for _t in record['ocr_tokens']]),
			'answers': [answer],
			'words': words,
			'boxes': boxes,
			'document_ids': self.docs[i],
		}
		return ret

class FTDataset(BaseDataset):
	def __init__(self, data_dir, split):
		self.data_dir = data_dir
		self.dset = os.path.basename(os.path.normpath(data_dir))
		self.split = split
		assert self.dset in ['docvqa', 'docvqav0', 'pfl'], ValueError(f"Invalid dataset: {dset}.")
		if 'pfl' in self.dset.lower():
			self.imext = '.jpg'
			split_npy_path = os.path.join(self.data_dir, self.split,
				f"blue_{self.split}.npy" if self.split in ['train', 'val'] else f"red_{self.split}.npy")
			self.all_data = np.load(split_npy_path, allow_pickle=True)
		elif 'docvqa' in self.dset.lower():
			self.imext = '.png'
			self.all_data = np.load(os.path.join(self.data_dir, self.split, "vqa.npy"), allow_pickle=True)
		else:
			raise ValueError(f"Invalid data: {data_dir}.")
		self.image_dir = os.path.join(self.data_dir, "images")

		if 'image_name' not in self.all_data[0]:
			self.all_data = self.all_data[1:]  # Ignore header line

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, i):
		record = self.all_data[i]

		image_path = os.path.join(self.image_dir, record['image_name'] + self.imext)
		image = Image.open(image_path).convert("RGB")
		words = record['ocr_tokens']
		boxes = np.array([bbox for bbox in record['ocr_normalized_boxes']])
		question = record['question']
		answer = random.sample(record['answers'], 1)[0] if isinstance(record['answers'], list) else record['answers']
		ret = {
			"images": image,
			'image_names': record['image_name'],
			'question_ids': i,
			'questions': question,
			'contexts': " ".join([_t for _t in record['ocr_tokens']]),
			'answers': [answer],
			'all_answers': record['answers'],
			'words': words,
			'boxes': boxes,
			'document_ids': record['image_name'] if 'docvqa' in self.dset.lower() else record['image_name'].split('_')[0]
		}
		return ret
