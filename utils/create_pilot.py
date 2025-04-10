import os, argparse
import json, random, shutil
import tarfile, zipfile
import requests
from tqdm import tqdm
from PIL import Image
import numpy as np

# Seperation needed to get the correct performance of each target model
DOCVQAv0 = "docvqav0"  # original annotations DocVQA benchmark
DOCVQAdue = "docvqa"  # annotations from DUE benchmark
PFLDOCVQA = "pfl"

parser = argparse.ArgumentParser(description='script to create pilot')
parser.add_argument('--data_dir',  type=str, default='/data/users/vkhanh/mia_docvqa/data/', help='data directory')
parser.add_argument('--dataset', type=str, required=True, help='dataset')
parser.add_argument('--pilot', type=int, required=True, help='size')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()
assert args.dataset in ['pfl', 'docvqav0', 'docvqa'], f"valid option: ['pfl', 'docvqav0', 'docvqa']"

pilot = args.pilot
seed = args.seed
dataset = args.dataset
random.seed(args.seed)

if 'docvqa' in args.dataset:
	docvqa_dir = os.path.join(args.data_dir, 'docvqa')
	docvqav0_dir = os.path.join(args.data_dir, 'docvqav0')
	due_document_dict = json.load(open(os.path.join(docvqa_dir, f'{DOCVQAdue}_mia.json'), 'r'))
	due_mia_npy = np.load(os.path.join(docvqa_dir, f'{DOCVQAdue}_mia.npy'), allow_pickle=True)
	v0_document_dict = json.load(open(os.path.join(docvqav0_dir, f'{DOCVQAv0}_mia.json'), 'r'))
	v0_mia_npy = np.load(os.path.join(docvqav0_dir, f'{DOCVQAv0}_mia.npy'), allow_pickle=True)
	due_train_dict = json.load(open(os.path.join(docvqa_dir, f'{DOCVQAdue}_train.json'), 'r'))
	due_nontrain_dict = json.load(open(os.path.join(docvqa_dir, f'{DOCVQAdue}_nontrain.json'), 'r'))

	print(f"****** Create DocVQA pilot (seed={seed}) and check...")
	os.makedirs(os.path.join(docvqa_dir, f'pilot/seed{seed}'), exist_ok=True)
	os.makedirs(os.path.join(docvqav0_dir, f'pilot/seed{seed}'), exist_ok=True)
	pilot_path = os.path.join(docvqa_dir, f'pilot/seed{seed}', f'{DOCVQAdue}_mia.json')
	if os.path.isfile(pilot_path):
		print(f"PATH {pilot_path} EXISTED!")
	else:
		DVQA_INs = random.sample([_d for _d in due_train_dict.keys()], pilot)
		DVQA_OUTs = random.sample([_d for _d in due_nontrain_dict.keys()], pilot)
		print(f"DocVQA pilot (members/non-members): {len(DVQA_INs)}/{len(DVQA_OUTs)}")
		print(f"5 members: {DVQA_INs[:5]}")
		print(f"5 non-members: {DVQA_OUTs[:5]}")

		due_pilot_npy = []
		due_pilot_document_dict = {_d:due_document_dict[_d] for _d in (DVQA_INs+DVQA_OUTs)}
		for _d in DVQA_INs+DVQA_OUTs:
			_d_data = []
			for _ind in due_pilot_document_dict[_d]['indices']:
				_d_data += [due_mia_npy[_ind]]
			due_pilot_document_dict[_d]['indices'] = list(range(len(due_pilot_npy),len(due_pilot_npy)+len(_d_data)))
			due_pilot_document_dict[_d]['count'] = len(_d_data)
			due_pilot_npy += _d_data

		with open(pilot_path, 'w') as f:
			json.dump(due_pilot_document_dict, f)
		np.save(os.path.join(docvqa_dir, f'pilot/seed{seed}', f'{DOCVQAdue}_mia.npy'), due_pilot_npy, allow_pickle=True)

		v0_pilot_npy = []
		v0_pilot_document_dict = {_d:v0_document_dict[_d] for _d in (DVQA_INs+DVQA_OUTs)}
		for _d in DVQA_INs+DVQA_OUTs:
			_d_data = []
			for _ind in v0_pilot_document_dict[_d]['indices']:
				_d_data += [v0_mia_npy[_ind]]
			v0_pilot_document_dict[_d]['indices'] = list(range(len(v0_pilot_npy),len(v0_pilot_npy)+len(_d_data)))
			v0_pilot_document_dict[_d]['count'] = len(_d_data)
			v0_pilot_npy += _d_data

		with open(os.path.join(docvqav0_dir, f'pilot/seed{seed}', f'{DOCVQAv0}_mia.json'), 'w') as f:
			json.dump(v0_pilot_document_dict, f)
		np.save(os.path.join(docvqav0_dir, f'pilot/seed{seed}', f'{DOCVQAv0}_mia.npy'), v0_pilot_npy, allow_pickle=True)
		print(f"Pilot SAVED to: {pilot_path}!")

elif args.dataset == 'pfl':
	pfl_dir = os.path.join(args.data_dir, 'pfl')
	document_dict = json.load(open(os.path.join(pfl_dir, 'pfl_mia.json'), 'r'))
	pfl_mia_npy = np.load(os.path.join(pfl_dir, 'pfl_mia.npy'), allow_pickle=True)

	print(f"****** Create PFL-DocVQA one pilot (seed={seed}) and check...")
	os.makedirs(os.path.join(pfl_dir, f'pilot/seed{seed}'), exist_ok=True)
	pilot_path = os.path.join(pfl_dir, f'pilot/seed{seed}', f'{PFLDOCVQA}_mia.json')
	if os.path.isfile(pilot_path):
		print(f"PATH {pilot_path} EXISTED!")
	else:
		PFL_INs = random.sample([_d for _d in document_dict.keys() if document_dict[_d]['label'] == 1], pilot)
	PFL_OUTs = random.sample([_d for _d in document_dict.keys() if document_dict[_d]['label'] == 0], pilot)
	print(f"PFL pilot (members/non-members): {len(PFL_INs)}/{len(PFL_OUTs)}")
	print(f"5 members: {PFL_INs[:5]}")
	print(f"5 non-members: {PFL_OUTs[:5]}")

	pilot_document_dict = {_d:document_dict[_d] for _d in (PFL_INs+PFL_OUTs)}
	for _d in PFL_INs+PFL_OUTs:
		_d_data = []
		for _ind in pilot_document_dict[_d]['indices']:
			_d_data += [pfl_mia_npy[_ind]]
		pilot_document_dict[_d]['count'] = len(_d_data)

	with open(pilot_path, 'w') as f:
		json.dump(pilot_document_dict, f)
	print(f"Pilot SAVED to: {pilot_path}!")
