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

parser = argparse.ArgumentParser(description='script to run attack')
parser.add_argument('--root', type=str, default='/data/users/vkhanh/mia_docvqa/data/', help='data root directory')
parser.add_argument('--dvqa_out', type=str, default='test', help='DocVQA split to sample non-members')
parser.add_argument('--pfl_out', type=str, default='test', help='PFL split to sample non-members')
parser.add_argument('--dvqa_pilot', default=False, action='store_true', help='create DVQA pilot')
parser.add_argument('--pfl_pilot', default=False, action='store_true', help='create PFL pilot')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()
assert args.dvqa_out in ['val', 'test'] and args.pfl_out in ['val', 'test'], f"valid option: ['val', 'test']"
assert args.dvqa_pilot ^ args.pfl_pilot

pilot = 300
seed = args.seed
random.seed(args.seed)

# └── DATA_ROOT
#     └── docvqav0
#         └── train/  (imdbs.zip)
#             └── vqa.npy
#         ├── val/
#         ├── test/
#         ├── images/ (spdocvqa_images.tar.gz)
#     └── pfl
#         └── train/  (imdbs.zip)
#             └── blue_train.npy
#         ├── val/
#             └── blue_val.npy
#         ├── test/
#             └── red_test.npy
#         ├── images/ (images.zip)
#     └── docvqa
#         └── train/
#             └── vqa.npy
#         ├── val/
#         ├── test/
#         ├── images/

def download_file(url, path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


print(f"###################### DOWNLOAD DATA...")



###################### DocVQA-v0 ######################
docvqav0_dir = os.path.join(args.root, DOCVQAv0)
assert os.path.isdir(docvqav0_dir)

# docvqav0_imdb_url = 'docvqa.v0/imdbs.zip'
# download_file(docvqav0_imdb_url, docvqav0_dir)
with zipfile.ZipFile(os.path.join(docvqav0_dir, 'imdbs.zip'), 'r') as zref:
    zref.extractall(docvqav0_dir)
with tarfile.open(os.path.join(docvqav0_dir, 'spdocvqa_images.tar.gz'), 'r:gz') as tref:
    extract_dir = os.path.join(docvqav0_dir, 'images')
    os.makedirs(extract_dir, exist_ok=True)
    tref.extractall(extract_dir)
print(f"****** DocVQA-v0 downloaded to: {docvqav0_dir}")



###################### DocVQA-due ######################
docvqa_dir = os.path.join(args.root, DOCVQAdue)
os.makedirs(docvqa_dir, exist_ok=True)

# docvqa_imdb_url = 'docvqa.due/imdbs.zip'
# download_file(docvqa_imdb_url, docvqa_dir)
with zipfile.ZipFile(os.path.join(docvqa_dir, 'imdbs.zip'), 'r') as zref:
    zref.extractall(docvqa_dir)
shutil.copytree(
    os.path.join(docvqa_dir, '../', DOCVQAv0, 'images'),
    os.path.join(docvqa_dir, 'images')
)
print(f"****** DocVQA-due downloaded to: {docvqa_dir}")



print(f"****** Check both DocVQA versions and convert...")
due_TRAIN = np.load(os.path.join(docvqa_dir, 'train/vqa.npy'), allow_pickle=True)
due_NON_TRAIN = np.load(os.path.join(docvqa_dir, f'{args.dvqa_out}/vqa.npy'), allow_pickle=True)

v0_TRAIN = np.load(os.path.join(docvqav0_dir, 'train/vqa.npy'), allow_pickle=True)[1:]
v0_NON_TRAIN = np.load(os.path.join(docvqav0_dir, f'{args.dvqa_out}/vqa.npy'), allow_pickle=True)[1:]

v0_train_questions = {_r['question_id']:_r for _r in v0_TRAIN}
v0_nontrain_questions = {_r['question_id']:_r for _r in v0_NON_TRAIN}

train_dict, nontrain_dict = dict(), dict()
for _ind, _train in enumerate(due_TRAIN):
    doc_id = _train['image_name']
    if doc_id in train_dict:
        train_dict[doc_id]['count'] += 1
        train_dict[doc_id]['indices'].append(_ind)
    else:
        train_dict[doc_id] = {
            'count': 1,
            'label': 1,
            'indices':[_ind],
        }
for _ind, _nontrain in enumerate(due_NON_TRAIN):
    doc_id = _nontrain['image_name']
    if doc_id in nontrain_dict:
        nontrain_dict[doc_id]['count'] += 1
        nontrain_dict[doc_id]['indices'].append(_ind)
    else:
        nontrain_dict[doc_id] = {
            'count': 1,
            'label': 0,
            'indices':[_ind],
        }
print(f"DocVQA-due (documents/questions): "
        +f"train=({len(train_dict)}/{len(due_TRAIN)}), "
        +f"non-train={len(nontrain_dict)}/{len(due_NON_TRAIN)}")
with open(os.path.join(docvqa_dir, f'{DOCVQAdue}_train.json'), 'w') as f:
    json.dump(train_dict, f)
with open(os.path.join(docvqa_dir, f'{DOCVQAdue}_nontrain.json'), 'w') as f:
    json.dump(nontrain_dict, f)

due_document_dict, due_mia_npy = dict(), []
v0_document_dict, v0_mia_npy = dict(), []
for _d in tqdm(list(train_dict.keys()), desc='member docs'):
    due_d_data, v0_d_data = [], []
    for _ind in train_dict[_d]['indices']:
        due_d_data.append(due_TRAIN[_ind])
        v0_d_data.append(v0_train_questions[due_TRAIN[_ind]['metadata']['question_id']])

        doc_id = due_TRAIN[_ind]['image_name']
        assert os.path.isfile(os.path.join(docvqa_dir, 'images', f'{doc_id}.png'))
        assert os.path.isfile(os.path.join(docvqav0_dir, 'images', f'{doc_id}.png'))

    due_document_dict[_d] = {
    'label': 1, 'indices': list(range(len(due_mia_npy), len(due_mia_npy)+len(due_d_data))), 'count': len(due_d_data)}
    v0_document_dict[_d] = {
    'label': 1, 'indices': list(range(len(v0_mia_npy), len(v0_mia_npy)+len(v0_d_data))), 'count': len(v0_d_data)}

    due_mia_npy.extend(due_d_data); v0_mia_npy.extend(v0_d_data)
assert len(due_mia_npy) == len(v0_mia_npy) and len(due_document_dict) == len(v0_document_dict) == len(train_dict)
cutoff_d = len(due_document_dict); cutoff_q = len(due_mia_npy)

for _d in tqdm(list(nontrain_dict.keys()), desc='non-member docs'):
    due_d_data, v0_d_data = [], []
    for _ind in nontrain_dict[_d]['indices']:
        due_d_data.append(due_NON_TRAIN[_ind])
        v0_d_data.append(v0_nontrain_questions[due_NON_TRAIN[_ind]['metadata']['question_id']])

        doc_id = due_NON_TRAIN[_ind]['image_name']
        assert os.path.isfile(os.path.join(docvqa_dir, 'images', f'{doc_id}.png'))
        assert os.path.isfile(os.path.join(docvqav0_dir, 'images', f'{doc_id}.png'))

    due_document_dict[_d] = {
    'label': 0, 'indices': list(range(len(due_mia_npy), len(due_mia_npy)+len(due_d_data))), 'count': len(due_d_data)}
    v0_document_dict[_d] = {
    'label': 0, 'indices': list(range(len(v0_mia_npy), len(v0_mia_npy)+len(v0_d_data))), 'count': len(v0_d_data)}

    due_mia_npy.extend(due_d_data); v0_mia_npy.extend(v0_d_data)
assert len(due_mia_npy) == len(v0_mia_npy) and len(due_document_dict)-cutoff_d == len(v0_document_dict)-cutoff_d == len(nontrain_dict)
assert all([
    (due_document_dict[_d]['label'] == 1
    and v0_document_dict[_d]['label'] == 1
    and due_document_dict[_d]['count'] == v0_document_dict[_d]['count'] == train_dict[_d]['count'])
    for _d in train_dict
])
assert all([
    (due_document_dict[_d]['label'] == 0
    and v0_document_dict[_d]['label'] == 0
    and due_document_dict[_d]['count'] == v0_document_dict[_d]['count'] == nontrain_dict[_d]['count'])
    for _d in nontrain_dict
])

print("STATS (documents/questions):\t")
print(f"DocVQA-due: "
        +f"train={len(due_document_dict)}/{len(due_mia_npy)}, "
        +f"member={len([_v for _v in due_document_dict.values() if _v['label'] == 1])}/{len(due_mia_npy[:cutoff_q])}, "
        +f"non-member={len([_v for _v in due_document_dict.values() if _v['label'] == 0])}/{len(due_mia_npy[cutoff_q:])}")
print(f"DocVQA-v0: "
        +f"train={len(v0_document_dict)}/{len(v0_mia_npy)}, "
        +f"member={len([_v for _v in v0_document_dict.values() if _v['label'] == 1])}/{len(v0_mia_npy[:cutoff_q])}, "
        +f"non-member={len([_v for _v in v0_document_dict.values() if _v['label'] == 0])}/{len(v0_mia_npy[cutoff_q:])}")
assert all([
    (_rdue['image_name'] == _rdue['image_name'])
    and (Image.open(os.path.join(docvqa_dir, 'images', f"{_rdue['image_name']}.png")).size == Image.open(os.path.join(docvqa_dir, 'images', f"{_rv0['image_name']}.png")).size)
    and (_rdue['question'] == _rv0['question'])
    for _rdue,_rv0 in zip(due_mia_npy,v0_mia_npy)
])

with open(os.path.join(docvqa_dir, f'{DOCVQAdue}_mia.json'), 'w') as f:
    json.dump(due_document_dict, f)
np.save(os.path.join(docvqa_dir, f'{DOCVQAdue}_mia.npy'), due_mia_npy, allow_pickle=True)
with open(os.path.join(docvqav0_dir, f'{DOCVQAv0}_mia.json'), 'w') as f:
    json.dump(v0_document_dict, f)
np.save(os.path.join(docvqav0_dir, f'{DOCVQAv0}_mia.npy'), v0_mia_npy, allow_pickle=True)

if args.dvqa_pilot:
    print(f"****** Create DocVQA one pilot (seed={seed}) and check...")
    os.makedirs(os.path.join(docvqa_dir, f'pilot/seed{seed}'), exist_ok=True)
    os.makedirs(os.path.join(docvqav0_dir, f'pilot/seed{seed}'), exist_ok=True)

    DVQA_INs = random.sample([_d for _d in train_dict.keys()], pilot)
    DVQA_OUTs = random.sample([_d for _d in nontrain_dict.keys()], pilot)
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

    with open(os.path.join(docvqa_dir, f'pilot/seed{seed}', f'{DOCVQAdue}_mia.json'), 'w') as f:
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
    print("")



##################### PFL-DocVQA ######################
pfl_dir = os.path.join(args.root, PFLDOCVQA)
assert os.path.isdir(pfl_dir)

pfl_imdb_url = 'pfl.docvqa/imdbs.zip'
download_file(pfl_imdb_url, pfl_dir)
with zipfile.ZipFile(os.path.join(pfl_dir, 'imdbs.zip'), 'r') as zref:
    zref.extractall(pfl_dir)
with zipfile.ZipFile(os.path.join(pfl_dir, 'images.zip'), 'r') as zref:
    zref.extractall(pfl_dir)
print(f"****** PFL-DocVQA downloaded to: {pfl_dir}")
print(f"****** Check PFL-DocVQA data and convert...")

BLUE_provider2docs = json.load(open(os.path.join(pfl_dir, 'train/provider2docs.json'), 'r'))
BLUE_datapoints = json.load(open(os.path.join(pfl_dir, 'train/data_points.json'), 'r'))
RED_doc2provider = json.load(open(os.path.join(pfl_dir, 'test/red_doc2provider.json'), 'r'))
pfl_TRAIN = np.load(os.path.join(pfl_dir, 'train/blue_train.npy'), allow_pickle=True)
if args.pfl_out == 'val':
    pfl_NON_TRAIN = np.load(os.path.join(pfl_dir, 'val/blue_val.npy'), allow_pickle=True)[1:]
elif args.pfl_out == 'test':
    pfl_NON_TRAIN = np.load(os.path.join(pfl_dir, 'test/red_test.npy'), allow_pickle=True)[1:]

# 1. Base the provider list on the RED test
RED_provider_dict = dict()
for _record in tqdm(pfl_NON_TRAIN, desc='RED providers'):
    docid = _record['image_name'].split('_')[0]
    prov = RED_doc2provider[docid]['provider']
    if prov not in RED_provider_dict:
        RED_provider_dict[prov] = {'label': (1 if _record["set_name"] == "red_test_positive" else 0)}

# 2. Add OUT documents from the RED test
document_dict = dict()
for _record in tqdm(pfl_NON_TRAIN, desc='non-member docs'):
    docid = _record['image_name'].split('_')[0]
    if docid not in document_dict and _record["set_name"] == "red_test_negative":
        document_dict[docid] = {'label': 0, 'indices': []}
cutoff_d = len(document_dict)

# 3. Add IN documents from the BLUE train
all_dinds = []
for _prov in RED_provider_dict:
    if RED_provider_dict[_prov]['label'] == 1:
        # if len(BLUE_provider2docs[_prov]) <= 10:
        #     _prov_docs = BLUE_provider2docs[_prov]
        # else:
        #     _prov_docs =random.sample(_prov[_prov], random.randint(1, 10))
        prov_docs = BLUE_provider2docs[_prov]
        for _docid in prov_docs:
            if _docid not in document_dict:
                document_dict[_docid] = {'label': 1, 'indices': []}
        all_dinds.extend(BLUE_datapoints[_prov])
print(f"PFL Documents (members/non-members): {len(document_dict)}({len(document_dict)-cutoff_d}/{cutoff_d})")

# 4. Discard IN documents if they are from RED test (because these are not used to train)
document_dict = {
    _d:_v for _d,_v in document_dict.items()
    if not (
        _d in RED_doc2provider
        and RED_doc2provider[_d]['provider'] in RED_provider_dict
        and RED_provider_dict[RED_doc2provider[_d]['provider']]['label'] == 1
    )
}
assert '1a41baf416dbf2e74b0fd6d5' not in document_dict  # random check
print(f"PFL Documents, after filtered (members/non-members): "
    +f"{len(document_dict)}({len(document_dict)-cutoff_d}/{cutoff_d})")

# 5. Now, extract data for each document, first IN documents
pfl_mia_npy = [
    _record for _record in tqdm(pfl_TRAIN[all_dinds], desc='member docs')
    if _record['image_name'].split('_')[0] in document_dict
]
cutoff_q = len(pfl_mia_npy)

# 6. Then OUT documents
for _record in tqdm(pfl_NON_TRAIN, desc='non-member docs'):
    docid = _record['image_name'].split('_')[0]
    if docid in document_dict:
        if _record['set_name'] == 'red_test_postive': # double-check step 4.
            document_dict.pop(docid)
        else:
            pfl_mia_npy.append(_record)
print(f"PFL Questions (members/non-members): {len(pfl_mia_npy)}({len(pfl_mia_npy)-cutoff_q}/{cutoff_q})")

# 7. Update document dict with data indices
for _ind,_record in tqdm(enumerate(pfl_mia_npy)):
    docid = _record['image_name'].split('_')[0]
    document_dict[docid]['indices'].append(_ind)
document_dict = {_k:_v for _k,_v in document_dict.items() if len(_v['indices']) > 0}

with open(os.path.join(pfl_dir, f'{PFLDOCVQA}_mia.json'), 'w') as f:
    json.dump(document_dict, f)
np.save(os.path.join(pfl_dir, f'{PFLDOCVQA}_mia.npy'), pfl_mia_npy, allow_pickle=True)

if args.pfl_pilot:
    print(f"****** Create PFL-DocVQA one pilot (seed={seed}) and check...")
    os.makedirs(os.path.join(pfl_dir, f'pilot/seed{seed}'), exist_ok=True)

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

    with open(os.path.join(pfl_dir, f'pilot/seed{seed}', f'{PFLDOCVQA}_mia.json'), 'w') as f:
        json.dump(pilot_document_dict, f)



print(f"###################### FINISHED!")
