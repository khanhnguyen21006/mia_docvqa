import os, json, pickle, copy, random, re
import time, datetime
import itertools, shutil
from functools import partial
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd

import editdistance
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn import metrics
from sklearn import preprocessing
import sklearn.cluster as cluster
import sklearn.mixture as mixture

BASES = ['accuracy', 'anls', 'loss']
FEATURES = ['update', 'anls', 'step']
AGG_FNS = ['avg', 'min', 'max', 'med']
PASSED = ['label', 'output', 'indices', 'query']

def anls(ans, pred, thresh=0.5):
	ans = [_an.lower().strip() for _an in ans]
	pred = pred.lower().strip()

	if len(pred) == 0: return 0;

	maxsim = max([1 - editdistance.eval(_an, pred) / max(len(_an), len(pred)) for _an in ans])
	anls = maxsim if maxsim >= thresh else 0
	return anls

def accuracy(ans, pred):
	ans = [_an.lower().strip() for _an in ans]
	pred = pred.lower().strip()

	if len(pred) == 0: return 0;

	for _an in ans:
		if _an == pred:
			return 1
	return 0

def get_pred_fn(name):
	PRE = {
		"threshold": run_threshold,
		"kmeans": run_kmeans,
		"gmm": run_gaussian_mixture,
	}
	assert name in PRE, f"Invalid predictor: {name}."
	return PRE[name]

def get_agg_fn(name):
	AGG = {
		"avg": lambda x: np.mean(x) if len(x) > 0 else 0,
		"sum": lambda x: np.sum(x) if len(x) > 0 else 0,
		"med": lambda x: np.median(x) if len(x) > 0 else 0,
		"min": lambda x: np.min(x) if len(x) > 0 else 0,
		"max": lambda x: np.max(x) if len(x) > 0 else 0,
	}
	assert name in AGG, f"Invalid aggregator: {name}."
	return AGG[name]

def set_prior(fs, p='first'):
	asc, desc = ['accuracy', 'anls', 'arrive_gt', 'zero_shot'], ['update', 'gradnorm', 'loss', 'step']
	if p == 'major':
		return len([_ for _f in fs if _f.split('.')[-1] in asc]) > len([_ for _f in fs if _f.split('.')[-1] in desc])
	elif p == 'first':
		return fs[0].split('.')[-1] in asc
	else:
		assert p in asc+desc, ValueError(f"Invalid prior {p}.")
		return p in asc

def normalize(X, scaler='minmax'):
	X = np.array(X)
	if scaler == 'standard':
		scaler = preprocessing.StandardScaler().fit(X)
		X_scaled = scaler.transform(X)
		return X_scaled
	elif scaler == 'minmax':
		scaler = preprocessing.MinMaxScaler()
		X_scaled = scaler.fit_transform(X)
		return X_scaled
	elif scaler == 'maxabs':
		scaler = preprocessing.MaxAbsScaler()
		X_scaled = scaler.fit_transform(X)
		return X_scaled
	else:
		raise ValueError(f"Invalid scaler {scaler}.")

def assign_cluster(X, preds, base=0, ascending=False, return_mu=False):
	if base == 'mean':
		# This computes the mean of all metrics and assign based only on its value
		c0 = np.mean([X[_i] for _i,_yp in enumerate(preds) if _yp == 0])
		c1 = np.mean([X[_i] for _i,_yp in enumerate(preds) if _yp == 1])
		if ascending:
			y_pred = preds if c1 >= c0 else (1-preds)
		else:
			y_pred = preds if c1 <= c0 else (1-preds)
	elif base == 'norm2':
		# This computes the controids and assign based only on its norm2
		c0 = np.mean([X[_i] for _i,_yp in enumerate(preds) if _yp == 0], axis=0)
		c1 = np.mean([X[_i] for _i,_yp in enumerate(preds) if _yp == 1], axis=0)
		if ascending:
			y_pred = preds if np.linalg.norm(c1) >= np.linalg.norm(c0) else (1-preds)
		else:
			y_pred = preds if np.linalg.norm(c1) <= np.linalg.norm(c0) else (1-preds)
	elif isinstance(base, int):
		assert base >= 0 and base < len(X[0])
		# This selects one metric and assign based only on its value
		c0 = np.mean([X[_i][base] for _i,_yp in enumerate(preds) if _yp == 0])
		c1 = np.mean([X[_i][base] for _i,_yp in enumerate(preds) if _yp == 1])
		if ascending:
			y_pred = preds if c1 >= c0 else (1-preds)
		else:
			y_pred = preds if c1 <= c0 else (1-preds)
	else:
		raise ValueError(f"Invalid assignment: {base}")
	if return_mu:
		return y_pred, c0, c1
	return y_pred

def run_threshold(X, thres=0.5, **kwargs):
	# normalized between [0,1]
	y_pred = [1 if x<thres else 0 for x in preprocessing.normalize([X])[0]]
	return y_pred

def run_kmeans(X, seed=None, ascending=False, base=0):
	kmeans = cluster.KMeans(init="random", n_clusters=2, n_init=10, max_iter=300, random_state=seed)
	kmeans.fit(X)
	y_pred, c0, c1 = assign_cluster(X, kmeans.labels_, base, ascending, return_mu=True)
	cmem = np.mean([X[_i] for _i in range(len(X)) if y_pred[_i] == 1], axis=0)
	dist = np.linalg.norm(X - cmem[None,:], axis=1)
	ranks = stats.rankdata(-dist, method='ordinal')
	return y_pred, ranks

def run_gaussian_mixture(X, seed=None, ascending=False, base=0):
	gm = mixture.GaussianMixture(n_components=2, n_init=10, max_iter=300, random_state=seed)
	gm.fit(X)
	gm_pred = gm.predict(X)
	y_pred, c0, c1 = assign_cluster(X, gm_pred, base, ascending, return_mu=True)
	if ascending:
		probs = gm.predict_proba(X)[:,1] if c1 >= c0 else gm.predict_proba(X)[:,0]
	else:
		probs = gm.predict_proba(X)[:,1] if c1 <= c0 else gm.predict_proba(X)[:,0]
	ranks = stats.rankdata(probs, method='ordinal')
	return y_pred, ranks

def create_df(all_result, aggs=['avg']):
	df_columns = list(list(all_result.values())[0].keys())
	df_data = {}
	for _col in df_columns:
		if _col not in ['indices', 'label', 'output']:
			for _agg in aggs:
				df_data[f"{_agg}.avg.{_col}"] = []
				if _col not in ['update', 'step']:
					df_data[f"{_agg}.last.{_col}"] = []
					df_data[f"{_agg}.min.{_col}"] = []
					df_data[f"{_agg}.max.{_col}"] = []
		else:
			df_data[_col] = []
			if _col == 'indices':
				df_data['query'] = []

	for _k,_v in all_result.items():
		for _col in df_columns:
			if _col in ['loss', 'accuracy', 'anls']:
				if isinstance(_v[_col], list) and all([isinstance(_vl, list) for _vl in _v[_col]]): # 2d array: len(dinds)*step
					# per-step aggregate
					_list_mean_val = [np.mean(_vl) if len(_vl) > 0 else 0.0 for _vl in _v[_col]]
					_list_last_val = [_vl[-1] if len(_vl) > 0 else 0.0 for _vl in _v[_col]]
					_list_min_val = [np.min(_vl) if len(_vl) > 0 else 0.0 for _vl in _v[_col]]
					_list_max_val = [np.max(_vl) if len(_vl) > 0 else 0.0 for _vl in _v[_col]]
					# per-query aggregate
					for _agg in aggs:
						df_data[f"{_agg}.avg.{_col}"].append(get_agg_fn(_agg)(_list_mean_val) if len(_list_mean_val) > 0 else 0)
						df_data[f"{_agg}.last.{_col}"].append(get_agg_fn(_agg)(_list_last_val) if len(_list_mean_val) > 0 else 0)
						df_data[f"{_agg}.min.{_col}"].append(get_agg_fn(_agg)(_list_min_val) if len(_list_mean_val) > 0 else 0)
						df_data[f"{_agg}.max.{_col}"].append(get_agg_fn(_agg)(_list_max_val) if len(_list_mean_val) > 0 else 0)
				elif isinstance(_v[_col], list) and all([np.isscalar(_vl) for _vl in _v[_col]]): # list: len(dinds)
					# per-query aggregate
					for _agg in aggs:
						df_data[f"{_agg}.avg.{_col}"].append(get_agg_fn(_agg)(_v[_col]))
						df_data[f"{_agg}.last.{_col}"].append(get_agg_fn(_agg)(_v[_col]))
						df_data[f"{_agg}.min.{_col}"].append(get_agg_fn(_agg)(_v[_col]))
						df_data[f"{_agg}.max.{_col}"].append(get_agg_fn(_agg)(_v[_col]))
				else: # scalar
					# per-query aggregate
					for _agg in aggs:
						df_data[f"{_agg}.avg.{_col}"].append(_v[_col])
						df_data[f"{_agg}.last.{_col}"].append(_v[_col])
						df_data[f"{_agg}.min.{_col}"].append(_v[_col])
						df_data[f"{_agg}.max.{_col}"].append(_v[_col])
			elif _col in ['update', 'step']: # len(dinds)
				for _agg in aggs:
					df_data[f"{_agg}.avg.{_col}"].append(get_agg_fn(_agg)(_v[_col]) if isinstance(_v[_col], list) else _v[_col])
			elif _col in ['indices']: # len(dinds)
				df_data[_col].append(_v[_col])
				df_data['query'].append(len(_v[_col]))
			elif _col in ['label']: # scalar
				df_data[_col].append("member" if _v[_col] == 1 else "non-member")
			else:
				df_data[_col].append(_v[_col])
	return pd.DataFrame(data=df_data)

def create_pilot(data_dir, data, seed, question, nmax=300):
	_path = os.path.join(data_dir, f'pilot/seed{seed}', f'pilot{nmax}_mia_{question}.json')
	if os.path.exists(_path): pilot = json.load(open(_path, 'r'));
	else:
		pilot, np, nn = dict(), 0, 0
		dkeys = list(data.keys()); random.shuffle(dkeys)
		for _k in dkeys:
			if data[_k]['label'] == 1 and np < nmax:
				np += 1; pilot[_k] = data[_k]
			if data[_k]['label'] == 0 and nn < nmax:
				nn += 1; pilot[_k] = data[_k]
			if np == nmax and nn == nmax:
				break
		with open(_path, 'w') as f:
			json.dump(pilot, f)
	return pilot

def compute_score(result, feature_set, predictor, bases, seed=None):
	pred_fn = get_pred_fn(predictor)

	X = [[_row[_m] for _m in feature_set] for _,_row in result.iterrows()]
	y = [1 if _label == 'member' else 0 for _label in result['label']]
	X = normalize(X)

	acc_lst, f1_lst, rank_lst = [], [], []
	tpr001fpr_lst, tpr003fpr_lst, tpr005fpr_lst = [], [], []
	for _base in bases:
		max_acc = -np.inf
		for _m in feature_set:
			_mn = _m.split('.')[-1]
			if _mn == _base:
				y_pred, rank = pred_fn(X, seed=seed,
					ascending=set_prior(feature_set, p=_mn), base=feature_set.index(_m))
				acc = metrics.accuracy_score(y, y_pred)
				if acc > max_acc:
					max_acc = acc
					max_f1 = metrics.f1_score(y, y_pred)
					# max_rank = rank
					fpr, tpr, _ = metrics.roc_curve(y, rank)
					max_tpr001fpr = tpr[np.where(fpr<.01)[0][-1]]
					max_tpr003fpr = tpr[np.where(fpr<.03)[0][-1]]
					max_tpr005fpr = tpr[np.where(fpr<.05)[0][-1]]
		acc_lst += [max_acc]; f1_lst += [max_f1]
		tpr001fpr_lst += [max_tpr001fpr]; tpr003fpr_lst += [max_tpr003fpr]
		tpr005fpr_lst += [max_tpr005fpr]#; rank_lst += [max_rank]

	max_acc_lst, max_acc_ind = np.max(acc_lst), np.argmax(acc_lst)
	result_dict = {
		"ACC": max_acc_lst*100,
		'F1': f1_lst[max_acc_ind]*100,
		# 'rank': rank_lst[max_acc_ind],
		'TPR@0.01FPR': tpr001fpr_lst[max_acc_ind]*100,
		'TPR@0.03FPR': tpr003fpr_lst[max_acc_ind]*100,
		'TPR@0.05FPR': tpr005fpr_lst[max_acc_ind]*100,
	}
	return result_dict

def evaluate(expt_dir, expt_cfg, predictor, features, bases, aggs, seed):
	"""
		Evaluation of unsupervised attacks.
		args:
			expt_dir (str): path to saved result
			predictor: clustering method
		return:
			result_dict: dictionary containing the MIA scores.
	"""
	result = pickle.load(open(os.path.join(expt_dir, f'{expt_cfg}.pkl'), 'rb'))
	result_df = create_df(result['result'], aggs=aggs)
	feature_set = [
		_col for _col in list(result_df.columns) 
		if _col not in PASSED and _col.split('.')[-1] in features]
	print(f"METRIC={feature_set}")

	result_dict = compute_score(result_df, feature_set, predictor, bases, seed=seed)
	return result_dict

def save_result(save_dir, name, result):
	os.makedirs(save_dir, exist_ok=True)

	with open(os.path.join(save_dir, f"{name}.pkl"), 'wb') as f:
		pickle.dump({"result": result}, f)
	return result
