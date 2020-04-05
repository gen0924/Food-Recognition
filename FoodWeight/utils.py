import torch
import numpy as np


def percentage_error(y_true, y_pred):
	
	y_true = y_true.detach().cpu().numpy()
	y_pred = y_pred.detach().cpu().numpy()

	diff = np.fabs(y_true - y_pred) / (y_true + 20)

	return np.mean(diff)


def p20(y_true, y_pred):
	
	y_true = y_true.detach().cpu().numpy()
	y_pred = y_pred.detach().cpu().numpy()

	sample1 = np.fabs(y_true - y_pred)
	sample2 = sample1 / (y_true + 20)

	ok = (sample1 < 20) | (sample2 < 0.2)
	ok.astype(np.float32)

	return np.mean(100.0 * ok)
