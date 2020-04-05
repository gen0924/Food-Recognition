import os
import torch
import numpy as np

from dataset import FWDataset


def get_mean_std(dataset):
	"""
	get mean and std by sample 
	"""
	means = [0, 0, 0]
	std = [0, 0, 0]
	numImg = len(dataset)
	for data in dataset:
		img = data[0]
		for i in range(3):
			means[i] += img[i, :, :].mean()
			std[i] += img[i, :, :].std()

	means = np.asarray(means) / numImg
	std = np.asarray(std) / numImg

	return means, std


if __name__ == '__main__':

	train_dataset = FWDataset(phase='TRAINING')
	val_dataset = FWDataset(phase='VALING')

	means1, std1 = get_mean_std(train_dataset)
	means2, std2 = get_mean_std(val_dataset)

	print("train:", means1, std1)
	print("val:", means2, std2)

