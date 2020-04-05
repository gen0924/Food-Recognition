import os
import torch
import warnings
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
from models import *
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_checkpoint(ckpt_path, pretrained=True):

	num_classes = 583
	num_attentions = 32
	feature_net = inception_v3(pretrained=True)
	net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

	checkpoint = torch.load(ckpt_path)
	state_dict = checkpoint['state_dict']
	net.load_state_dict(state_dict)

	return net


def food_recognition(image):

	image_size = (512, 512)
	crop_size = (256, 256)
	theta_c = 0.5
	transformer = transforms.Compose([
		transforms.Resize(image_size),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.6301, 0.5246, 0.3962], std=[0.2747, 0.2881, 0.3135])
		])

	image_tensor = transformer(image)
	image_tensor.unsqueeze_(0)

	model = load_checkpoint(ckpt_path='012.ckpt', pretrained=True)
	model.eval()

	input_var = Variable(image_tensor)
	# Raw Image
	y_pred_raw, feature_matrix, attention_map = model(input_var)
	# Object Location and Refinement
	crop_mask = F.upsample_bilinear(attention_map, size=image_size) > theta_c
	crop_images = []
	for batch_index in range(crop_mask.size(0)):
		nonozero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
		height_min = nonozero_indices[:, 0].min()
		height_max = nonozero_indices[:, 0].max()
		width_min = nonozero_indices[:, 1].min()
		width_max = nonozero_indices[:, 1].max()
		crop_images.append(F.upsample_bilinear(input_var[batch_index:batch_index+1, :, height_min:height_max, width_min:width_max], size=crop_size))
	crop_images = torch.cat(crop_images, dim=0)

	y_pred_crop, _, _ = model(crop_images)

	y_pred = (y_pred_raw + y_pred_crop) / 2

	return y_pred


def accuracy(output, target, label_list, topk=(1,)):
	"""
	Compute tge precision@k for the specified values of k
	output: [list]
	target: label,str
	label_lsit: [label1, label2,label2,...]
	"""

	maxk = max(topk)
	output_sorted = sorted(output, reverse=True)
	result_list = output_sorted[:maxk]

	pred_label = []
	for result in result_list:
		index = output.index(result)
		label = label_list[index]  #label在此处代表的就是子文件夹的名字
		pred_label.append(label)
	
	# 判断label中是否存在与target这一类别（文件夹的名字）
	accuracy_list = []
	# print(target, type(target))
	# print(pred_label, len(pred_label))
	for k in topk:
		pred_list = pred_label[:k]
		if target in pred_list:
			accuracy_list.append(1)
		else:
			accuracy_list.append(0)

	return accuracy_list, pred_label[0]


def list_add(list1, list2):
	add_list = []
	if len(list1) == len(list2):
		for i in range(len(list1)):
			add_list.append(list1[i] + list2[i])
		return add_list
	else:
		return 0


if __name__ == '__main__':

	test_path = '/home/liu/icarbonx/food-recognition/food_583_splited/test/'

	dataset = ImageFolder(test_path)
	label_list = dataset.classes

	matrix_label = []
	matrix_pred = []

	label_result = []
	top1 = []
	top2 = []
	top5 = []
	
	for label in tqdm(os.listdir(test_path)):
		test_folder = os.path.join(test_path, label)

		# 用于计算每一类的准确率
		accuracy_class = [0, 0, 0]

		for image_names in tqdm(os.listdir(test_folder)):
			imagefile = os.path.join(test_folder, image_names)
			img = Image.open(imagefile).convert('RGB')
			# Prediction
			pred = food_recognition(img)
			softmax_list = pred.tolist()[0]
			# pred_np = pred.detach().numpy()
			# e_x = np.exp(pred_np)
			# softmax = e_x / e_x.sum()
			# softmax_list = softmax[0, :].tolist()
			# print(len(softmax_list))

			# Accuracy
			accuracy_sample, pred_sample = accuracy(softmax_list, label, label_list, topk=(1,2,5))
			# print(accuracy_sample, pred_sample)
			accuracy_class = list_add(accuracy_class, accuracy_sample)

			matrix_label.append(label)
			matrix_pred.append(pred_sample)

		num = len(os.listdir(test_folder))
		Acc = [i/num for i in accuracy_class]

		pred_acc1 = Acc[0]
		pred_acc2 = Acc[1]
		pred_acc3 = Acc[2]

		top1.append(pred_acc1)
		top2.append(pred_acc2)
		top5.append(pred_acc3)
		label_result.append(label)

		print(40*'=')
		print(label+'TOP1:', pred_acc1)
		print(label+'TOP2:', pred_acc2)
		print(label+'TOP5:', pred_acc3)

	dataframe = pd.DataFrame({'label':matrix_label, 'pred':matrix_pred})
	dataframe.to_csv('result.csv', index=False)

	dataframe_acc = pd.DataFrame({'label':label_result, 'top1':top1, 'top2':top2, 'top5':top5})
	dataframe_acc.to_csv('acc.csv', index=False)
