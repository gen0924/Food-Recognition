import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import accuracy
from models import *
from dataset import *


def load_checkpoint(ckpt_path, pretrained=True):

	num_classes = 583
	num_attentions = 32
	feature_net = inception_v3(pretrained=True)
	net = WSDAN(num_classes=num_classes, M=num_attentions, net=feature_net)

	checkpoint = torch.load(ckpt_path)
	state_dict = checkpoint['state_dict']
	net.load_state_dict(state_dict)

	return net


image_size = (512, 512)
num_classes = 583
num_attentions = 32


# Load dataset
testPath = '/data1/pub/dataset/food_583_splited/test/'
normalize = transforms.Normalize(mean=[0.6301, 0.5246, 0.3962],
	std=[0.2747, 0.2881, 0.3135])

test_dataset = ImageFolder(testPath, transforms.Compose([
	transforms.Resize(image_size),
	transforms.ToTensor(),
	normalize
	]))

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model = load_checkpoint(ckpt_path='012.ckpt', pretrained=True)
model.to(torch.device('cuda'))
model.eval()

theta_c = 0.5
crop_size = (256, 256)

batches = 0
epoch_loss = 0
epoch_acc = np.array([0,0,0], dtype='float')


with torch.no_grad():
	for i, (x, y) in enumerate(test_dataloader):
		x = x.to(torch.device('cuda'))
		y = y.to(torch.device('cuda'))
		print(y, type(y))

		y_pred_raw, feature_matrix, attention_map = model(x)
		# Object Location and Refinement
		crop_mask = F.upsample_bilinear(attention_map, size=image_size) > theta_c
		crop_images = []
		for batch_index in range(crop_mask.size(0)):
			nonozero_indices = torch.nonzero(crop_mask[batch_index, 0, ...])
			height_min = nonozero_indices[:, 0].min()
			height_max = nonozero_indices[:, 0].max()
			width_min = nonozero_indices[:, 1].min()
			width_max = nonozero_indices[:, 1].max()
			crop_images.append(F.upsample_bilinear(x[batch_index:batch_index+1, :, height_min:height_max, width_min:width_max], size=crop_size))
		crop_images = torch.cat(crop_images, dim=0)

		y_pred_crop, _, _ = model(crop_images)

		y_pred = (y_pred_raw + y_pred_crop) / 2
		print(y_pred, type(y_pred))

		epoch_acc = epoch_acc + accuracy(y_pred, y ,topk=(1,2,5))

		batches += 1

epoch_acc /= batches

print(epoch_acc)
