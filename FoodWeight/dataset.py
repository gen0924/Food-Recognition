import os
import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from prep import findRegFile, prepReg
from PIL import Image


class FWDataset(Dataset):
	"""docstring for  FWDataset"""
	def __init__(self, phase='TRAINING'):
		assert phase in ['TRAINING', 'TESTING', 'VALING']

		super(FWDataset, self).__init__()
		self.phase = phase
		self.rootPath = '/media/liu/Elements/'
		self.file_list = findRegFile(self.rootPath, phase=phase)
		self.result_list = prepReg(self.file_list, phase=phase)
		self.transform = transforms.Compose([
											transforms.Resize(size=(256, 256)),
											transforms.ToTensor(),
											transforms.Normalize(
												mean=[0.555, 0.416, 0.362],
												std=[0.770, 0.836, 0.905])
											])


	def __getitem__(self, index):
		dataInfo = self.result_list[index]
		label = dataInfo[1] # 食物重量
		imgFile = dataInfo[0] # 图像路径
		imgPath = self.rootPath + self.phase + imgFile
		imgInfo = Image.open(imgPath).convert('RGB')
		img = self.transform(imgInfo)

		return img, label


	def __len__(self):
		 
		 return len(self.result_list)