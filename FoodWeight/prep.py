import os
import numpy as np
import pandas as pd

from tqdm import tqdm


# 寻找各个文件夹中reg.all的文件，并存放列表dst_list中
def findRegFile(rootPath, phase='TRAINING'):
	assert phase in ['TRAINING', 'TESTING', 'VALING']
	
	dst_list = []
	dataPath = os.path.join(rootPath, phase)
	print(dataPath)
	if not os.path.exists(dataPath):
		print('PATH NOT FOUND')
	else:
		for folder in os.listdir(dataPath):
			fileFolder = os.path.join(dataPath, folder)
			if not os.path.isfile(fileFolder):
				if 'cut' in os.listdir(fileFolder):
					file = fileFolder + '/cut/reg.all'
					if os.path.exists(file):
						if not os.path.getsize(file):
							continue
						dst_list.append(file)

	return dst_list


# 处理reg.all文件，找到图像路径-食物重量对，结果存在到list中
def prepReg(file_list, phase='TRAINING'):
	assert phase in ['TRAINING', 'TESTING', 'VALING']

	result_list = []
	for _, filePath in tqdm(enumerate(file_list)):
		# print(filePath)
		# if not os.path.getsize(filePath):
			# continue
		temp = []
		datas = pd.read_csv(filePath, header=None)
		for _, data in enumerate(datas[0]):
			result = data.split('\t')
			imgPath = result[0]
			if phase == 'TESTING':
				imgPath = imgPath.split(phase)[1]
			else:
				imgPath = imgPath.split('TRAINING')[1]
			temp.append(imgPath)
			imgWeight = float(result[1])
			temp.append(imgWeight)

			result_list.append(temp)

	return result_list
