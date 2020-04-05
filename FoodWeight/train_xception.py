import os
import copy
import time
import torch
import logging
import warnings
import torchvision
import numpy as np
import torch.nn as nn
import torch.functional as F

from torch.utils.data import DataLoader
from utils import p20, percentage_error 
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from optparse import OptionParser
from xception import xception
from dataset import FWDataset
from tqdm import tqdm


print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

logging.basicConfig(format='%(message)s', level=logging.INFO)
warnings.filterwarnings('ignore')


parser = OptionParser()
parser.add_option('--workers', dest='workers', default=4, type='int')
parser.add_option('--epochs', dest='epochs', default=100, type='int')
parser.add_option('--batch_size', dest='batch_size', default='160', type='int')
parser.add_option('--lr', dest='lr', default=1e-3, type='float')
parser.add_option('--validation_step', dest='validation_step', default=1, type='int')
parser.add_option('--save_dir', dest='save_dir', default='./model_result', type='str')
(options, args) = parser.parse_args()


def train_model(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs, device):
	val_acc_history = []
	validation_step = 1
	test_step = 5
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = float('inf')
	best_PE = 0.0
	best_P20E = 0.0

	for epoch in range(num_epochs):
		logging.info('INFO:TRAIN Epoch {:0>3d}/{:0>3d}'.format(epoch + 1, num_epochs))
		model.train()

		running_loss = 0.0
		pc_error = 0.0
		p20_error = 0.0
		epoch_time_train = time.time()

		# Interate over data:
		for inputs, labels in tqdm(train_dataloader):
			inputs = inputs.to(device)
			labels = labels.to(device).float()

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward
			outputs = model(inputs).squeeze()

			loss = criterion(outputs, labels)
			error1 = percentage_error(labels, outputs)
			error2 = p20(labels, outputs)

			# _, preds = torch.max(outputs, 1)
			loss.backward()
			optimizer.step()

			# statistics
			pc_error += error1.item() * inputs.size(0)
			p20_error += error2 * inputs.size(0)
			running_loss += loss.item() * inputs.size(0)

		epoch_loss = running_loss / len(train_dataloader.dataset) 
		epoch_PE = pc_error / len(train_dataloader.dataset)
		epoch_P20 = p20_error / len(train_dataloader.dataset)
		
		train_time = time.time() - epoch_time_train
		logging.info('INFO: Loss: {:.4f} trainPercentageError: {:.4f} trainP20Error: {:.4f} trainTime: {:.3f}'.format(epoch_loss, epoch_PE, epoch_P20, train_time))

		if (epoch + 1) % validation_step == 0:
			logging.info('INFO:VALID Epoch {:0>3d}/{:0>3d}'.format(epoch + 1, num_epochs))
			val_loss, val_PE, val_P20E, val_time = val_model(model, val_dataloader, device)
			logging.info('INFO: Loss: {:.4f} valPencentageError: {:.4f} valP20Error: {:.4f} valTime: {:.3f}'.format(val_loss, val_PE, val_P20E, val_time))

			if val_loss < best_loss:
				best_PE = val_PE
				best_P20E = val_P20E
				best_model_wts = copy.deepcopy(model.state_dict())

		scheduler.step(val_loss)
		model.load_state_dict(best_model_wts)

		if (epoch + 1) % test_step == 0:
			logging.info('INFO:TEST Epoch {:0>3d}/{:0>3d}'.format(epoch + 1, num_epochs))
			test_loss, test_PE, test_P20E, test_time = test_model(model, test_dataloader, device)
			logging.info('INFO: Loss: {:.4f} testPencentageError: {:.4f} testP20Error: {:.4f} Time: {:.3f}'.format(test_loss, test_PE, test_P20E, test_time))

		# save model
		model_name = 'FoodWeightModel.pth'
		if not os.path.exists(options.save_dir):
			os.mkdir(options.save_dir)
		model_path = os.path.join(options.save_dir, model_name)
		torch.save(model, model_path)


def val_model(model, val_dataloader, device):
	
	running_loss = 0.0
	pe_error = 0.0
	p20_error = 0.0
	startTime = time.time()

	with torch.no_grad():
		model.eval()

		for inputs, labels in tqdm(val_dataloader):
			inputs = inputs.to(device)
			labels = labels.to(device).float()

			outputs = model(inputs).squeeze()

			error1 = percentage_error(labels, outputs)
			error2 = p20(labels, outputs)
			loss = nn.L1Loss()(outputs, labels)
			
			pe_error += error1 * inputs.size(0)
			p20_error += error2 * inputs.size(0) 
			running_loss += loss.item() * inputs.size(0)

		epoch_percentage_error = pe_error / len(val_dataloader.dataset)
		epoch_p20_error = p20_error / len(val_dataloader.dataset)
		epoch_loss = running_loss / len(val_dataloader.dataset)

	val_time = time.time() - startTime

	return epoch_loss, epoch_percentage_error, epoch_p20_error, val_time


def test_model(model, test_dataloader, device):
	
	running_loss = 0.0
	pe_error = 0.0
	p20_error = 0.0
	startTime = time.time()

	with torch.no_grad():
		model.eval()

		for inputs, labels in tqdm(test_dataloader):
			inputs = inputs.to(device)
			labels = labels.to(device).float()

			outputs = model(inputs).squeeze()

			error1 = percentage_error(labels, outputs)
			error2 = p20(labels, outputs)
			loss = nn.L1Loss()(outputs, labels)
			
			pe_error += error1 * inputs.size(0)
			p20_error += error2 * inputs.size(0) 
			running_loss += loss.item() * inputs.size(0)

		epoch_percentage_error = pe_error / len(test_dataloader.dataset)
		epoch_p20_error = p20_error / len(test_dataloader.dataset)
		epoch_loss = running_loss / len(test_dataloader.dataset)

	test_time = time.time() - startTime

	return epoch_loss, epoch_percentage_error, epoch_p20_error, test_time


def main():
	#############################
	# Load Dataset
	#############################
	train_dataset = FWDataset(phase = 'TRAINING')
	val_dataset = FWDataset(phase = 'VALING')
	test_dataset = FWDataset(phase = 'TESTING')

	train_dataloader = DataLoader(
								dataset = train_dataset,
								batch_size = options.batch_size,
								num_workers = options.workers,
								shuffle = False,
								pin_memory = False
								)

	val_dataloader = DataLoader(
								dataset = val_dataset,
								batch_size = options.batch_size,
								num_workers = options.workers,
								shuffle = False,
								pin_memory = False
								)

	test_dataloader = DataLoader(
								dataset = test_dataset,
								batch_size = options.batch_size,
								num_workers = options.workers,
								shuffle = False,
								pin_memory = False
								)

	#############################
	# Model
	#############################
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model_ft = xception(pretrained='imagenet')
	model_ft = model_ft.to(device)
	params_to_update = model_ft.parameters()

	#############################
	# Optimizer and LR Scheduler
	#############################
	optimizer = torch.optim.SGD(params_to_update, lr=options.lr, momentum=0.9, weight_decay=1e-5)
	#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, min_lr=1e-5, eps=1e-8)
	loss = nn.L1Loss() # MAE

	#############################
	# Train and Validation
	#############################
	train_model(model_ft, train_dataloader, val_dataloader, test_dataloader, loss, optimizer, scheduler, num_epochs=options.epochs, device=device)


if __name__ == '__main__':
	main()
