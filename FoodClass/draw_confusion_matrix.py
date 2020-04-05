import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix

dataset_path = '/home/liu/icarbonx/food-recognition/food_583_splited/test/'
dataset = ImageFolder(dataset_path)
labels = dataset.classes

csv_file = './result.csv'
csv1 = pd.read_csv(csv_file)
y_labels = csv1.label
y_preds = csv1.pred

y_true = y_labels.tolist()
y_pred = y_preds.tolist()

tick_marks = np.array(range(len(labels))) - 0.5


def plot_confusion_matrix(cm, title='Food confusion Matrix', cmap=plt.cm.Blues):

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	xlocation = np.array(range(len(labels)))
	plt.xticks(xlocation, labels, rotation=90)
	plt.yticks(xlocation, labels)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
# print (cm_normalized)
plt.figure(figsize=(60,60), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
	c = cm_normalized[y_val][x_val]
	if c > 0.1:
		plt.text(x_val, y_val,'%0.3f'%(c), color='red', fontsize=2, va='center', ha='center')

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

# show confusion matrix
plt.savefig('./confusion_matrix.pdf', format='pdf')
plt.show()