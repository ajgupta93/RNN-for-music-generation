import numpy as np
from keras.utils.np_utils import *
import matplotlib.pyplot as plt

def one_hot_encoding(labels, n_classes):
	return to_categorical(labels, nb_classes=n_classes)

def plot_result( hist, fname ):
	
	epochs = np.array(range(len(hist['acc']))) + 1

	
	fig = plt.figure(1)
	
	plt.subplot(211)
	plt.plot(epochs, hist['acc'], 'b-', label = "Training Accuracy")
	plt.plot(epochs, hist['val_acc'], 'r-', label = " Validation Accuracy")
	plt.ylabel('Accuracy')
	plt.xlabel('#Epochs')
	plt.legend()
	
	plt.subplot(212)
	plt.plot(epochs, hist['loss'], 'b-', label = "Training Loss")
	plt.plot(epochs, hist['val_loss'], 'r-', label = "Validation Loss")
	plt.legend()
	plt.ylabel('Loss')
	plt.xlabel('#Epochs')
	plt.show()

	fig.savefig("../results/" + fname + ".jpg")