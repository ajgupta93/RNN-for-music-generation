import numpy as np
import pickle
import utility as util
import rnn_model as model
import pdb 
from sys import argv
import argparse

def command_line_initialization():
	ap = argparse.ArgumentParser()
	default_arguments = {'-hid': '100', # number of hidden units
						 '-af': 'tanh', # activation-function -> "tanh", "sigmoid", "relu", "hard_sigmoid", "linear", "softplus"
						 '-opt': 'adam', # optimizer -> "adam", "sgd", "rmsp", "adagrad"
						 '-ts': '25', # timesteps -> [25-30]
						 '-bs': '5', # batch-size -> [5 - 8] (not sure)
						 '-e': '10', # number of epochs
						 '-T': '1.0', # T -> 0.5, 1, 2
						 '-dr': '0.0', # dropout -> 0.1, 0.2, 0.3
						 '-lr': '0.001', # learning rate
						 '-dec': '0.000001', # decay parameter
						 '-b_1': '0.9', # beta_1 for adam
						 '-b_2': '0.999', # beta_2 for adam
						 '-m': '0.5', # momentum for sgd
						 '-rho': '0.9', # rho for rmsp
						 '-saveas': 'plot'} # name of filename to save image
	
	for current_argument in default_arguments:
		ap.add_argument(current_argument, default = default_arguments[current_argument])

	pa = ap.parse_args()
	return int(pa.hid), pa.af, pa.opt, int(pa.ts), int(pa.bs), int(pa.e), float(pa.T), float(pa.dr), float(pa.lr), float(pa.dec), float(pa.b_1), float(pa.b_2), float(pa.m), float(pa.rho), pa.saveas


def load_data():
	char_dict = pickle.load(open('../data/char_dict.p', 'rb'))
	all_music = np.load('../data/all_music.npy')

	all_music_encoded = util.one_hot_encoding(all_music, len(char_dict.keys()))

	return char_dict, all_music_encoded


# Call python file as python main.py -hid <#num hidden units> [so on and so forth]

if __name__ == '__main__':
	
	hidden, af, optimizer, timesteps, batch_size, nb_epochs, T, p, l_rate, decay, b_1, b_2, mom, rho, fname = command_line_initialization()

	print hidden, af, optimizer, timesteps, batch_size, nb_epochs, T, p, l_rate, decay, b_1, b_2, mom, rho, fname
	
	#load data
	char_dict, all_music_encoded = load_data()
	
	input_dim = len(char_dict.keys())

	#get model
	rnn = model.build_simplernn_model( input_dim, hidden, af, optimizer, T, p, l_rate, decay, b_1, b_2, mom, rho )

	#Train model
	hist = model.train_rnn( rnn, all_music_encoded, timesteps, batch_size, nb_epochs )
	util.plot_result( hist.history, fname )

	#Generation of music

	#load trained weights from file to model
	#rnn.load_weights("../model/weights.09-2.05.hdf5")

	#pick a random input pattern as our seed sequence, then generate music character by character
	prime_text = '$' #K:F\r\n X:2\r\n'
	
	generated_music = model.generate_music_soft(rnn, prime_text, char_dict)

	text_file = open("../music/music_soft.txt", "w")
	text_file.write("%s" % generated_music)
	text_file.close()
	