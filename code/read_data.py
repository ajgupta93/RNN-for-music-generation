import numpy as np
import pdb
import pickle

def convert_chars_nums(abc_input):
	char_dict = {}
	all_music = []

	i = 0
	for char in abc_input:

		if char not in char_dict:
			char_dict[char] = i
			i += 1	

		all_music.append(char_dict[char])

	
	inv_char_dict = {v: k for k, v in char_dict.iteritems()}
	return inv_char_dict, np.array(all_music)

def read_file(filename):
	raw_text = open(filename).read()

	#remove <end> tag because we have <start> tag too so <end> tag is redundant
	raw_text = raw_text.replace('<end>', '')
	
	#hard code: replace <start> tag with one character which is not used by input music
	raw_text = raw_text.replace('<start>', '$')
	
	return convert_chars_nums(raw_text)

if __name__ == '__main__':
	char_dict, all_music = read_file('../data/input.txt')

	np.save('../data/all_music.npy', all_music)
	pickle.dump(char_dict, open('../data/char_dict.p', 'wb'))
