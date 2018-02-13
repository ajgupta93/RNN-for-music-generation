import keras
from keras.layers.recurrent import *
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import *
from keras.callbacks import *
import utility as util
import pdb

def get_optimizer(name = 'adagrad', l_rate = 0.0001, dec = 0.0, b_1 = 0.9, b_2 = 0.999, mom = 0.5, rh = 0.9):
	eps = 1e-8
	
	adam = Adam(lr = l_rate, beta_1 = b_1, beta_2 = b_2, epsilon = eps, decay = dec)
	sgd = SGD(lr = l_rate, momentum = mom, decay = dec, nesterov = True)
	rmsp = RMSprop(lr = l_rate, rho = rh, epsilon = eps, decay = dec)
	adagrad = Adagrad(lr = l_rate, epsilon = eps, decay = dec)
	
	optimizers = {'adam': adam, 'sgd':sgd, 'rmsp': rmsp, 'adagrad': adagrad}

	return optimizers[name]

def build_simplernn_model(input_units, hidden_units = 100, af = 'tanh', optimizer = 'adam', T = 1, p = 0.0, l_rate = 0.001, dec = 0.0, b_1 = 0.9, b_2 = 0.999, mom = 0.5, rh = 0.9):
	#define network architecture
	model = Sequential()
	model.add(SimpleRNN(input_dim = input_units, output_dim = hidden_units, activation = af, return_sequences = True, dropout_U = p))
	model.add(Lambda(lambda x: x * 1.0/ T))
	model.add(Dense(output_dim = input_units, activation = 'softmax'))

	#compile model
	opt = get_optimizer(optimizer, l_rate, dec, b_1, b_2, mom, rh)
	model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')

	print model.summary()

	return model

def train_rnn(model, train_data, timesteps = 25, b_size = 5, n_epoch = 10):
	#lets make length of train data to be a multiple of timesteps
	#don't shuffle, as sequence matters
	l = len(train_data)
	train_data = train_data[:l - l%timesteps,]

	train_data = np.reshape(train_data, (-1, timesteps, len(train_data[0])))
	
	#make length of train_data to be multiple of b_size
	l = len(train_data)
	train_data = train_data[:l - l%b_size]
	print l, len(train_data)
	#Predict the next character in the sequence
	X = train_data[:,:-1,]
	Y = train_data[:,1:,]

	#setting stateful to true
	model.layers[0].stateful = True
	model.layers[0].batch_input_shape = np.shape(X)

	#Callbacks
	hist = History()
	checkpoint = ModelCheckpoint('../model/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
	callbacks_list = [hist, checkpoint]

	
	history = model.fit(X, Y, batch_size=b_size, nb_epoch=n_epoch, verbose=1, callbacks=callbacks_list,
		validation_split=0.2, validation_data=None, shuffle=False, class_weight=None, sample_weight=None, initial_epoch=0)

	print hist.history
	return hist

def generate(model, pattern, char_dict, total_chars=1500):
	inv_char_dict = {v:k for k,v in char_dict.iteritems()}
	music_str = pattern

	#setting stateful to true
	model.layers[0].stateful = True

	pattern = [inv_char_dict[c] for c in pattern]
	#generate characters
	for i in range(0, total_chars):
		pattern = util.one_hot_encoding(pattern, len(char_dict.keys()))
		pattern = np.reshape(pattern, (1,)+pattern.shape)
		prediction = model.predict(pattern, verbose = 0)
	
		index = np.random.choice(range(len(char_dict)), p = prediction[0,np.size(prediction,axis=1)-1,:])
		
		pattern = index
		music_str += char_dict[index]

	return music_str
		

def generate_music_soft(model, pattern, char_dict, total_chars = 1500):
    inv_char_dict = {v:k for k,v in char_dict.iteritems()}
    music_str = pattern
    
    for i in range(total_chars):
        enc_pat = [inv_char_dict[c] for c in pattern]
        oneHot_pattern = util.one_hot_encoding(enc_pat,len(char_dict.keys()))
        prediction = model.predict(np.reshape(oneHot_pattern, (1,) + oneHot_pattern.shape),verbose =0)
        
        index = np.random.choice(range(len(char_dict)), p = prediction[0,np.size(prediction,axis=1)-1,:])
        
        pattern = (char_dict[index])
        music_str = music_str + pattern
        pattern = music_str
        
    return music_str
