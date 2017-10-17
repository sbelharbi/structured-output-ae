from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.optimizers import SGD

import theano.tensor as T
import datetime as DT
import os
import sys
import cPickle as pkl
import numpy as np
from sop_embed.tools import split_data_to_minibatchs_eval
from sop_embed.tools import split_data_to_minibatchs_embed


def baseline_model():
    # create model
    input_shape = (1, 50, 50)
    model = Sequential()
    model.add(Conv2D(16, (3, 3),
                 activation='sigmoid',
                 strides=(1, 1),
                 data_format='channels_first',
                 padding='same',
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Conv2D(48, kernel_size=(3, 3),
                 activation='sigmoid',
                 strides=(1, 1),
                 data_format="channels_first",
                 padding="same",
                 input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='sigmoid',
                 strides=(1, 1),
                 data_format="channels_first",
                 padding="same",
                 input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
    model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='sigmoid',
                 strides=(1, 1),
                 data_format="channels_first",
                 padding="same",
                 input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(68*2, activation='tanh'))
    # Compile model
    sgd = SGD(lr=1e-4, momentum=0.9, decay=1e-6, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return model

model = baseline_model()

type_mod = ""
faceset = "lfpw"
fd_data = "../../inout/data/face/" + faceset + "_data/"
path_valid = fd_data + type_mod + "valid.pkl"
w, h = 50, 50
if type_mod is not None and type_mod is not "":
    w, h = dim_in, 1
input = T.tensor4("x_input")
output = T.fmatrix("y_output")

# Create mixed data
nbr_sup, nbr_xx, nbr_yy = 676, 0, 0
id_data = type_mod + "ch_tr_" + str(nbr_sup) + '_' + str(nbr_xx) + '_' +\
    str(nbr_yy)
# List train chuncks
l_ch_tr = [
    fd_data + id_data + "_" + str(i) + ".pkl" for i in range(0, 1)]

time_exp = DT.datetime.now().strftime('%m_%d_%Y_%H_%M_%s')
fold_exp = "../../exps/" + faceset + "_deep_convaeIN_" + time_exp + "_in"
if not os.path.exists(fold_exp):
    os.makedirs(fold_exp)


with open(path_valid, 'r') as f:
    l_samples_vl = pkl.load(f)
xvl, yvl = l_samples_vl["x"], l_samples_vl["y"]

model = baseline_model()
best_err_vl = 100.

tr_batch_size = 10
vl_batch_size = 8000

with open(path_valid, 'r') as f:
    l_samples_vl = pkl.load(f)
    # convert to 3D
    nbr_xx = l_samples_vl["x"].shape[0]
    l_samples_vl["x"] = l_samples_vl["x"].reshape((nbr_xx, 1, h, w))
    xvl, yvl = l_samples_vl["x"], l_samples_vl["y"]

max_epochs = int(1000)
lr_vl = 1e-4
l_ch_tr_vl = []
for ch in l_ch_tr:
    with open(ch, 'r') as f:
        l_samples = pkl.load(f)
        nbr_xx = l_samples["x"].shape[0]
        l_samples["x"] = l_samples["x"].reshape((nbr_xx, 1, h, w))
    # l_ch_tr_vl.append(l_samples)
    list_minibatchs_tr = split_data_to_minibatchs_embed(
            l_samples, tr_batch_size, share=False, sharpe=False)

def get_params(mo):
    para = []
    for la in mo.layers:
        par = la.get_weights()

        para += par
    return para


def set_params(mo, bparams):
    i = 0
    for la in mo.layers:
        we = bparams[i:i+2]
        print len(we)
        la.set_weights(we)
        i += 2
    return mo

#with open("best_model_keras.pkl", 'r') as f:
#    b_params = pkl.load(f)
#
#model = set_params(model, b_params)
#out = model.predict(xvl, batch_size=xvl.shape[0], verbose=0)
#error = np.mean(np.mean(np.power(out - yvl, 2), axis=1))
#print "Error vl", error
#sys.exit()

#init_p = get_params(model)
#with open("init_keras_param.pkl", 'w') as f:
#    pkl.dump(init_p, f)
kk = 0
while i <  max_epochs:
    print "\n Epoch", i
    for mn in list_minibatchs_tr:
        x, y = mn["sup"]["x"], mn["sup"]["y"]
        loss = model.train_on_batch(x, y)
        del x
        del y
        # valid
        out = model.predict(xvl, batch_size=xvl.shape[0], verbose=0)
        error = np.mean(np.mean(np.power(out - yvl, 2), axis=1))
        if error < best_err_vl:
            if (kk % 100) == 0:
                best_err_vl = error
                init_p = get_params(model)
                with open("best_model_keras.pkl", 'w') as f:
                    pkl.dump(init_p, f)
        kk += 1
        print "\r loss", loss, " vl error:", error, " epoch", i
    i += 1
