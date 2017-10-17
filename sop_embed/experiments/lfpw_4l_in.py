from sop_embed.da import DenoisingAutoencoder
from sop_embed.tools import NonLinearity
from sop_embed.tools import CostType
from sop_embed.tools import ModelMLP
from sop_embed.tools import train_one_epoch_chuncks
from sop_embed.tools import theano_fns
from sop_embed.tools import sharedX_value
from sop_embed.tools import collect_stats_epoch
from sop_embed.tools import plot_stats
from sop_embed.tools import split_data_to_minibatchs_eval
from sop_embed.tools import split_data_to_minibatchs_embed
from sop_embed.tools import evaluate_model
from sop_embed.tools import StaticAnnealedWeightRate
from sop_embed.tools import StaticExponentialDecayWeightRate
from sop_embed.tools import StaticExponentialDecayWeightRateSingle
from sop_embed.tools import StaticAnnealedWeightRateSingle
from sop_embed.tools import print_stats_train
from sop_embed.learning_rule import AdaDelta
from sop_embed.learning_rule import Momentum

import theano.tensor as T
import theano
import numpy as np
import cPickle as pkl
import datetime as DT
import os
import inspect
import sys
import shutil
from random import shuffle

# Alexnet: 9216
# VGG16: 25088


def standardize(data):
    """
    Normalize the data with respect to finding the mean and the standard
    deviation of it and dividing by mean and standard deviation.
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    if sigma.nonzero()[0].shape[0] == 0:
        raise Exception("Std dev should not be zero")

    norm_data = (data - mu) / sigma
    return norm_data


if __name__ == "__main__":
    type_mod = ""
    if type_mod is "alexnet":
        dim_in = 9216
    if type_mod is "vgg_16":
        dim_in = 25088
    if type_mod is "vgg_19":
        dim_in = 9216
    if type_mod is "googlenet":
        dim_in = 9216
    faceset = "lfpw"
    fd_data = "../../inout/data/face/" + faceset + "_data/"
    path_valid = fd_data + type_mod + "valid.pkl"
    w, h = 50, 50
    if type_mod is not None and type_mod is not "":
        w, h = dim_in, 1
    input = T.fmatrix("x_input")
    output = T.fmatrix("y_output")

    # Create mixed data
    nbr_sup, nbr_xx, nbr_yy = 676, 0, 0
    id_data = type_mod + "ch_tr_" + str(nbr_sup) + '_' + str(nbr_xx) + '_' +\
        str(nbr_yy)
    # List train chuncks
    l_ch_tr = [
        fd_data + id_data + "_" + str(i) + ".pkl" for i in range(0, 1)]

    time_exp = DT.datetime.now().strftime('%m_%d_%Y_%H_%M_%s')
    fold_exp = "../../exps/" + faceset + "_" + time_exp
    if not os.path.exists(fold_exp):
        os.makedirs(fold_exp)
    nbr_layers = 4
    init_w_path = "../../inout/init_weights/" + str(nbr_layers) + '_' +\
        faceset + '_layers/'
    if not os.path.exists(init_w_path):
        os.makedirs(init_w_path)

    rnd = np.random.RandomState(1231)
    nhid_l0 = 1025
    nhid_l1 = 512
    nhid_l2 = 64

    # Create the AE in 1
    nvis, nhid = w*h, nhid_l0
    path_ini_params_l0 = init_w_path + "dae_w_l0_init_" + str(nvis) + '_' +\
        str(nhid) + ".pkl"
    dae_l0 = DenoisingAutoencoder(input,
                                  nvis=nvis,
                                  nhid=nhid,
                                  L1_reg=0.,
                                  L2_reg=1e-2,
                                  rnd=rnd,
                                  nonlinearity=NonLinearity.SIGMOID,
                                  cost_type=CostType.MeanSquared,
                                  reverse=False,
                                  corruption_level=0.2)

    if not os.path.isfile(path_ini_params_l0):
        dae_l0.save_params(path_ini_params_l0)
    else:
        dae_l0.set_params_vals(path_ini_params_l0)

    # Create the AE in 2
    nvis, nhid = nhid_l0, nhid_l1
    path_ini_params_l1 = init_w_path + "dae_w_l1_init_" + str(nvis) + '_' +\
        str(nhid) + ".pkl"
    dae_l1 = DenoisingAutoencoder(dae_l0.encode((input)),
                                  nvis=nhid_l0,
                                  nhid=nhid_l1,
                                  L1_reg=0.,
                                  L2_reg=1e-2,
                                  rnd=rnd,
                                  nonlinearity=NonLinearity.SIGMOID,
                                  cost_type=CostType.MeanSquared,
                                  reverse=False,
                                  corruption_level=0.01)

    if not os.path.isfile(path_ini_params_l1):
        dae_l1.save_params(path_ini_params_l1)
    else:
        dae_l1.set_params_vals(path_ini_params_l1)

    # Create the AE in 3
    nvis, nhid = nhid_l1, nhid_l2
    path_ini_params_l2 = init_w_path + "dae_w_l2_init_" + str(nvis) + '_' +\
        str(nhid) + ".pkl"
    dae_l2 = DenoisingAutoencoder(dae_l1.encode(dae_l0.encode((input))),
                                  nvis=nhid_l1,
                                  nhid=nhid_l2,
                                  L1_reg=0.,
                                  L2_reg=0.,
                                  rnd=rnd,
                                  nonlinearity=NonLinearity.TANH,
                                  cost_type=CostType.MeanSquared,
                                  reverse=False)

    if not os.path.isfile(path_ini_params_l2):
        dae_l2.save_params(path_ini_params_l2)
    else:
        dae_l2.set_params_vals(path_ini_params_l2)

    # Create the AE out
    nvis, nhid = 68*2, nhid_l2
    path_ini_params_l3 = init_w_path + "dae_w_l3_init_" + str(nvis) + '_' +\
        str(nhid) + ".pkl"
    dae_l3 = DenoisingAutoencoder(output,
                                  L1_reg=0.,
                                  L2_reg=1e-2,
                                  nvis=nvis,
                                  nhid=nhid,
                                  rnd=rnd,
                                  nonlinearity=NonLinearity.TANH,
                                  cost_type=CostType.MeanSquared,
                                  reverse=True)

    if not os.path.isfile(path_ini_params_l3):
        dae_l3.save_params(path_ini_params_l3)
    else:
        dae_l3.set_params_vals(path_ini_params_l3)
    # Create the network
    rng = np.random.RandomState(23455)

    layer0 = {
        "rng": rng,
        "n_in": w*h,
        "n_out": nhid_l0,
        "W": dae_l0.hidden.W,
        "b": dae_l0.hidden.b,
        "activation": NonLinearity.SIGMOID
        }
    layer1 = {
        "rng": rng,
        "n_in": nhid_l0,
        "n_out": nhid_l1,
        "W": dae_l1.hidden.W,
        "b": dae_l1.hidden.b,
        "activation": NonLinearity.SIGMOID
        }
    layer2 = {
        "rng": rng,
        "n_in": nhid_l1,
        "n_out": nhid_l2,
        "W": dae_l2.hidden.W,
        "b": dae_l2.hidden.b,
        "activation": NonLinearity.TANH
        }
    layer3 = {
        "rng": rng,
        "n_in": nhid_l2,
        "n_out": 68*2,
        "W": dae_l3.hidden.W_prime,
        "b": dae_l3.hidden.b_prime,
        "activation": NonLinearity.TANH
        }
    layers = [layer0, layer1, layer2, layer3]
#    dropout = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
#               float(sys.argv[4])]
    dropout = [0.0, 0.0, 0.0, 0.0]
    # number of the hidden layer just before the output ae. Default: None
    id_code = None
    model = ModelMLP(layers, input, l1_reg=0., l2_reg=0., reg_bias=False,
                     dropout=dropout, id_code=id_code)
    aes_in = [dae_l0]
    aes_out = []
    if id_code is not None:
        assert aes_out != []
    # Train
    # Data
    tr_batch_size = 10
    vl_batch_size = 8000

    with open(path_valid, 'r') as f:
        l_samples_vl = pkl.load(f)
    list_minibatchs_vl = split_data_to_minibatchs_eval(
        l_samples_vl, vl_batch_size)
    max_epochs = int(1000)
    lr_vl = 1e-3
    lr = sharedX_value(lr_vl, name="lr")
    # cost weights
    separate = True
    l_in = [sharedX_value(1., name="l_in"), sharedX_value(0.0, name="l_in2")]
    l_out = [sharedX_value(0., name="l_out")]
    l_sup = sharedX_value(1., name="l_sup")
    l_code = sharedX_value(0.0, name="l_code")
    if not separate:
        assert l_sup.get_value() + l_in.get_value() + l_out.get_value() == 1.
    if l_in[0].get_value() != 0. and aes_in == []:
        raise ValueError("You setup the l_in but no aes in found.")
    if l_out[0].get_value() != 0. and aes_out == []:
        raise ValueError("You setup the l_out but no aes out found.")
    # Train criterion
    cost_type = CostType.MeanSquared  # CostType.MeanSquared
    # Compile the functions
#    Momentum(0.9, nesterov_momentum=False,
#                           imagenet=False, imagenetDecay=5e-4,
#                           max_colm_norm=False)
    train_updates, eval_fn = theano_fns(
        model, aes_in, aes_out, l_in, l_out, l_sup, l_code, lr,
        cost_type,
        updaters={
            "all": Momentum(0.9, nesterov_momentum=False,
                            imagenet=False, imagenetDecay=5e-4,
                            max_colm_norm=False),
            "in": Momentum(0.9, nesterov_momentum=False,
                           imagenet=False, imagenetDecay=5e-4,
                           max_colm_norm=False),
            "out": Momentum(0.9, nesterov_momentum=False,
                            imagenet=False, imagenetDecay=5e-4,
                            max_colm_norm=False),
            "code": None},
        max_colm_norm=False, max_norm=15.0, eye=False)

    # How to update the weight costs
    updater_wc = StaticAnnealedWeightRate(anneal_end=500, anneal_start=0)
    updater_wc_in = StaticAnnealedWeightRateSingle(anneal_end=500, down=True,
                                                   init_vl=1., end_vl=0.,
                                                   anneal_start=100)
    updater_wc_in2 = StaticAnnealedWeightRateSingle(anneal_end=500, down=True,
                                                    init_vl=0.0, end_vl=0.,
                                                    anneal_start=400)
    updater_wc_out = StaticAnnealedWeightRateSingle(anneal_end=700, down=True,
                                                    init_vl=1., end_vl=0.,
                                                    anneal_start=100)
    # how to update the weight code
    # l_code_updater = StaticExponentialDecayWeightRateSingle(slop=20,
#                                                            anneal_start=0)
    to_update = {"l_in": True, "l_out": True}
    if aes_in == []:
        to_update["l_in"] = False
    if aes_out == []:
        to_update["l_out"] = False

    # Train
    i = 0
    # Stats
    train_stats = {"in_cost": [], "out_cost": [],
                   "all_cost": [], "tr_pure_cost": [], "code_cost": [],
                   "in_cost_mb": [], "out_cost_mb": [], "all_cost_mb": [],
                   "tr_pure_cost_mb": [], "error_tr": [], "error_vl": [],
                   "error_tr_mb": [], "error_vl_mb": [], "code_cost_mb": [],
                   "best_epoch": 0, "best_mb": 0}
    # tag
    if aes_in == [] and aes_out == []:
        tag = "sup"
    elif aes_in != [] and aes_out == []:
        tag = "sup + in"
    elif aes_in == [] and aes_out != []:
        tag = "sup + out"
    elif aes_in != [] and aes_out != []:
        tag = "sup + in + out"

    tag += ", data: " + faceset + " " + id_data
    # First evaluation on valid
    error_mn, _ = evaluate_model(list_minibatchs_vl, eval_fn)
    vl_error_begin = np.mean(error_mn)

    shutil.copy(inspect.stack()[0][1], fold_exp)
    l_ch_tr_vl = []
    for ch in l_ch_tr:
        with open(ch, 'r') as f:
            l_samples = pkl.load(f)
        l_ch_tr_vl.append(l_samples)

    shutil.copy(inspect.stack()[0][1], fold_exp)

    stop = False
    while i < max_epochs:
        stop = (i == max_epochs - 1)
        stats = train_one_epoch_chuncks(
            train_updates, eval_fn, l_ch_tr_vl,
            l_in, l_out, l_sup, l_code, list_minibatchs_vl,
            model, aes_in, aes_out, i, fold_exp, train_stats,
            vl_error_begin, tag, tr_batch_size, stop=stop)
        # Shuffle the minibatchs: to avoid periodic behavior.
#        for ts in xrange(100):
#            shuffle(l_ch_tr_vl)
        # Collect stats
        train_stats = collect_stats_epoch(stats, train_stats)
        # Print train stats
#        print_stats_train(train_stats, i, "", 0)
        # reduce the frequency of the disc access, it costs too much!!!
        if stop:
            # Plot stats: epoch
            plot_stats(train_stats, "epoch", fold_exp, tag)
            # Save stats
            with open(fold_exp + "/train_stats.pkl", 'w') as f_ts:
                pkl.dump(train_stats, f_ts)
        # Update the cost weights
        if aes_in != [] or aes_out != []:
#            updater_wc(l_sup, l_in, l_out, i, to_update)
            updater_wc_in(l_in[0], i)
            updater_wc_in2(l_in[1], i)
            updater_wc_out(l_out[0], i)
        print "\n", l_sup.get_value(), l_out[0].get_value()
        for el in l_in:
            print el.get_value(),
        print ""
        if id_code is not None:
            l_code_updater(l_code, i)
            print "l_code:", l_code.get_value()
        # Check the stopping criterion
        # [TODO]
        # Update lr
#        if (i % 1 == 0):
#            lr.set_value(np.cast[theano.config.floatX](lr.get_value()/1.0001))
#            print "lr:", lr.get_value()
        i += 1
        del stats
    # Eval
    cmd = "python evaluate_face_new_data.py " + str(faceset) + " " +\
        str(fold_exp) + " mlp"
#    with open("./" + str(time_exp) + ".py", "w") as python_file:
#        python_file.write("import os \n")
#        cmd2 = 'os.system("' + cmd + '")'
#        python_file.write(cmd2)
    os.system(cmd)
#    # std_data = standardize(x_data)
#    std_data = np.asarray(x_data, dtype=theano.config.floatX)
#
#    dae_l0.fit(learning_rate=9.96*1e-3,
#               shuffle_data=True,
#               data=std_data,
#               weights_file=weights_file_l0,
#               recons_img_file=None,
#               corruption_level=0.095,
#               batch_size=40,
#               n_epochs=2)
#
#    dae_l0_obj_out = open("dae_l0_obj.pkl", "wb")
#    pkl.dump(dae_l0, dae_l0_obj_out, protocol=pkl.HIGHEST_PROTOCOL)
#
#    dae_l0_out = dae_l0.encode((input))
#    dae_l0_h = dae_l0.encode(std_data)
#    dae_l0_h_fn = theano.function([], dae_l0_h)
#    dae_l1_in = dae_l0_h_fn()
#    dae_l1_in = np.asarray(dae_l1_in, dtype=theano.config.floatX)
#
#    dae_l1 = DenoisingAutoencoder(dae_l0_out,
#                                  L1_reg=1e-4,
#                                  L2_reg=6*1e-4,
#                                  nvis=nhid_l0,
#                                  nhid=nhid_l1,
#                                  rnd=rnd,
#                                  reverse=True)
#
#    dae_l1.fit(learning_rate=0.95*1e-2,
#               data=dae_l1_in,
#               shuffle_data=True,
#               recons_img_file=None,
#               weights_file=weights_file_l1,
#               corruption_level=0.1,
#               batch_size=25,
#               n_epochs=2)  # 1400
#
#    dae_l1_obj_out = open("dae_l1_obj.pkl", "wb")
#    pkl.dump(dae_l1, dae_l1_obj_out, protocol=pkl.HIGHEST_PROTOCOL)
