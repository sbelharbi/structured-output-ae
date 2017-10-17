from sop_embed.da import DenoisingAutoencoder
from sop_embed.tools import NonLinearity
from sop_embed.tools import CostType
from sop_embed.tools import ModelCNN
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
from sop_embed.da import ConvolutionalAutoencoder


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
import copy

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
    fold_exp = "../../exps/" + faceset + "_deep_convaeIN_" + time_exp
    if not os.path.exists(fold_exp):
        os.makedirs(fold_exp)
    nbr_layers = 5
    init_w_path = "../../inout/init_weights/deep_conv_ae_IN_" +\
        str(nbr_layers) + '_' + faceset + '_layers/'
    if not os.path.exists(init_w_path):
        os.makedirs(init_w_path)

    rnd = np.random.RandomState(1231)
    nhid_l0 = 100
    nhid_l1 = 64
    # nhid_l2 = 64

    # Deep conv.ae.IN

    # configure encoder
    encode_cae_l0 = {"type": "conv",
                     "rng": rnd,
                     "filter_shape": (4, 3, 3),
                     "activation": "sigmoid",
                     "padding": (1, 1),
                     "W": None,
                     "b": None,
                     "b_v": 0.,
                     "stride": (1, 1)}
    encode_cae_l1 = {"type": "downsample",
                     "poolsize": (2, 2),
                     "ignore_border": False,
                     "mode": "max"}
    # Dim: 16 * 25x25.
    encode_cae_l2 = {"type": "conv",
                     "rng": rnd,
                     "filter_shape": (8, 3, 3),
                     "activation": "sigmoid",
                     "padding": (1, 1),
                     "W": None,
                     "b": None,
                     "b_v": 0.,
                     "stride": (1, 1)}
    encode_cae_l3 = {"type": "downsample",
                     "poolsize": (2, 2),
                     "ignore_border": True,
                     "mode": "max"}
#    # Dim: 8 * 12x12.

    encoder_config = [encode_cae_l0, encode_cae_l1, encode_cae_l2,
                      encode_cae_l3]

    # configure decoder
    decode_cae_l0 = {"type": "conv",
                     "rng": rnd,
                     "filter_shape": (4, 3, 3),
                     "activation": "sigmoid",
                     "padding": (1, 1),
                     "W": None,
                     "b": None,
                     "b_v": 0.,
                     "stride": (1, 1)}
    decode_cae_l1 = {"type": "upsample",
                     "ratio": 2,
                     "use_1D_kernel": False}
#    # Dim: 8 * 24x24
    decode_cae_l2 = {"type": "conv",
                     "rng": rnd,
                     "filter_shape": (2, 3, 3),
                     "activation": "sigmoid",
                     "padding": (1, 1),
                     "W": None,
                     "b": None,
                     "b_v": 0.,
                     "stride": (1, 1)}
    decode_cae_l3 = {"type": "upsample",
                     "ratio": 2,
                     "use_1D_kernel": False}
    # Dim: 16 * 48x48
    decode_cae_l4 = {"type": "conv",
                     "rng": rnd,
                     "filter_shape": (1, 3, 3),
                     "activation": "sigmoid",
                     "padding": (2, 2),
                     "W": None,
                     "b": None,
                     "b_v": 0.,
                     "stride": (1, 1)}
    # Dim: 1 * 50x50
    decoder_config = [decode_cae_l0, decode_cae_l1, decode_cae_l2,
                      decode_cae_l3, decode_cae_l4]
    crop_size = [1, h, w]
    batch_size = 10  # must be the same.
    tr_batch_size = 10
    vl_batch_size = 8000
    deep_conv_ae_in = ConvolutionalAutoencoder(input=input,
                                               encoder_config=encoder_config,
                                               decoder_config=decoder_config,
                                               crop_size=crop_size,
                                               cost_type=CostType.MeanSquared,
                                               l1_reg=0., l2_reg=1e-3,
                                               reg_bias=False)
    id_deep_conv_ae_in = ""
    deep_conv_contact = encoder_config + decoder_config
    for e in deep_conv_contact:
        for k in e.keys():
            if k not in ["rng", "W", "b"]:
                id_deep_conv_ae_in += str(e[k]) + '|'
        # id_deep_conv_ae += "+"

    id_deep_conv_ae_in = id_deep_conv_ae_in.replace(" ", "")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace("(", "")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace(")", "")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace("conv", "cv")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace("False", "F")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace("True", "T")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace("sigmoid", "sig")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace("downsample", "dw")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace("upsample", "up")
    id_deep_conv_ae_in = id_deep_conv_ae_in.replace("max", "mx")
    print len(id_deep_conv_ae_in), id_deep_conv_ae_in
    path_ini_params_deep_conv_ae_in = init_w_path +\
        "deep_conv_ae_init_" + id_deep_conv_ae_in + ".pkl"
    if not os.path.isfile(path_ini_params_deep_conv_ae_in):
        deep_conv_ae_in.save_params(path_ini_params_deep_conv_ae_in)
    else:
        deep_conv_ae_in.set_params_vals(path_ini_params_deep_conv_ae_in)
    # Create the AE in 1
    nvis, nhid = deep_conv_ae_in.encoder.output_size_flatten, nhid_l0
    path_ini_params_l0 = init_w_path + "dae_w_l0_init_" + str(nvis) + '_' +\
        str(nhid) + ".pkl"
    dae_l0 = DenoisingAutoencoder(deep_conv_ae_in.encoder.output.flatten(2),
                                  nvis=nvis,
                                  nhid=nhid,
                                  L1_reg=0.,
                                  L2_reg=0.,
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
    dae_l1 = DenoisingAutoencoder(dae_l0.encode(),
                                  nvis=nhid_l0,
                                  nhid=nhid_l1,
                                  L1_reg=0.,
                                  L2_reg=0.,
                                  rnd=rnd,
                                  nonlinearity=NonLinearity.TANH,
                                  cost_type=CostType.MeanSquared,
                                  reverse=False,
                                  corruption_level=0.01)

    if not os.path.isfile(path_ini_params_l1):
        dae_l1.save_params(path_ini_params_l1)
    else:
        dae_l1.set_params_vals(path_ini_params_l1)

    # Create the AE in 3
#    nvis, nhid = nhid_l1, nhid_l2
#    path_ini_params_l2 = init_w_path + "dae_w_l2_init_" + str(nvis) + '_' +\
#        str(nhid) + ".pkl"
#    dae_l2 = DenoisingAutoencoder(dae_l1.encode(dae_l0.encode()),
#                                  nvis=nhid_l1,
#                                  nhid=nhid_l2,
#                                  L1_reg=0.,
#                                  L2_reg=0.,
#                                  rnd=rnd,
#                                  nonlinearity=NonLinearity.TANH,
#                                  cost_type=CostType.MeanSquared,
#                                  reverse=False)
#
#    if not os.path.isfile(path_ini_params_l2):
#        dae_l2.save_params(path_ini_params_l2)
#    else:
#        dae_l2.set_params_vals(path_ini_params_l2)

    # Create the AE out
    nvis, nhid = 68*2, nhid_l1
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

    # Plugin the Deep-conv-ae-in. Take the params of the encoder.
    # we consider the encoder as a single layer.
    layer00 = copy.deepcopy(encoder_config)
    # Now take its params.
    for i in range(len(layer00)):
        el = layer00[i]
        if el["type"] == "conv":  # parametric layer!!!!
            layer00[i]["W"] = deep_conv_ae_in.encoder.layers[i].W
            layer00[i]["b"] = deep_conv_ae_in.encoder.layers[i].b

    layer00 = {"type": "deep_conv_ae_in",
               "layer": layer00}
    layer0 = {
        "rng": rng,
        "n_in": deep_conv_ae_in.encoder.output_size_flatten,
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
        "activation": NonLinearity.TANH
        }
#    layer2 = {
#        "rng": rng,
#        "n_in": nhid_l1,
#        "n_out": nhid_l2,
#        "W": dae_l2.hidden.W,
#        "b": dae_l2.hidden.b,
#        "activation": NonLinearity.TANH
#        }
    layer3 = {
        "rng": rng,
        "n_in": nhid_l1,
        "n_out": 68*2,
        "W": dae_l3.hidden.W_prime,
        "b": dae_l3.hidden.b_prime,
        "activation": NonLinearity.TANH
        }
    layers = [layer00, layer0, layer1, layer3]
#    dropout = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
#               float(sys.argv[4])]
    dropout = [0.0, 0.0, 0.0, 0.0, 0.0]
    # number of the hidden layer just before the output ae. Default: None
    id_code = None
    model = ModelCNN(layers, input, crop_size, l1_reg=0., l2_reg=0.001,
                     reg_bias=False, dropout=dropout, id_code=id_code)
    aes_in = []
    aes_out = []
    if id_code is not None:
        assert aes_out != []
    # Train
    # Data
    tr_batch_size = 10
    vl_batch_size = 8000

    with open(path_valid, 'r') as f:
        l_samples_vl = pkl.load(f)
        # convert to 3D
        nbr_xx = l_samples_vl["x"].shape[0]
        l_samples_vl["x"] = l_samples_vl["x"].reshape((nbr_xx, 1, h, w))
    list_minibatchs_vl = split_data_to_minibatchs_eval(
        l_samples_vl, vl_batch_size)
    max_epochs = int(1000)
    lr_vl = 1e-4
    lr = sharedX_value(lr_vl, name="lr")
    # cost weights
    separate = True
    l_in = [sharedX_value(0., name="l_in"), sharedX_value(0.0, name="l_in2")]
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
    in3D = True
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
        max_colm_norm=False, max_norm=15.0, eye=False, in3D=in3D)

    # How to update the weight costs
    updater_wc = StaticAnnealedWeightRate(anneal_end=500, anneal_start=0)
    updater_wc_in = StaticAnnealedWeightRateSingle(anneal_end=500, down=True,
                                                   init_vl=0., end_vl=0.,
                                                   anneal_start=100)
    updater_wc_in2 = StaticAnnealedWeightRateSingle(anneal_end=500, down=True,
                                                    init_vl=0.0, end_vl=0.,
                                                    anneal_start=400)
    updater_wc_out = StaticAnnealedWeightRateSingle(anneal_end=700, down=True,
                                                    init_vl=0., end_vl=0.,
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
            nbr_xx = l_samples["x"].shape[0]
            l_samples["x"] = l_samples["x"].reshape((nbr_xx, 1, h, w))
        l_ch_tr_vl.append(l_samples)

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

        print "\n", l_sup.get_value(), l_out[0].get_value()
        for el in l_in:
            print el.get_value(),
        print ""
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
        str(fold_exp) + " " + "cnn"
    with open("./" + str(time_exp) + ".py", "w") as python_file:
        python_file.write("import os \n")
        cmd2 = 'os.system("' + cmd + '")'
        python_file.write(cmd2)
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
