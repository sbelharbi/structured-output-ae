from sop_embed.da import ConvolutionalAutoencoder
from sop_embed.ae import CostType
from sop_embed.tools import split_data_to_minibatchs_eval
from sop_embed.tools import split_data_to_minibatchs_embed
from sop_embed.tools import plot_all_x_xhat
from sop_embed.facedataset import FaceDataset
from sop_embed.tools import plot_fig
from sop_embed.tools import sharedX_value
from sop_embed.tools import evaluate_model_3D_unsup
from sop_embed.learning_rule import Momentum
from sop_embed.learning_rule import AdaDelta

import theano.tensor as T
import datetime as DT
import os
import sys
import numpy as np
import cPickle as pkl
import theano
import inspect
import shutil
import yaml


type_mod = ""
faceset = "lfpw"
fd_data = "../../inout/data/face/" + faceset + "_data/"
path_valid = fd_data + type_mod + "valid.pkl"
w, h = 50, 50
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
lr_vl = 1e-4
optimizer = "momentum"
print "Optimizer:", optimizer, " lr:", lr_vl
fold_exp = "../../exps/" + faceset + "_deep_convae_" + optimizer + "_lr_" +\
    str(lr_vl) + "_" + time_exp
if not os.path.exists(fold_exp):
    os.makedirs(fold_exp)

nbr_layers = 4
init_w_deep_conv_ae_path = "../../inout/init_weights/deep_conv_ae_" +\
    str(nbr_layers) + '_' + faceset + '_layers/'
if not os.path.exists(init_w_deep_conv_ae_path):
    os.makedirs(init_w_deep_conv_ae_path)

rnd = np.random.RandomState(1231)
nhid_l0 = 1025
nhid_l1 = 512
nhid_l2 = 64

# Deep conv.ae

# configure encoder
encode_cae_l0 = {"type": "conv",
                 "rng": rnd,
                 "filter_shape": (16, 3, 3),
                 "activation": "relu",
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
                 "activation": "relu",
                 "padding": (1, 1),
                 "W": None,
                 "b": None,
                 "b_v": 0.,
                 "stride": (1, 1)}
encode_cae_l3 = {"type": "downsample",
                 "poolsize": (2, 2),
                 "ignore_border": True,
                 "mode": "max"}
# Dim: 8 * 12x12.

encoder_config = [encode_cae_l0, encode_cae_l1, encode_cae_l2, encode_cae_l3]

# configure decoder
decode_cae_l0 = {"type": "conv",
                 "rng": rnd,
                 "filter_shape": (8, 3, 3),
                 "activation": "relu",
                 "padding": (1, 1),
                 "W": None,
                 "b": None,
                 "b_v": 0.,
                 "stride": (1, 1)}
decode_cae_l1 = {"type": "upsample",
                 "ratio": 2,
                 "use_1D_kernel": False}
# Dim: 8 * 24x24
decode_cae_l2 = {"type": "conv",
                 "rng": rnd,
                 "filter_shape": (16, 3, 3),
                 "activation": "relu",
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
decoder_config = [decode_cae_l0, decode_cae_l1, decode_cae_l2, decode_cae_l3,
                  decode_cae_l4]
crop_size = [1, h, w]
batch_size = 10  # must be the same.
tr_batch_size = 10
vl_batch_size = 8000
deep_conv_ae = ConvolutionalAutoencoder(input=input,
                                        encoder_config=encoder_config,
                                        decoder_config=decoder_config,
                                        crop_size=crop_size,
                                        cost_type=CostType.MeanSquared)
id_deep_conv_ae = ""
deep_conv_contact = encoder_config + decoder_config
for e in deep_conv_contact:
    for k in e.keys():
        if k not in ["rng", "W", "b"]:
            id_deep_conv_ae += str(e[k]) + '|'
    # id_deep_conv_ae += "+"

id_deep_conv_ae = id_deep_conv_ae.replace(" ", "")
id_deep_conv_ae = id_deep_conv_ae.replace("(", "")
id_deep_conv_ae = id_deep_conv_ae.replace(")", "")
print len(id_deep_conv_ae), id_deep_conv_ae
path_ini_params_deep_conv_ae = init_w_deep_conv_ae_path +\
    "deep_conv_ae_init_" + id_deep_conv_ae + ".pkl"
if not os.path.isfile(path_ini_params_deep_conv_ae):
    deep_conv_ae.save_params(path_ini_params_deep_conv_ae)
else:
    deep_conv_ae.set_params_vals(path_ini_params_deep_conv_ae)

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
optimizer = "momentum"
if optimizer == "adadelta":
    updater = AdaDelta(decay=0.95)
elif optimizer == "momentum":
    updater = Momentum(0.9, nesterov_momentum=True,
                       imagenet=False, imagenetDecay=5e-4,
                       max_colm_norm=False)
else:
    raise ValueError("Optimizer not recognized.")

# Prepare costs.
cost_train, _, _ = deep_conv_ae.get_train_cost()
# updates
grads = T.grad(cost_train, deep_conv_ae.params)
lr_sc_all = list([sharedX_value(1.) for i in xrange(len(deep_conv_ae.params))])
updates = updater.get_updates(lr, deep_conv_ae.params, grads, lr_sc_all)
x_train = T.tensor4('x_train')
sgd_update = theano.function(
    inputs=[x_train],
    outputs=[T.mean(cost_train)],
    updates=updates,
    givens={deep_conv_ae.x: x_train},
    on_unused_input="ignore")
# eval function
x_val = T.tensor4("x_vl")
recons_error = deep_conv_ae.get_reconstruction_error()
eval_fn = theano.function(
    inputs=[x_val],
    outputs=[recons_error, deep_conv_ae.output, deep_conv_ae.encoder.output],
    givens={deep_conv_ae.x: x_val})

# validate before start training
error_mn, _, _ = evaluate_model_3D_unsup(list_minibatchs_vl, eval_fn)
vl_error_begin = np.mean(error_mn)
print "Val before train:", vl_error_begin

shutil.copy(inspect.stack()[0][1], fold_exp)
l_ch_tr_vl = []
for ch in l_ch_tr:
    with open(ch, 'r') as f:
        l_samples = pkl.load(f)
        nbr_xx = l_samples["x"].shape[0]
        l_samples["x"] = l_samples["x"].reshape((nbr_xx, 1, h, w))
    l_ch_tr_vl.append(l_samples)

# Train
stop = False
i = 0
in_cost, error_vl = [], []
freq_vl = 10
count, best_count = 0, 0
catched_one = False
while i < max_epochs:
    for ch in l_ch_tr_vl:
        if isinstance(ch, str):
            print "\n Chunk:", ch
            with open(ch, 'r') as f:
                l_samples = pkl.load(f)
        else:
            l_samples = ch
        sharpe = False
        list_minibatchs_tr = split_data_to_minibatchs_embed(
            l_samples, tr_batch_size, share=False, sharpe=sharpe)
#        for ts in xrange(100):
#            shuffle(list_minibatchs_tr)
        for minibatch in list_minibatchs_tr:
            count += 1
            t0 = DT.datetime.now()
            in_cost_mb, out_cost_mb, all_cost_mb = 0., 0., 0.
            # share data ONLY when needed (save the GPU memory).
            d_in = minibatch['in']
            d_out = minibatch['out']
            d_sup = minibatch['sup']
            # Update unsupervised task in
            if (d_in['x'] is not None):
                xx = theano.shared(
                    d_in['x'], borrow=True).get_value(borrow=True)
                in_cost_mb = sgd_update(xx)[0]
                in_cost.append(in_cost_mb)
                print in_cost_mb
                del xx
            # Val.
            if (count % freq_vl == 0):
                error_mn, _, _ = evaluate_model_3D_unsup(list_minibatchs_vl,
                                                         eval_fn)
                error_vl.append(np.mean(error_mn))
                if len(error_vl) > 1:
                    min_val = np.min(error_vl[:-1])
                else:
                    min_val = vl_error_begin
                print "Min val:", min_val, " current val:", error_vl[-1]
                if error_vl[-1] < min_val:
                    deep_conv_ae.catch_params()
                    catched_one = True
                    best_count = count
    i += 1

# End training
# Plot training figures.
plot_fig(in_cost, "Train DeepConvAE cost (in).", "Minibatch", "Train cost",
         fold_exp+"/train_cost_in_deep_conv_ae.png", best_count)
plot_fig(error_vl, "Valid error DeepConvAE (in, MSE).", "Minibatch",
         "Valid error (MSE)", fold_exp+"/valid_error_in_deep_conv_ae.png",
         best_count)
deep_conv_ae.save_params(fold_exp+"/model.pkl", catched=catched_one)

# For evaluation: swicth to best params.
deep_conv_ae.switch_to_catched_params()
# Evaluation error.
# Train
final_tr_error = 0
example_tr = None
for ch in l_ch_tr_vl:
    list_minibatchs_tr = split_data_to_minibatchs_eval(ch, vl_batch_size)
    error, output, code = evaluate_model_3D_unsup(list_minibatchs_tr, eval_fn)
    final_tr_error += np.mean(error)
    example_tr = (ch, output, code)

folder_imgs = fold_exp + "/train_img_recons"
if not os.path.exists(folder_imgs):
    os.makedirs(folder_imgs)
plot_all_x_xhat(example_tr[0]["x"], example_tr[1], 100, folder_imgs)
# Valid
error_mn, output, code = evaluate_model_3D_unsup(list_minibatchs_vl, eval_fn)
final_vl_error = np.mean(error_mn)
example_vl = (l_samples_vl, output, code)

folder_imgs = fold_exp + "/valid_img_recons"
if not os.path.exists(folder_imgs):
    os.makedirs(folder_imgs)
plot_all_x_xhat(example_vl[0]["x"], example_vl[1], 100, folder_imgs)

# Test
fd_data = "../../inout/data/face/color/"
path_test = fd_data + faceset + "_ts.pkl"
with open(path_test, 'r') as f:
        loaded = pkl.load(f)
        ds = FaceDataset()
        x, y, l_infos = ds.sample_from_list_to_test(loaded, w, h)
        l_samples_tst = {"x": x, "y": y}
        nbr_xx = x.shape[0]
        l_samples_tst["x"] = l_samples_tst["x"].reshape((nbr_xx, 1, h, w))
list_minibatchs_tst = split_data_to_minibatchs_eval(l_samples_tst,
                                                    vl_batch_size)
error_mn, output, code = evaluate_model_3D_unsup(list_minibatchs_tst, eval_fn)
final_tst_error = np.mean(error_mn)
example_tst = (l_samples_tst, output, code)

folder_imgs = fold_exp + "/test_img_recons"
if not os.path.exists(folder_imgs):
    os.makedirs(folder_imgs)
plot_all_x_xhat(example_tst[0]["x"], example_tst[1], 100, folder_imgs)

print ""
print "Errors: tr:", final_tr_error, " vl:", final_vl_error, " test:",\
    final_tst_error
print "VL begins with:", vl_error_begin

with open(fold_exp+"/perf_tr_vl_tst.yaml", "w") as fd:
    to_dump = {"final_tr_error": float(final_tr_error),
               "final_vl_error": float(final_vl_error),
               "final_tst_error": float(final_tst_error),
               "vl_error_begin": float(vl_error_begin),
               "optimizer": optimizer,
               "lr": lr_vl}
    yaml.dump(to_dump, fd, default_flow_style=False)

with open(fold_exp+"/predictions_in_code_output.pkl", "w") as fd:
    to_dump = {"tr": {"error": final_tr_error, "example_tr": example_tr},
               "vl": {"error": final_vl_error, "example_vl": example_vl},
               "ts": {"error": final_tst_error, "example_tst": example_tst}}
    pkl.dump(to_dump, fd, protocol=pkl.HIGHEST_PROTOCOL)
