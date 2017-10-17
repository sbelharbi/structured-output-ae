from sop_embed.dataset import Dataset
from sop_embed.tools import split_data_to_minibatchs_eval
from sop_embed.tools import get_eval_fn
from sop_embed.tools import evaluate_model
from sop_embed.tools import plot_all_x_y_yhat
from sop_embed.tools import ModelMLP
from sop_embed.tools import ModelCNN
from sop_embed.facedataset import FaceDataset

import theano.tensor as T
import numpy as np
import cPickle as pkl
import sys
import os
import shutil
import theano

from Routines import Routines 
ro = Routines()


floating = 10
prec2 = "%."+str(floating)+"f"


def unit_test(fd_imgs, w, h, path_mean_shap, eval_fn, ds):
    with open(path_mean_shap, 'r') as f:
        msh = pkl.load(f)
    w_im = np.ones((w, h, 3), dtype=theano.config.floatX)
    b_im = np.zeros((w, h, 3), dtype=theano.config.floatX)
    # mean shape
    x_pred = (msh[:68] + 0.5) * h
    y_pred = (msh[68:] + 0.5) * w
    bb_gt = [0, 0, w, h]
    fig_w = ds.plot_over_img(w_im, x_pred, y_pred, x_pred, y_pred, bb_gt)
    fig_b = ds.plot_over_img(b_im, x_pred, y_pred, x_pred, y_pred, bb_gt)
    inp = np.empty((2, w*h), dtype=theano.config.floatX)
    outp = np.empty((2, 68*2), dtype=theano.config.floatX)
    inp[0, :] = w_im[:, :, 0].flatten()
    inp[1, :] = b_im[:, :, 0].flatten()
    [error_mn, output_mn] = eval_fn(inp, outp)
    x_pred = (output_mn[0, :68] + 0.5) * h
    y_pred = (output_mn[0, 68:] + 0.5) * w
    fig_w_model = ds.plot_over_img(w_im, x_pred, y_pred, x_pred, y_pred,
                                   bb_gt)
    x_pred = (output_mn[1, :68] + 0.5) * h
    y_pred = (output_mn[1, 68:] + 0.5) * w
    fig_b_model = ds.plot_over_img(b_im, x_pred, y_pred, x_pred, y_pred,
                                   bb_gt)

    if not os.path.exists(fd_imgs):
        os.makedirs(fd_imgs)
    fig_w.savefig(fd_imgs + "msh_w.png", bbox_inches='tight', pad_inches=0)
    fig_b.savefig(fd_imgs + "msh_b.png", bbox_inches='tight', pad_inches=0)
    fig_w_model.savefig(fd_imgs + "model_w.png", bbox_inches='tight',
                        pad_inches=0)
    fig_b_model.savefig(fd_imgs + "model_b.png", bbox_inches='tight',
                        pad_inches=0)


def evaluate_mean_shape(path_mean_shap, l_infos, w, h, y, save_ims=False):
    with open(path_mean_shap, 'r') as f:
        mean_shape = pkl.load(f)
    if y is None:
        output = np.empty((len(l_infos), 68*2), dtype=theano.config.floatX)
    else:
        output = np.empty((y.shape[0], 68*2), dtype=theano.config.floatX)
    output[:, :] = mean_shape
    # MSE:
    if y is not None:
        mse = np.mean(np.mean(np.power(output - y, 2), axis=1))
        print "Error (MSE) mean shape", mse
        return 0, 0, 0
    # Calculate the face errors: CDF0.1, AUC
    l_nrmse, figs = ds.calculate_face_errors_octave(output, l_infos, w, h)
    cdf, cdf0_1, auc = ds.calculate_auc_cdf(l_nrmse, bx, dx)
    print "\n Test error [mean shape]: cdf0.1:", cdf0_1, " AUC:", auc
    fig = ds.plot_cdf(cdf, tag, cdf0_1, auc, bx, dx)
    fig.savefig(fold_exp+"/cdf_curve_mean_shape.png", bbox_inches='tight')
    with open(fold_exp + "/errors_mean_shape.pkl", "w") as f:
        todump = {"output": output, "l_nrmse": l_nrmse, "cdf": cdf,
                  "cdf0_1": cdf0_1, "auc": auc}
        pkl.dump(todump, f, protocol=pkl.HIGHEST_PROTOCOL)
    if not save_ims:
        return cdf, cdf0_1, auc
    # Plot the images
    fd_imgs = fold_exp + '/images_test_mean_shape/'
    if not os.path.exists(fd_imgs):
        os.makedirs(fd_imgs)
    else:
        shutil.rmtree(fd_imgs)
        os.makedirs(fd_imgs)
    figs = sorted(figs, key=lambda kk: kk["nrmse"], reverse=True)
    i = 0
    # Do not save figs!!!
#    for e in figs:
#        print "\r Saving image: ", i,
#        fig = e["fig"]
#        img_name = e["img_name"]
#        nrmse = e["nrmse"]
#        out_name = str(i) + "_" + str(prec2 % nrmse) + "_" + img_name
#        fig.savefig(fd_imgs + "/" + out_name + ".png", bbox_inches='tight',
#                    pad_inches=0)
#        i += 1
#        sys.stdout.flush()

    return cdf, cdf0_1, auc

if __name__ == "__main__":
    faceset = str(sys.argv[1])
    print "Start evaluting Dataset: ", faceset, " ..."
    fd_data = "../../inout/data/face/color/"
    path_test = fd_data + faceset + "_ts.pkl"
    path_mean_shap = "../../inout/data/face/" + faceset +\
        "_data/mean_shape.pkl"
    w, h = 50, 50
    bx, dx = 0.5, 0.001
    border_x = bx
    model_type = sys.argv[3]  # mlp, cnn, fcn.
    with open(path_test, 'r') as f:
        l_samples = pkl.load(f)

    output = T.fmatrix("y_output")

    ds = FaceDataset()
    x, y, l_infos = ds.sample_from_list_to_test(l_samples, w, h)
    ts_batch_size = 1000
    if model_type == "cnn":
        nbr_xx = x.shape[0]
        x = x.reshape((nbr_xx, 1, h, w))
    list_minibatchs_vl = split_data_to_minibatchs_eval(
        {"x": x, "y":  y}, ts_batch_size)
    fold_exp = sys.argv[2]
    with open(fold_exp+"/model.pkl", 'r') as f:
        stuff = pkl.load(f)
        layers_infos, params_vl = stuff["layers_infos"], stuff["params_vl"]
        print layers_infos
        tag = stuff["tag"]
        dropout = stuff["dropout"]
        if model_type == "mlp":
            rng = np.random.RandomState(23455)
            input = T.fmatrix("x_input")
            for l in layers_infos:
                l["W"], l["b"], l["rng"] = None, None, rng
            model = ModelMLP(layers_infos, input, dropout=dropout)
        elif model_type == "cnn":
            input = T.tensor4("x_input")
            config_arch = stuff["config_arch"]
            crop_size = stuff["crop_size"]
            model = ModelCNN(config_arch, input, crop_size)
    model.set_params_vals(fold_exp+"/model.pkl")

    in3D = False
    if model_type == "cnn":
        in3D = True
    eval_fn = get_eval_fn(model, in3D=in3D)
    # Unit test
    # unit_test(fold_exp+"/unit_imgs/", w, h, path_mean_shap, eval_fn, ds)
    # Perf mean shape.
    # TRAIN
    if faceset == "lfpw":
        tr_path = "../../inout/data/face/" + faceset + "_data/ch_tr_676_0_0_0.pkl"
    elif faceset == "helen":
        tr_path = "../../inout/data/face/" + faceset + "_data/ch_tr_1800_0_0_0.pkl"
    print "TRAIN EVAL:"
    with open(tr_path, 'r') as f:
        tr_data = pkl.load(f)
    if model_type == "cnn":
        nbr_xx_tr = tr_data["x"].shape[0]
        tr_data["x"] = tr_data["x"].reshape((nbr_xx_tr, 1, h, w))
    list_minibatchs_train = split_data_to_minibatchs_eval(
        {"x": tr_data["x"], "y":  tr_data['y']}, ts_batch_size)
    cdf_ms, cdf0_1_ms, auc_ms = evaluate_mean_shape(path_mean_shap, l_infos,
                                                    w, h, y=tr_data['y'],
                                                    save_ims=False)
    # Evaluation
    error, output = evaluate_model(list_minibatchs_train, eval_fn)
    print "Error (MSE) model train data", np.mean(error)
    mse_train = np.mean(error)
    # VALID
    vl_path = "../../inout/data/face/" + faceset + "_data/valid.pkl"
    print "VALID EVAL:"
    with open(vl_path, 'r') as f:
        vl_data = pkl.load(f)
    if model_type == "cnn":
        nbr_xx_vl = vl_data["x"].shape[0]
        vl_data["x"] = vl_data["x"].reshape((nbr_xx_vl, 1, h, w))
    list_minibatchs_valid = split_data_to_minibatchs_eval(
        {"x": vl_data["x"], "y":  vl_data['y']}, ts_batch_size)
    cdf_ms, cdf0_1_ms, auc_ms = evaluate_mean_shape(path_mean_shap, l_infos,
                                                    w, h, y=vl_data['y'],
                                                    save_ims=False)
    # Evaluation
    error, output = evaluate_model(list_minibatchs_valid, eval_fn)
    print "Error (MSE) model valid data", error
    mse_vl = error
    # TEST
    cdf_ms, cdf0_1_ms, auc_ms = evaluate_mean_shape(path_mean_shap, l_infos,
                                                    w, h, y=None,
                                                    save_ims=False)
    # Evaluation
    error, output = evaluate_model(list_minibatchs_vl, eval_fn)

    # Calculate the face errors: CDF0.1, AUC
    l_nrmse, figs = ds.calculate_face_errors_octave(output, l_infos, w, h,
                                                    seg=True)
    cdf, cdf0_1, auc = ds.calculate_auc_cdf(l_nrmse, bx, dx)
    print "\n Test error [model]: cdf0.1:", cdf0_1, " AUC:", auc
    test_auc, test_cdf0_1 = auc, cdf0_1
    # save results
    with open(fold_exp + "/errors.pkl", "w") as f:
        todump = {"output": output, "l_nrmse": l_nrmse, "cdf": cdf,
                  "cdf0_1": cdf0_1, "auc": auc,
                  "cdf_ms": cdf_ms, "cdf0_1_ms": cdf0_1_ms,
                  "auc_ms": auc_ms}
        pkl.dump(todump, f, protocol=pkl.HIGHEST_PROTOCOL)

    # Plot the images
    fd_imgs = fold_exp + '/images_test/'
    if not os.path.exists(fd_imgs):
        os.makedirs(fd_imgs)
    else:
        shutil.rmtree(fd_imgs)
        os.makedirs(fd_imgs)

    fig = ds.plot_cdf_model_and_meansh(
        [cdf, cdf_ms], tag, [cdf0_1, cdf0_1_ms], [auc, auc_ms], bx, dx)
    fig.savefig(fold_exp+"/cdf_curve.png", bbox_inches='tight')
    with open(fold_exp+"/perf.txt", "w") as fper:
        fper.write("Datasets:" + faceset + "\n")
        fper.write("MSE tr:" + str(mse_train) + "\n")
        fper.write("MSE vl:" + str(mse_vl) + "\n")
        fper.write("Test AUC:" + str(test_auc) + "\n")
        fper.write("Test cdf0.1:" + str(test_cdf0_1) + "\n")
    # figs = sorted(figs, key=lambda kk: kk["nrmse"], reverse=True)
    i = 0
    # No figs.
#    for e in figs:
#        print "\r Saving image: ", i,
#        fig = e["fig"]
#        img_name = e["img_name"]
#        nrmse = e["nrmse"]
#        out_name = str(i) + "_" + str(prec2 % nrmse) + "_" + img_name
#        fig.savefig(fd_imgs + "/" + out_name + ".png", bbox_inches='tight',
#                    pad_inches=0)
#        i += 1
#        sys.stdout.flush()
