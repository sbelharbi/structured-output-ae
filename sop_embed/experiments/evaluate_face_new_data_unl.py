from sop_embed.dataset import Dataset
from sop_embed.tools import split_data_to_minibatchs_eval
from sop_embed.tools import get_eval_fn
from sop_embed.tools import evaluate_model
from sop_embed.tools import plot_all_x_y_yhat
from sop_embed.tools import ModelMLP
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
    # No need to images.
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

    if faceset == "helen":
        unl = "lfpw"
    elif faceset == "lfpw":
        unl = "helen"
    fd_data = "../../inout/data/face/color/"
    path_test = fd_data + faceset + "_ts.pkl"
    path_mean_shap = "../../inout/data/face/" + faceset +\
        "_u_" + unl + "/mean_shape.pkl"
    w, h = 50, 50
    bx, dx = 0.5, 0.001
    border_x = bx
    with open(path_test, 'r') as f:
        l_samples = pkl.load(f)
    input = T.fmatrix("x_input")
    output = T.fmatrix("y_output")

    ds = FaceDataset()
    x, y, l_infos = ds.sample_from_list_to_test(l_samples, w, h)
    ts_batch_size = 1000

    list_minibatchs_vl = split_data_to_minibatchs_eval(
        {"x": x, "y":  y}, ts_batch_size)
    fold_exp = sys.argv[2]
    with open(fold_exp+"/model.pkl", 'r') as f:
        stuff = pkl.load(f)
        layers_infos, params_vl = stuff["layers_infos"], stuff["params_vl"]
        print layers_infos
        tag = stuff["tag"]
        dropout = stuff["dropout"]
        rng = np.random.RandomState(23455)
        input = T.fmatrix("x_input")
        for l in layers_infos:
            l["W"], l["b"], l["rng"] = None, None, rng
        model = ModelMLP(layers_infos, input, dropout=dropout)
    model.set_params_vals(fold_exp+"/model.pkl")

    eval_fn = get_eval_fn(model)
    # Perf mean shape.
    # TRAIN
    if faceset == "lfpw":
        tr_path = "../../inout/data/face/" + faceset + "_data/ch_tr_676_0_0_0.pkl"
    elif faceset == "helen":
        tr_path = "../../inout/data/face/" + faceset + "_data/ch_tr_1800_0_0_0.pkl"
    print "TRAIN EVAL:"
    with open(tr_path, 'r') as f:
        tr_data = pkl.load(f)
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
    l_nrmse, figs = ds.calculate_face_errors_octave(output, l_infos, w, h)
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
        fper.write("Datasets:" + faceset + " ul: " + str(unl) + "\n")
        fper.write("MSE tr:" + str(mse_train) + "\n")
        fper.write("MSE vl:" + str(mse_vl) + "\n")
        fper.write("Test AUC:" + str(test_auc) + "\n")
        fper.write("Test cdf0.1:" + str(test_cdf0_1) + "\n")
    # figs = sorted(figs, key=lambda kk: kk["nrmse"], reverse=True)
    i = 0
    # Not necessary!!!!!!
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
