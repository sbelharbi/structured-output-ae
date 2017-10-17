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


floating = 2
prec2 = "%."+str(floating)+"f"


def evaluate_mean_shape(path_mean_shap, l_infos, w, h, save_ims=False):
    with open(path_mean_shap, 'r') as f:
        mean_shape = pkl.load(f)
    output = np.empty((len(l_infos), 68*2), dtype=theano.config.floatX)
    output[:, :] = mean_shape
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
    for e in figs:
        print "\r Saving image: ", i,
        fig = e["fig"]
        img_name = e["img_name"]
        nrmse = e["nrmse"]
        out_name = str(i) + "_" + str(prec2 % nrmse) + "_" + img_name
        fig.savefig(fd_imgs + "/" + out_name + ".png", bbox_inches='tight',
                    pad_inches=0)
        i += 1
        sys.stdout.flush()

    return cdf, cdf0_1, auc

if __name__ == "__main__":
    faceset = "lfpw"
    fd_data = "../../inout/data/face/color/"
    path_test = fd_data + faceset + "_ts.pkl"
    path_mean_shap = "../../inout/data/face/" + faceset +\
        "_data/mean_shape.pkl"
    w, h = 50, 50
    bx, dx = 0.5, 0.001
    border_x = bx
#    with open(path_test, 'r') as f:
#        l_samples = pkl.load(f)
    input = T.fmatrix("x_input")
    output = T.fmatrix("y_output")

    ds = FaceDataset()
#    x, y, l_infos = ds.sample_from_list_to_test(l_samples, w, h)
    ts_batch_size = 1000
    with open("../../inout/data/face/"+faceset+"_data/test.pkl", 'r') as fx:
        dumped = pkl.load(fx)
    with open("../../inout/data/face/"+faceset+"_data/valid.pkl", 'r') as fx:
        dumped_vl = pkl.load(fx)
    x = dumped["x"]
    y = dumped["y"]
    nfids = dumped["nfids"]
    bboxesT = dumped["bboxesT"]
    bboxesT_original = dumped["bboxesT_original"]
    base_name = dumped["base_name"]
    list_minibatchs_vl = split_data_to_minibatchs_eval(
        {"x": x, "y":  y}, ts_batch_size)
    fold_exp = "../../exps/" + sys.argv[1]
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

#    with open("best_model_keras.pkl", 'r') as f:
#        keras_p = pkl.load(f)
#    for param, param_vl in zip(model.params, keras_p):
#        param.set_value(param_vl)
#    cdf_ms, cdf0_1_ms, auc_ms = evaluate_mean_shape(
#        path_mean_shap, l_infos, w, h, save_ims=False)
    # Evaluation
    eval_fn = get_eval_fn(model)
    error, output = evaluate_model(list_minibatchs_vl, eval_fn)
    # Mean shape
    with open(path_mean_shap, 'r') as f:
        mean_shape = pkl.load(f)
    test_mean_shape_y = y * 0.
    nbr_test = y.shape[0]
    for i in xrange(nbr_test):
        test_mean_shape_y[i] = mean_shape

    #  Reproject the mean shape.
    x = np.arange(0, border_x, dx)
    test_mean_shape_y_reproj = ro.reproject_shape(
        bboxesT, test_mean_shape_y, nfids)
    test_mean_shape_y = ro.back_to_original_size_shapes(
        set_y=test_mean_shape_y_reproj, bbox=bboxesT,
        bbox_original=bboxesT_original)
    mu1_mean_shape, muAll_mean_shape, fail_mean_shape, loss_mean_shape =\
        ro.eval_regression(y, test_mean_shape_y, base_name)
    cdf_loss_mean_shape, auc_loss_mean_shape = ro.calculate_auc_of_loss(
        loss=loss_mean_shape, x=x, border_x=border_x, dx=dx)
    print "auc_loss_mean_shape:", auc_loss_mean_shape
    # ... Mean shape ends.
    model_output_reproj = ro.reproject_shape(bboxesT, output, nfids)
    # back to the original size
    output = ro.back_to_original_size_shapes(
        set_y=model_output_reproj, bbox=bboxesT,
        bbox_original=bboxesT_original)
    mu1, muAll, fail, loss = ro.eval_regression(y, output, base_name)
    print '...Mean(loss)', np.mean(loss)

    cdf_loss, auc_loss = ro.calculate_auc_of_loss(loss=loss, x=x,
                                                  border_x=border_x, dx=dx)
    print "auc_loss model: ", auc_loss
    fig = ds.plot_cdf_model_and_meansh(
        [cdf_loss, cdf_loss_mean_shape], tag, [mu1, mu1_mean_shape],
        [auc_loss, auc_loss_mean_shape], bx, dx)
    fig.savefig(fold_exp+"/cdf_curve.png", bbox_inches='tight')
    sys.exit()
    # Calculate the face errors: CDF0.1, AUC
    l_nrmse, figs = ds.calculate_face_errors_octave(output, l_infos, w, h)
    cdf, cdf0_1, auc = ds.calculate_auc_cdf(l_nrmse, bx, dx)
    print "\n Test error [model]: cdf0.1:", cdf0_1, " AUC:", auc
    # save results
    with open(fold_exp + "/errors.pkl", "w") as f:
        todump = {"output": output, "l_nrmse": l_nrmse, "cdf": cdf,
                  "cdf0_1": cdf0_1, "auc": auc}
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
    figs = sorted(figs, key=lambda kk: kk["nrmse"], reverse=True)
    i = 0
    for e in figs:
        print "\r Saving image: ", i,
        fig = e["fig"]
        img_name = e["img_name"]
        nrmse = e["nrmse"]
        out_name = str(i) + "_" + str(prec2 % nrmse) + "_" + img_name
        fig.savefig(fd_imgs + "/" + out_name + ".png", bbox_inches='tight',
                    pad_inches=0)
        i += 1
        sys.stdout.flush()
