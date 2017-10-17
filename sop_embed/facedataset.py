from __future__ import division

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import copy
import matplotlib.cm as cm
import theano
import cv2
import glob
import scipy.misc as sm
import sys
from scipy.integrate import simps, trapz
from oct2py import octave

# from sop_embed.convnetskeras.convnets import convnet
from keras.optimizers import SGD

floating = 6
prec2 = "%."+str(floating)+"f"

corruption_level = 0.01  # For data augmentation


class FaceDataset(object):
    def __init__(self):
        pass

    def create_ft_extractor(self, type_mod, weights_path):
        """Extract the features from x using a convnet model."""
#        model = convnet(type_mod, weights_path=weights_path, heatmap=False,
#                        W_regularizer=None,
#                        activity_regularizer=None,
#                        dense=False)
#        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#        model.compile(optimizer=sgd, loss="mse")
#        print "Summary:", model.summary()
        model = None

        return model

    def debug_plot_over_img(self, img, x, y, bb_d, bb_gt):
        """Plot the landmarks over the image with the bbox."""
        plt.close("all")
        fig = plt.figure()  # , figsize=(15, 10.8), dpi=200
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.imshow(img, aspect="auto", cmap='Greys_r')
        ax.scatter(x, y, s=10, color='r')
        rect1 = patches.Rectangle(
            (bb_d[0], bb_d[1]), bb_d[2]-bb_d[0], bb_d[3]-bb_d[1],
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        rect2 = patches.Rectangle(
            (bb_gt[0], bb_gt[1]), bb_gt[2]-bb_gt[0], bb_gt[3]-bb_gt[1],
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect2)
        fig.add_axes(ax)

        return fig

    def plot_over_img(self, img, x, y, x_pr, y_pr, bb_gt):
        """Plot the landmarks over the image with the bbox."""
        plt.close("all")
        fig = plt.figure(frameon=False)  # , figsize=(15, 10.8), dpi=200
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect="auto")
        ax.scatter(x, y, s=10, color='r')
        ax.scatter(x_pr, y_pr, s=10, color='g')
        rect = patches.Rectangle(
            (bb_gt[0], bb_gt[1]), bb_gt[2]-bb_gt[0], bb_gt[3]-bb_gt[1],
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        fig.add_axes(ax)

        return fig

    def plot_over_img_seg(self, img, x, y, x_pr, y_pr, bb_gt, tag_oc=None):
        """Plot the landmarks over the image with the bbox."""
        plt.close("all")
        fig = plt.figure(frameon=False)  # , figsize=(15, 10.8), dpi=200
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        bb_gt = [int(xx) for xx in bb_gt]
        hight, width = bb_gt[3]-bb_gt[1], bb_gt[2]-bb_gt[0]
        if tag_oc is None:
            img_oc = copy.deepcopy(img)
        elif tag_oc is "left":
            img_oc = copy.deepcopy(img)
            p = int((20/50.) * width)  # we took only 20 pixels from 50.
            img_oc[bb_gt[1]:bb_gt[3],
                   bb_gt[0]:bb_gt[0]+p, :] = np.uint8(255/2.)
        elif tag_oc is "right":
            img_oc = copy.deepcopy(img)
            p = int((20/50.) * width)  # we took only 20 pixels from 50.
            img_oc[bb_gt[1]:bb_gt[3],
                   bb_gt[2]-p:bb_gt[2], :] = np.uint8(255/2.)
        elif tag_oc is "up":
            img_oc = copy.deepcopy(img)
            p = int((20/50.) * hight)  # we took only 20 pixels from 50.
            img_oc[bb_gt[1]:bb_gt[1]+p,
                   bb_gt[0]:bb_gt[2], :] = np.uint8(255/2.)
        elif tag_oc is "down":
            img_oc = copy.deepcopy(img)
            p = int((20/50.) * hight)  # we took only 20 pixels from 50.
            img_oc[bb_gt[3]-p:bb_gt[3],
                   bb_gt[0]:bb_gt[2], :] = np.uint8(255/2.)
        elif tag_oc is "middle":
            img_oc = copy.deepcopy(img)
            p1 = int((15/50.) * hight)  # we took only from 15 pixels from 50.
            p2 = int((35/50.) * hight)  # we took only to 35 pixels from 50.
            img_oc[bb_gt[1]+p1:bb_gt[1]+p2,
                   bb_gt[0]:bb_gt[2], :] = np.uint8(255/2.)
        ax.imshow(cv2.cvtColor(img_oc, cv2.COLOR_BGR2RGB), aspect="auto")
        for i in xrange(68):
            ax.plot([x[i], x_pr[i]], [y[i], y_pr[i]], '-r')

        fig.add_axes(ax)

        return fig

    def debug_draw_set(self, path_f, fd_out):
        if not os.path.exists(fd_out):
            os.makedirs(fd_out)
        with open(path_f, 'r') as f:
            stuff = pkl.load(f)
        for el in stuff:
            img_name = el["img_name"]
            print img_name
            img_gray = el["img_gray"]
            bb_d = el["bb_detector"]
            bb_gt = el["bb_ground_truth"]
            x = el["annox"]
            y = el["annoy"]
            fig = self.debug_plot_over_img(img_gray, x, y, bb_d, bb_gt)
            fig.savefig(
                fd_out+"/"+img_name, bbox_inches='tight', pad_inches=0,
                frameon=False, cmap=cm.Greys_r)
            del fig

    def rescale_bb(self, img_shape, bb):
        return bb
        # How much to inrease the booox
        inc_w, inc_h = 0., 0.
        x1, y1, x2, y2 = bb
        w_bb = y2 - y1
        h_bb = x2 - x1
        c_x, c_y = x1 + h_bb/2, y1 + w_bb/2
        w_bb = w_bb + inc_w * w_bb
        h_bb = h_bb + inc_h * h_bb
        h_bb = h_bb + h_bb/2
        x1, y1 = c_x - h_bb/2, c_y - w_bb/2
        x2, y2 = c_x + h_bb/2, c_y + w_bb/2
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 >= img_shape[1]:
            x2 = img_shape[1] - 1
        if y2 >= img_shape[0]:
            y2 = img_shape[0] - 1
        return [x1, y1, x2, y2]

    def normalizr_img_color_cnn(self, img, color_mode):
        if color_mode == "bgr":
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        img = img.transpose((2, 0, 1))

        return img

    def sample_no_ag(self, img, bb, annox, annoy, w, h,
                     color_mode=None, ft_extract=None, get_original=False):
        [x1, y1, x2, y2] = [int(e) for e in bb]
        w_bb, h_bb = y2 - y1, x2 - x1
        if ft_extract is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.normalizr_img_color_cnn(img, color_mode)

        if ft_extract is None:
            cropped = img[y1:y2+1, x1:x2+1] / 255.
        else:
            cropped = img[:, y1:y2+1, x1:x2+1] / 255.
            cropped = cropped.transpose((1, 2, 0))

        im = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        if ft_extract is not None:
            im = im.transpose((2, 0, 1))
            input_cnn = np.empty((1, 3, w, h), dtype=theano.config.floatX)
            input_cnn[0] = im
            im = ft_extract.predict(input_cnn)

        #fig = self.debug_plot_over_img(cropped, annox, annoy, bb, bb)
        #fig.savefig("2.png",  bbox_inches='tight')
        #raw_input("wait")

        # Normlaize in new box
        if not get_original:
            annox = (annox - x1)
            annoy = (annoy - y1)

            rx = h / float(h_bb)
            ry = w / float(w_bb)
            annox *= rx
            annoy *= ry
            annox /= float(h)  # in [0, 1] (no scaling!!!!)
            annoy /= float(w)  # in [0, 1] (no scaling!!!!)
            annox -= 0.5  # normalize into [-1, 1]
            annoy -= 0.5  # normalize into [-1, 1]

        # Rormlaize in new box END.
        # Or rescale only into the original box
#        annox = (annox - x1) / float(h_bb)  # in [0, 1] (no scaling!!!!)
#        annoy = (annoy - y1) / float(w_bb)  # in [0, 1] (no scaling!!!!)
        #fig = self.debug_plot_over_img(cropped, annox, annoy, bb, bb)
        #fig.savefig("3.png",  bbox_inches='tight')
        #raw_input("wait 2")
        return im, annox, annoy

    def inter_ocular_dist(self, annox, annoy):
        """Calculate the Euclidean  distance between the outer corners of the
            eyes."""
        return np.sqrt((annox[37] - annox[46])**2 + (annoy[37] - annoy[46])**2)

    def sample_from_list_to_test(self, l_samples, w, h):
        x, y = self.sample_from_list(l_samples, w, h, d=0, get_original=True)
        l_infos = []
        i = 0
        for e in l_samples:
            l_infos.append({
                "im_rgb": e["img_rgb"],
                "im_gray": e["img_gray"],
                "im_name": e["img_name"],
                "annox": y[i, :68],
                "annoy": y[i, 68:],
                "bb_gt": e["bb_ground_truth"],
                "d_eyes": self.inter_ocular_dist(e["annox"], e["annoy"])})
            i += 1

        return x, y, l_infos

    def calculate_face_errors(self, pred, l_infos, w, h):
        l_nrmse = []
        out = []
        for i in range(pred.shape[0]):
            el = l_infos[i]
            img_rgb = el["im_rgb"]
            img_name = el["im_name"]
            annox, annoy = el["annox"], el["annoy"]
            d_eyes, bb_gt = el["d_eyes"], el["bb_gt"]
            x_pred, y_pred = pred[i, 0:68], pred[i, 68:]
            # Convert prediction
            bb_gt = self.rescale_bb(el["im_gray"].shape, bb_gt)
            [x1, y1, x2, y2] = [int(e) for e in bb_gt]
            w_bb, h_bb = y2 - y1, x2 - x1
            # Norlalize in the new box

            rx = h / float(h_bb)
            ry = w / float(w_bb)
            x_pred += 0.5
            y_pred += 0.5
            x_pred *= h
            y_pred *= w
            x_pred /= rx
            y_pred /= ry
            x_pred += x1
            y_pred += y1

            # Rescale only in the original box.
#            x_pred = x_pred * float(h_bb) + x1
#            y_pred = y_pred * float(w_bb) + y1
            tmpx = np.power(np.subtract(annox, x_pred), 2)
            tmpy = np.power(np.subtract(annoy, y_pred), 2)
            tmpxy = np.sum(np.power((tmpx + tmpy), 0.5))
            nrmse = tmpxy / (68. * d_eyes)
            l_nrmse.append(nrmse)
            fig = self.plot_over_img(img_rgb, annox, annoy, x_pred,
                                     y_pred, bb_gt)
            out.append({"fig": fig, "img_name": img_name, "nrmse": nrmse})

        return np.asarray(l_nrmse), out

    def calculate_face_errors_octave(self, pred, l_infos, w, h, seg=False,
                                     tags_oc=None):
        octave.addpath("../")
        l_nrmse = []
        out = []
        nfids = np.int(pred.shape[1]/2)
        n_ims = pred.shape[0]
        y_reshaped = np.zeros((nfids, 2, n_ims))
        model_output_reshaped = np.zeros((nfids, 2, n_ims))
        saved_pred = copy.deepcopy(pred)

        for i in range(pred.shape[0]):
            el = l_infos[i]
            img_rgb = el["im_rgb"]
            img_name = el["im_name"]
            annox, annoy = el["annox"], el["annoy"]
            d_eyes, bb_gt = el["d_eyes"], el["bb_gt"]
            x_pred, y_pred = pred[i, 0:68], pred[i, 68:]
            # Convert prediction
            bb_gt = self.rescale_bb(el["im_gray"].shape, bb_gt)
            [x1, y1, x2, y2] = [int(e) for e in bb_gt]
            w_bb, h_bb = y2 - y1, x2 - x1
            # Normalize in the new box

            rx = h / float(h_bb)
            ry = w / float(w_bb)

            x_pred += 0.5
            y_pred += 0.5
            x_pred *= h
            y_pred *= w
            x_pred /= rx
            y_pred /= ry
            x_pred += x1
            y_pred += y1
            y_reshaped[:, 0, i] = annox  # x
            y_reshaped[:, 1, i] = annoy  # y
            model_output_reshaped[:, 0, i] = x_pred  # x
            model_output_reshaped[:, 1, i] = y_pred  # x

        l_nrmse = octave.compute_error_ibug(y_reshaped, model_output_reshaped)
        for i in range(saved_pred.shape[0]):
            el = l_infos[i]
            img_rgb = el["im_rgb"]
            img_name = el["im_name"]
            annox, annoy = el["annox"], el["annoy"]
            d_eyes, bb_gt = el["d_eyes"], el["bb_gt"]
            x_pred, y_pred = saved_pred[i, 0:68], saved_pred[i, 68:]
            # Convert prediction
            bb_gt = self.rescale_bb(el["im_gray"].shape, bb_gt)
            [x1, y1, x2, y2] = [int(e) for e in bb_gt]
            w_bb, h_bb = y2 - y1, x2 - x1
            # Norlalize in the new box

            rx = h / float(h_bb)
            ry = w / float(w_bb)
            x_pred += 0.5
            y_pred += 0.5
            x_pred *= h
            y_pred *= w
            x_pred /= rx
            y_pred /= ry
            x_pred += x1
            y_pred += y1

            nrmse = l_nrmse[i]
            tag_oc = None
            if tags_oc is not None:
                tag_oc = tags_oc[i]
            if seg:
                fig = self.plot_over_img_seg(img_rgb, annox, annoy, x_pred,
                                             y_pred, bb_gt, tag_oc=tag_oc)
            else:
                fig = self.plot_over_img(img_rgb, annox, annoy, x_pred,
                                         y_pred, bb_gt)
            out.append({"fig": fig, "img_name": img_name, "nrmse": nrmse})

        return np.asarray(l_nrmse), out

    def calculate_auc(self, y, dx):
        """ calculate the Area Under the Curve using the average of the
            estimated areas by the two composites the Simpsons's and the
            trapezoidal rules.
        """
        AUC = (simps(y, dx=dx) + trapz(y, dx=dx)) / 2.
        return AUC

    def calculate_auc_cdf(self, l_nrmse, bx, dx):
        """Draw the CDF curve and calculate its AUC."""
        x = np.arange(0, bx, dx)
        nbr_x = x.size
        cdf = np.zeros(nbr_x)
        # CDF curve
        for i in xrange(nbr_x):
            cdf[i] = np.sum((l_nrmse <= x[i]) * 1.)
        cdf /= float(len(l_nrmse))  # Get the percentage
        cdf0_1 = np.sum((l_nrmse <= 0.1) * 1.) / float(len(l_nrmse))
        # AUC
        area = self.calculate_auc(cdf, dx)
        all_area = bx * 1.
        auc = 100. * area/all_area

        return cdf, cdf0_1, auc

    def plot_cdf(self, cdf, tag, cdf0_1, auc, bx, dx):
        """Plot the cdf curve"""
        plt.close("all")
        x = np.arange(0, bx, dx)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(x, cdf, label="CDF")
        ax.grid(True)
        plt.xlabel("NRMSE")
        plt.ylabel("Data proportion")
        plt.legend(loc=4, prop={'size': 8}, fancybox=True, shadow=True)
        plt.title("CDF curve: " + tag + ". CDF0.1: " + str(prec2 % cdf0_1) +
                  " . AUC:" + str(prec2 % auc) + ".")
        return fig

    def plot_cdf_model_and_meansh(self, cdfs, tag, cdf0_1s, aucs, bx, dx):
        plt.close("all")
        x = np.arange(0, bx, dx)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(x, cdfs[0], label="CDF model")
        ax.plot(x, cdfs[1], label="CDF mean shape")
        ax.grid(True)
        plt.xlabel("NRMSE")
        plt.ylabel("Data proportion")
        plt.legend(loc=4, prop={'size': 8}, fancybox=True, shadow=True)
        plt.title(
            "CDF curve: " + tag + ". Model: CDF0.1: " +
            str(prec2 % cdf0_1s[0]) + " . AUC:" + str(prec2 % aucs[0]) +
            ".\n" + ". MSh: CDF0.1: " +
            str(prec2 % cdf0_1s[1]) + " . AUC:" + str(prec2 % aucs[1]) + ".\n")
        return fig

    def sample_from_list(self, l_samples, w, h, d=10, color_mode=None,
                         ft_extract=None, get_original=False):
        x = []  # np.empty((1, w*h), dtype=theano.config.floatX)
        y = []  # np.empty((1, 68*2), dtype=theano.config.floatX)
        kk = 0
        for el in l_samples:
            im_gray = el["img_gray"]
            print "\r Image ", kk, "/", len(l_samples),
            sys.stdout.flush()
            kk += 1
            bb_gt = el["bb_ground_truth"]
            annox = el["annox"]
            annoy = el["annoy"]
            bb = self.rescale_bb(im_gray.shape, bb_gt)
            img_shape = im_gray.shape
            im_rgb = None
            if ft_extract is not None:
                im_rgb = el["img_rgb"].astype(np.float32)
            if d == 0:
                if ft_extract is None:
                    s, xs, ys = self.sample_no_ag(
                        im_gray, bb, annox, annoy, w, h,
                        get_original=get_original)
                else:
                    s, xs, ys = self.sample_no_ag(
                        im_rgb, bb, annox, annoy, w, h, color_mode, ft_extract,
                        get_original=get_original)
                x.append(s.flatten())
                y.append(xs.tolist() + ys.tolist())
            else:
                k = 0
                # First: Extract the original bbox
                if ft_extract is None:
                    s, xs, ys = self.sample_no_ag(
                        im_gray, bb, annox, annoy, w, h,
                        get_original=get_original)
                else:
                    s, xs, ys = self.sample_no_ag(
                        im_rgb, bb, annox, annoy, w, h, color_mode, ft_extract,
                        get_original=get_original)
                x.append(s.flatten())
                y.append(xs.tolist() + ys.tolist())
                # Second: do data augmentation
                for dx in range(-d, d+1, 4):
                    for dy in range(-d, d+1, 4):
                        [x1, y1, x2, y2] = bb
                        x1 += dx
                        y1 += dy
                        x2 += dx
                        y2 += dy
                        if x1 < 0:
                            x1 = 0
                        if y1 < 0:
                            y1 = 0
                        if x2 >= img_shape[1]:
                            x2 = img_shape[1] - 1
                        if y2 >= img_shape[0]:
                            y2 = img_shape[0] - 1
                        bb_new = [x1, y1, x2, y2]
                        if x2 - x1 < 50:
                            # print ">>>>>>>>>>>>>>>>>>> Skipped x", bb_new
                            continue
                        if y2 - y1 < 50:
                            # print ">>>>>>>>>>>>>>>>>>> Skipped y", bb_new
                            continue
#                        fig = self.debug_plot_over_img(im_gray, annox, annoy, bb_new, bb_gt)
#                        fig.savefig("wtf2.png")
                        if ft_extract is None:
                            s, xs, ys = self.sample_no_ag(
                                im_gray, bb_new, annox, annoy, w, h,
                                get_original=get_original)
                        else:
                            s, xs, ys = self.sample_no_ag(
                                im_rgb, bb_new, annox, annoy, w, h, color_mode,
                                ft_extract, get_original=get_original)
#                        sm.imsave("fd/" + str(k) + '.png', s)
#                        print k
#                        k += 1
                        # Add noise to the images
                        s = s.flatten()
                        n_to_prod = np.array(np.random.binomial(
                            n=1, p=1 - corruption_level,
                            size=s.size),
                            dtype=theano.config.floatX)
#                        sm.imsave("fd/" + str(k) + '.png',
#                                  np.multiply(s, n_to_prod).reshape((w, h)))
#                        sm.imsave("fd/" + str(k) + '_orig.png',
#                                  s.reshape((w, h)))
#                        k += 1
#                        raw_input("Halt")
                        x.append(np.multiply(s, n_to_prod))
                        y.append(xs.tolist() + ys.tolist())

        x = np.asarray(x, dtype=theano.config.floatX)
        x = self.normalize_x_e_wise(x)
        y = np.asarray(y, dtype=theano.config.floatX)
        return x, y

    def normalize_y_negative(self, y):
        """Normalize y intro [-1, 1]."""
        y = y - 0.5

        return y

    def normalize_x_e_wise(self, x):
        """Normalize each row to be in [-1, 1]."""
        for i in range(x.shape[0]):
            vec = x[i]
            min_vec, max_vec = np.min(vec), np.max(vec)
            if max_vec != min_vec:
                vec = (vec - min_vec)/(max_vec-min_vec)
            else:
                vec = vec - min_vec
            x[i] = vec

        return x

    def suffle_x_y(self, x, y):
        xh = x.shape[1]
        data = np.hstack((x, y))
        for i in range(10):
            np.random.shuffle(data)
        x = data[:, :xh]
        y = data[:, xh:]

        return x, y

    def get_data(self, l_samples, xy, x, y):
        xy_data = l_samples[:xy]
        x_data = l_samples[xy:xy+x]
        y_data = l_samples[xy+x:xy+x+y]
        return xy_data, x_data, y_data

    def turn_to_nan(self, x):
        for i in range(x.shape[0]):
            x[i, :] = np.float32(np.nan)

        return x

    def chunks(self, list_s, n):
        out = []
        for i in range(0, len(list_s), n):
            out.append(list_s[i:i+n])
        return out

    def split_data(self, list_samples, xys, xy, x, y, w, h, fd_out, d=10,
                   type_mod=None, weights_path=None, color_mode=None):
        """xys: number of samples (x,y) in package."""
        ft_extract = None
        if type_mod is not None:
            ft_extract = self.create_ft_extractor(type_mod, weights_path)
        xys = int(xys)
        xy_data, x_data, y_data = self.get_data(list_samples, xy, x, y)
        xy_chunks = self.chunks(xy_data, xys)
        for ch in xy_chunks:
            print len(ch)
        x_chunks, y_chunks = None, None
        if x != 0:
            x_chunks = self.chunks(x_data, xys)
        if y != 0:
            y_chunks = self.chunks(y_data, xys)
        if type_mod is None:
            type_mod = ""
        if x == 0:
            i = 0
            for ch in xy_chunks:
                print "\n Chunk ", i, "/", len(xy_chunks)
                xy_x, xy_y = self.sample_from_list(ch, w, h, d, color_mode,
                                                   ft_extract)
                xy_x, xy_y = self.suffle_x_y(xy_x, xy_y)
                out = {"x": xy_x, "y": xy_y}
                p_file = type_mod + "ch_tr_" + str(xy) + "_" + str(x) + "_" +\
                    str(y) + "_" + str(i) + ".pkl"
                with open(fd_out + "/" + p_file, 'w') as f:
                    pkl.dump(out, f, pkl.HIGHEST_PROTOCOL)
                del xy_x
                del xy_y
                del out
                i += 1
        else:
            for i in range(len(xy_chunks)):
                print "\n Chunk ", i, "/", len(xy_chunks)
                ch_xy = xy_chunks[i]
                xy_x, xy_y = self.sample_from_list(ch_xy, w, h, d, color_mode,
                                                   ft_extract)
                xy_x, xy_y = self.suffle_x_y(xy_x, xy_y)
                x_x, x_y, y_x, x_y = None, None, None, None
                if i < len(x_chunks):
                    ch_x = x_chunks[i]
                    x_x, x_y = self.sample_from_list(ch_x, w, h, d, color_mode,
                                                     ft_extract)
                    x_x, x_y = self.suffle_x_y(x_x, x_y)
                    x_y = self.turn_to_nan(x_y)
                    xy_x = np.vstack((xy_x, x_x))
                    xy_y = np.vstack((xy_y, x_y))
                    xy_x, xy_y = self.suffle_x_y(xy_x, xy_y)
                if i < len(y_chunks):
                    ch_y = y_chunks[i]
                    y_x, y_y = self.sample_from_list(ch_y, w, h, d, color_mode,
                                                     ft_extract)
                    y_x, y_y = self.suffle_x_y(y_x, y_y)
                    y_x = self.turn_to_nan(y_x)
                    xy_x = np.vstack((xy_x, y_x))
                    xy_y = np.vstack((xy_y, y_y))
                    xy_x, xy_y = self.suffle_x_y(xy_x, xy_y)
                out = {"x": xy_x, "y": xy_y}
                p_file = type_mod + "ch_tr_" + str(xy) + "_" + str(x) + "_" +\
                    str(y) + "_" + str(i) + ".pkl"
                with open(fd_out + "/" + p_file, 'w') as f:
                    pkl.dump(out, f, pkl.HIGHEST_PROTOCOL)
                del xy_x
                del xy_y
                del out
                del x_x
                del x_y
                del y_x
                del y_y

    def prapere_valid(self, list_samples, fd_out, w, h, d=10,
                      type_mod=None, weights_path=None, color_mode=None):
        ft_extract = None
        if type_mod is not None:
            ft_extract = self.create_ft_extractor(type_mod, weights_path)
        x, y = self.sample_from_list(list_samples, w, h, d, color_mode,
                                     ft_extract)
        if type_mod is None:
            type_mod = ""
        with open(fd_out + '/' + type_mod + "valid.pkl", 'w') as f:
            out = {"x": x, "y": y}
            pkl.dump(out, f, pkl.HIGHEST_PROTOCOL)
            del x
            del y
            del out

    def find_mean_shape(self, fd):
        y_mean = None
        nbr = 0.
        for file in glob.glob(fd + "/ch_*.pkl"):
            with open(file, 'r') as f:
                if y_mean is None:
                    stuff = pkl.load(f)
                    y_mean = np.sum(stuff["y"], axis=0)
                else:
                    y_mean += np.sum(pkl.load(f)["y"], axis=0)
                nbr += stuff["y"].shape[0]
        y_mean /= nbr
        print y_mean.shape
        with open(fd + "mean_shape.pkl", 'w') as f:
            pkl.dump(y_mean, f, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
#    path_lfpw_tr = "../inout/data/face/lfpw_tr.pkl"
#    fd_out = "../inout/data/face/lfpw_tr"
#    data = FaceDataset()
#    data.debug_draw_set(path_lfpw_tr, fd_out)

    # LFPW: tr: 811, test: 224
    # Split1: tr: 550 vl: 61, x:0, y:0
    # Split2: tr: 550 vl: 61, x:100, y:100
    # Split 3: tr:750, vl: 61
    # w: 80, h: 100

    # ONLY LABELED DATA (x,y)
    chx = 3
    if chx == 0:
        # LFPW
        color = False
        if not color:
            path_lfpw_tr = "../inout/data/face/lfpw_tr.pkl"
            w, h = 50, 50
            print "Going gray"
        else:
            path_lfpw_tr = "../inout/data/face/color/lfpw_tr.pkl"
            print "Going rgb"
        new_size = [227, 227]
        color_mode = None
        type_mod = None
        weights_path = None
        if type_mod is "alexnet":
            weights_path = "../inout/weights/alexnet_weights.h5"
            new_size = [227, 227]
            color_mode = "rgb"
        if type_mod is "vgg_16":
            weights_path = "../inout/weights/vgg16_weights.h5"
            new_size = [224, 224]
            color_mode = "bgr"
        if type_mod is "vgg_19":
            weights_path = "../inout/weights/vgg19_weights.h5"
            new_size = [224, 224]
            color_mode = "bgr"
        if type_mod is "googlenet":
            weights_path = "../inout/weights/googlenet_weights.h5"
            new_size = [224, 224]
            color_mode = "bgr"
        if color:
            w, h = new_size[0], new_size[1]
        d = 0
        with open(path_lfpw_tr, 'r') as f:
            lfpw_all_tr = pkl.load(f)
        data = FaceDataset()
        # Size of mega lfpw tr: 4123
        sx = 135
        # vl
        l_s_vl = lfpw_all_tr[:sx]
        lfpw_out = "../inout/data/face/lfpw_data/"
        data.prapere_valid(l_s_vl, lfpw_out, w, h, d, type_mod, weights_path,
                           color_mode)
        # Split 1:
        xy, x, y = 811-sx, 0, 0
        xys = 1000
        d = 0
        data.split_data(lfpw_all_tr[sx:], xys, xy, x, y, w, h, lfpw_out, d,
                        type_mod, weights_path, color_mode)
        data.find_mean_shape(lfpw_out)

    elif chx == 1:
        # HELEN
        path_helen_tr = "../inout/data/face/helen_tr.pkl"
        w, h = 50, 50
        d = 0
        type_mod = None
        weights_path = None
        color_mode = None
        with open(path_helen_tr, 'r') as f:
            helen_all_tr = pkl.load(f)
        data = FaceDataset()
        # Size of mega helen tr: 4123
        sx = 200
        # vl
        l_s_vl = helen_all_tr[:sx]
        helen_out = "../inout/data/face/helen_data/"
        if not os.path.exists(helen_out):
            os.makedirs(helen_out)
        data.prapere_valid(l_s_vl, helen_out, w, h, d, type_mod, weights_path,
                           color_mode)
        # Split 1:
        xy, x, y = 2000-sx, 0, 0
        xys = 2000
        d = 0
        data.split_data(helen_all_tr[sx:], xys, xy, x, y, w, h, helen_out, d,
                        type_mod, weights_path, color_mode)
        data.find_mean_shape(helen_out)

    # Labeled + Unlabeled data
    chx = 3
    if chx == 2:
        # Use Helen (train, valid, test) as unlabeled data:
        fd1 = "../inout/data/face/helen_data/"
        # Get train and valid (already sampled)
        with open(fd1 + "ch_tr_1800_0_0_0.pkl", 'r') as f:
            tr = pkl.load(f)
        with open(fd1 + "valid.pkl", 'r') as f:
            vl = pkl.load(f)
        # Sample test
        path_ts = "../inout/data/face/helen_ts.pkl"
        with open(path_ts, 'r') as f:
            l_ts = pkl.load(f)
        data = FaceDataset()
        w, h = 50, 50
        d = 0
        x, y = data.sample_from_list(l_ts, w, h, d, color_mode=None,
                                     ft_extract=None)
        # x without y
        x1_helen = np.vstack((tr["x"], vl["x"], x))
        y1_helen = np.empty((x1_helen.shape[0], y.shape[1]), dtype=np.float32)
        y1_helen[:, :] = np.float32(np.nan)
        # y without x
        y2_helen = np.vstack((tr["y"], vl["y"], y))
        x2_helen = np.empty((y2_helen.shape[0], x.shape[1]), dtype=np.float32)
        x2_helen[:, :] = np.float32(np.nan)

        # LFPW:
        fd = "../inout/data/face/lfpw_data/"
        fd_out = "../inout/data/face/lfpw_u_helen/"
        if not os.path.exists(fd_out):
            os.makedirs(fd_out)
        with open(fd + "ch_tr_676_0_0_0.pkl", 'r') as f:
            lfpw_tr = pkl.load(f)
        mean_shape = np.mean(np.vstack((y2_helen, lfpw_tr['y'])), axis=0)
        with open(fd_out + "mean_shape.pkl", 'w') as f:
            pkl.dump(mean_shape, f, protocol=pkl.HIGHEST_PROTOCOL)
        mega_x = np.vstack((lfpw_tr["x"], x1_helen, x2_helen))
        mega_y = np.vstack((lfpw_tr['y'], y1_helen, y2_helen))
        # shuffle very well:
        mega_mtrix = np.hstack((mega_x, mega_y))
        nbr = mega_x.shape[1]
        for i in range(10000):
            print "Shuffling", i
            np.random.shuffle(mega_mtrix)
        mega_x = mega_mtrix[:, :nbr]
        mega_y = mega_mtrix[:, nbr:]
        with open(fd_out + "train.pkl", "w") as f:
            pkl.dump({'x': mega_x, 'y': mega_y}, f,
                     protocol=pkl.HIGHEST_PROTOCOL)
        with open(fd + "valid.pkl", 'r') as f:
            lfpw_vl = pkl.load(f)
        # Keep the same valid set.
        with open(fd_out + "valid.pkl", "w") as f:
            pkl.dump(lfpw_vl, f, protocol=pkl.HIGHEST_PROTOCOL)

    elif chx == 3:
        # Use LFPW (train, valid, test) as unlabeled data:
        fd1 = "../inout/data/face/lfpw_data/"
        # Get train and valid (already sampled)
        with open(fd1 + "ch_tr_676_0_0_0.pkl", 'r') as f:
            tr = pkl.load(f)
        with open(fd1 + "valid.pkl", 'r') as f:
            vl = pkl.load(f)
        # Sample test
        path_ts = "../inout/data/face/lfpw_ts.pkl"
        with open(path_ts, 'r') as f:
            l_ts = pkl.load(f)
        data = FaceDataset()
        w, h = 50, 50
        d = 0
        x, y = data.sample_from_list(l_ts, w, h, d, color_mode=None,
                                     ft_extract=None)
        # x without y
        x1_lfpw = np.vstack((tr["x"], vl["x"], x))
        y1_lfpw = np.empty((x1_lfpw.shape[0], y.shape[1]), dtype=np.float32)
        y1_lfpw[:, :] = np.float32(np.nan)
        # y without x
        y2_lfpw = np.vstack((tr["y"], vl["y"], y))
        x2_lfpw = np.empty((y2_lfpw.shape[0], x.shape[1]), dtype=np.float32)
        x2_lfpw[:, :] = np.float32(np.nan)

        # HELEN:
        fd = "../inout/data/face/helen_data/"
        fd_out = "../inout/data/face/helen_u_lfpw/"
        if not os.path.exists(fd_out):
            os.makedirs(fd_out)
        with open(fd + "ch_tr_1800_0_0_0.pkl", 'r') as f:
            helen_tr = pkl.load(f)
        mean_shape = np.mean(np.vstack((y2_lfpw, helen_tr['y'])), axis=0)
        with open(fd_out + "mean_shape.pkl", 'w') as f:
            pkl.dump(mean_shape, f, protocol=pkl.HIGHEST_PROTOCOL)
        mega_x = np.vstack((helen_tr["x"], x1_lfpw, x2_lfpw))
        mega_y = np.vstack((helen_tr['y'], y1_lfpw, y2_lfpw))
        # shuffle very well:
        mega_mtrix = np.hstack((mega_x, mega_y))
        nbr = mega_x.shape[1]
        for i in range(10000):
            print "Shuffling", i
            np.random.shuffle(mega_mtrix)
        mega_x = mega_mtrix[:, :nbr]
        mega_y = mega_mtrix[:, nbr:]
        with open(fd_out + "train.pkl", "w") as f:
            pkl.dump({'x': mega_x, 'y': mega_y}, f,
                     protocol=pkl.HIGHEST_PROTOCOL)
        with open(fd + "valid.pkl", 'r') as f:
            helen_vl = pkl.load(f)
        # Keep the same valid set.
        with open(fd_out + "valid.pkl", "w") as f:
            pkl.dump(helen_vl, f, protocol=pkl.HIGHEST_PROTOCOL)
