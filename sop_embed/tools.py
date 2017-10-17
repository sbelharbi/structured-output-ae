from sop_embed.layers import HiddenLayer

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import numpy as np
import os
import sys
import datetime as DT
import matplotlib.pylab as plt
import math
import cPickle as pkl
from scipy.misc import imresize
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from random import shuffle
import yaml

from sop_embed.layers import DropoutHiddenLayer
from sop_embed.layers import dropout_from_layer
from sop_embed.layers import DropoutIdentityHiddenLayer

from sop_embed.extra import relu
from sop_embed.extra import NonLinearity
from sop_embed.extra import get_non_linearity_fn
from sop_embed.extra import CostType
from sop_embed.extra import sharedX_value
from da import ConvolutionalAutoencoder
from da import DeepConvolutionLayer


floating = 10
prec2 = "%."+str(floating)+"f"


#def relu(x):
#    return T.switch(x > 0, x, 0)
#
#
#class NonLinearity:
#    RELU = "rectifier"
#    TANH = "tanh"
#    SIGMOID = "sigmoid"
#
#
#def get_non_linearity_fn(nonlinearity):
#        if nonlinearity == NonLinearity.SIGMOID:
#            return T.nnet.sigmoid
#        elif nonlinearity == NonLinearity.RELU:
#            return relu
#        elif nonlinearity == NonLinearity.TANH:
#            return T.tanh
#        elif nonlinearity is None:
#            return None


#class CostType:
#    MeanSquared = "MeanSquaredCost"
#    CrossEntropy = "CrossEntropy"


class ModelMLP(object):
    def __init__(self, layers_infos, input, l1_reg=0., l2_reg=0., tag="",
                 reg_bias=False, dropout=None, ft_extractor=None,
                 id_code=None):
        self.layers_infos = [
            {"n_in": l["n_in"],
             "n_out": l["n_out"],
             "activation": l["activation"]} for l in layers_infos]
        if dropout is None:
            dropout = [0. for i in range(len(layers_infos))]
        self.tag = tag
        self.layers = []
        self.layers_dropout = []
        self.layer_dropout_code = None
        self.layer_code = None
        self.params_until_code = None
        self.dropout = dropout
        self.ft_extractor = ft_extractor
        self.state_train = theano.shared(0.)  # not used in this class.
        # catch the model's params in the memory WHITHOUT saving
        # them on disc because disc acces is so expensive on somme
        # servers.
        self.catched_params = []
        input_dropout = input
        if ft_extractor is None:
            self.x = input
        else:
            self.x = self.ft_extractor.inputs
            input_dropout = self.ft_extractor.output_dropout
        self.trg = T.fmatrix("y")
        self.params = []
        input_lr = input
        rng = np.random.RandomState(23355)
        input_lr_dp = DropoutIdentityHiddenLayer(rng, input_dropout,
                                                 dropout[0], False).output
        i = 0
        for layer, dout in zip(layers_infos, self.dropout[1:] + [0.]):
            self.layers_dropout.append(
                DropoutHiddenLayer(
                    rng=layer["rng"],
                    input=input_lr_dp,
                    n_in=layer["n_in"],
                    n_out=layer["n_out"],
                    dropout_rate=dout,
                    rescale=False,
                    W=layer["W"],
                    b=layer["b"],
                    activation=get_non_linearity_fn(layer["activation"])
                    )
            )
            self.layers.append(
                HiddenLayer(
                    input=input_lr,
                    n_in=layer["n_in"],
                    n_out=layer["n_out"],
                    W=self.layers_dropout[-1].W * (1 - dout),
                    b=self.layers_dropout[-1].b,
                    activation=get_non_linearity_fn(layer["activation"]),
                    rng=layer["rng"]
                    )
                )
            input_lr = self.layers[-1].output
            input_lr_dp = self.layers_dropout[-1].output
            self.params += self.layers_dropout[-1].params
            if id_code is not None:
                if i == id_code:
                    self.layer_dropout_code = self.layers_dropout[-1]
                    self.layer_code = self.layers[-1]
                    self.params_until_code = self.params[:]

        self.output = self.layers[-1].output
        self.output_dropout = self.layers_dropout[-1].output
        self.l1 = 0.
        self.l2 = 0.
        if self.ft_extractor is not None:
            self.params.extend(self.ft_extractor.params)
        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "model contains %i parameters" % nparams
        # dump params for debug
        todump = []
        for p in self.params:
            todump.append(p.get_value())
        with open("paramsmlp.pkl", "w") as f:
            pkl.dump(todump, f, protocol=pkl.HIGHEST_PROTOCOL)

#        with open("init_keras_param.pkl", 'r') as f:
#            keras_params = pkl.load(f)
#            print "initialized using keras params."
#            for param, param_vl in zip(self.params, keras_params):
#                param.set_value(param_vl)
        for l in self.layers:
            if l1_reg != 0.:
                self.l1 += abs(l.W).sum()
                if reg_bias:
                    self.l1 += abs(l.b).sum()
            if l2_reg != 0.:
                self.l2 += (l.W**2).sum()
                if reg_bias:
                    self.l2 += (l.b**2).sum()
        if self.l1 != 0.:
            self.l1 *= sharedX_value(l1_reg, "l1_reg")
        if self.l2 != 0.:
            self.l2 *= sharedX_value(l2_reg, "l2_reg")

    def catch_params(self):
        self.catched_params = [param.get_value() for param in self.params]

    def save_params(self, weights_file, catched=False):
        """Save the model's params."""
        with open(weights_file, "w") as f:
            if catched:
                if self.catched_params != []:
                    params_vl = self.catched_params
                else:
                    raise ValueError(
                        "You asked to save catched params," +
                        "but you didn't catch any!!!!!!!")
            else:
                params_vl = [param.get_value() for param in self.params]
            ft_extractor = False
            if self.ft_extractor is not None:
                ft_extractor = True
            stuff = {"layers_infos": self.layers_infos,
                     "params_vl": params_vl,
                     "tag": self.tag,
                     "dropout": self.dropout,
                     "ft_extractor": ft_extractor}
            pkl.dump(stuff, f, protocol=pkl.HIGHEST_PROTOCOL)

    def set_params_vals(self, weights_file):
        """Set the model's params."""
        with open(weights_file, "r") as f:
            stuff = pkl.load(f)
            layers_infos, params_vl = stuff["layers_infos"], stuff["params_vl"]
            # Stuff to check
            keys = ["n_in", "n_out", "activation"]
            assert all(
                [l1[k] == l2[k]
                    for k in keys
                    for (l1, l2) in zip(layers_infos, self.layers_infos)])
            for param, param_vl in zip(self.params, params_vl):
                param.set_value(param_vl)


class ModelCNN(object):
    def __init__(self, layers_infos, input, crop_size, l1_reg=0., l2_reg=0.,
                 tag="", reg_bias=False, dropout=None, ft_extractor=None,
                 id_code=None, corruption_level=0., rnd=None):
        assert ft_extractor is None
        assert id_code is None
        self.crop_size = crop_size
        self.layers_infos = []
        # things to keep:
        dic_keys = ["n_in", "n_out", "activation", "type", "filter_shape",
                    "padding", "stride", "poolsize", "ignore_border", "mode",
                    "ratio", "use_1D_kernel"]
        self.dic_keys = dic_keys
        for l in layers_infos:
            tmp = {}
            if "type" in l.keys():
                assert l["type"] == "deep_conv_ae_in"
                sub_l = l["layer"]
                for ml in sub_l:
                    for k in ml.keys():
                        if k in dic_keys:
                            tmp[k] = ml[k]
                    self.layers_infos.append(tmp)
            else:
                for k in l.keys():
                    if k in dic_keys:
                        tmp[k] = l[k]
                self.layers_infos.append(tmp)
        # For recreating the network.
        config_arch = []
        for l in layers_infos:
            if "type" in l.keys():  # deep in conv ae.
                assert l["type"] == "deep_conv_ae_in"
                tmp = []
                for subl in l["layer"]:
                    tmp1 = {}
                    for k in subl.keys():
                        if k not in ["W", "b", "rng"]:
                            tmp1[k] = subl[k]
                        else:
                            tmp1[k] = None
                    tmp.append(tmp1)
                config_arch.append({"type": "deep_conv_ae_in",
                                    "layer": tmp})
            else:
                tmp = {}
                for k in l.keys():
                    if k not in ["W", "b", "rng"]:
                        tmp[k] = l[k]
                    else:
                        tmp[k] = None
                config_arch.append(tmp)

        self.config_arch = config_arch

        if dropout is None:
            dropout = [0. for i in range(len(layers_infos))]
        self.tag = tag
        self.layers = []
        self.layers_dropout = []
        self.layer_dropout_code = None
        self.layer_code = None
        self.params_until_code = None
        self.dropout = dropout
        self.ft_extractor = ft_extractor
        if rnd is None:
            rnd = np.random.RandomState(1231)
        self.theano_rng = RandomStreams(rnd.randint(2 ** 30))
        # catch the model's params in the memory WHITHOUT saving
        # them on disc because disc acces is so expensive on somme
        # servers.
        self.catched_params = []
        input_dropout = input
        if ft_extractor is None:
            self.x = input
        else:
            self.x = self.ft_extractor.inputs
            input_dropout = self.ft_extractor.output_dropout
        self.trg = T.fmatrix("y")
        self.params = []
        # a hack to use input noise to the supervised task
        # without duplicating the network: input_train, input_valid.
        self.corruption_level = corruption_level
        self.state_train = theano.shared(np.float32(1.))
        self.one = theano.shared(np.float32(1.))
        if self.corruption_level == 0.:
            input_layer = self.x
        else:
            input_layer = self.theano_rng.binomial(
                    self.x.shape, n=1, p=1-self.corruption_level,
                    dtype=theano.config.floatX) * self.x * self.state_train +\
                    (self.one - self.state_train) * self.x

        for layer in layers_infos:
            if "type" in layer.keys():  # deep conv ae.
                if layer["type"] == "deep_conv_ae_in":
                    self.layers.append(
                        DeepConvolutionLayer(input=input_layer,
                                             layers=layer["layer"],
                                             crop_size=crop_size))
                    input_layer = self.layers[-1].output.flatten(2)
            else:
                self.layers.append(
                    HiddenLayer(
                        input=input_layer,
                        n_in=layer["n_in"],
                        n_out=layer["n_out"],
                        W=layer["W"],
                        b=layer["b"],
                        activation=get_non_linearity_fn(layer["activation"]),
                        rng=layer["rng"]
                        )
                )
                input_layer = self.layers[-1].output
            self.params += self.layers[-1].params

        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output  # no dropout.
        self.l1 = 0.
        self.l2 = 0.
        if self.ft_extractor is not None:
            self.params.extend(self.ft_extractor.params)
        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "model contains %i parameters" % nparams
        # dump params for debug
#        todump = []
#        for p in self.params:
#            todump.append(p.get_value())
#        with open("paramsmlp.pkl", "w") as f:
#            pkl.dump(todump, f, protocol=pkl.HIGHEST_PROTOCOL)

        for param in self.params:
            if ("w" in param.name) or ("W" in param.name):
                if l1_reg != 0.:
                    self.l1 += abs(param).sum()
                if l2_reg != 0.:
                    self.l2 += (param**2).sum()
            elif ("b" in param.name) or ("B" in param.name):
                if l1_reg != 0. and reg_bias:
                    self.l1 += abs(param).sum()
                if l2_reg != 0. and reg_bias:
                    self.l2 += (param**2).sum()
        if self.l1 != 0.:
            self.l1 *= sharedX_value(l1_reg, "l1_reg")
            print "Performing l1 reg. Lambda=", l1_reg
        if self.l2 != 0.:
            self.l2 *= sharedX_value(l2_reg, "l2_reg")
            print "Performing l2 reg. Lambda=", l2_reg

    def catch_params(self):
        self.catched_params = [param.get_value() for param in self.params]

    def save_params(self, weights_file, catched=False):
        """Save the model's params."""
        with open(weights_file, "w") as f:
            if catched:
                if self.catched_params != []:
                    params_vl = self.catched_params
                else:
                    raise ValueError(
                        "You asked to save catched params," +
                        "but you didn't catch any!!!!!!!")
            else:
                params_vl = [param.get_value() for param in self.params]
            ft_extractor = False
            if self.ft_extractor is not None:
                ft_extractor = True
            stuff = {"layers_infos": self.layers_infos,
                     "params_vl": params_vl,
                     "tag": self.tag,
                     "dropout": self.dropout,
                     "ft_extractor": ft_extractor,
                     "dic_keys": self.dic_keys,
                     "config_arch": self.config_arch,
                     "crop_size": self.crop_size}
            pkl.dump(stuff, f, protocol=pkl.HIGHEST_PROTOCOL)

    def set_params_vals(self, weights_file):
        """Set the model's params."""
        with open(weights_file, "r") as f:
            stuff = pkl.load(f)
            layers_infos, params_vl = stuff["layers_infos"], stuff["params_vl"]
            # Stuff to check
            keys = stuff["dic_keys"]
            for (l1, l2) in zip(layers_infos, self.layers_infos):
                for k in keys:
                    if k in l1.keys():
                        assert l1[k] == l2[k]
            for param, param_vl in zip(self.params, params_vl):
                param.set_value(param_vl)


class StaticExponentialDecayWeightRate(object):
    """
    A callback to adjust the cost weight rate.
    The weight is used to weight the PRE-TRAINING sub-costs in the embedded
    pre-training.

    The weights are updated by: weight_sup = 1 - v, weight_in = v/2
    weight_out = v/2.
    v > 0, v = exp(-epochs_seen/slope)
    Parameters:
        anneal_start: int (default 0). The epoch when to start annealing.
        slope: float. The slope of the exp.
    """
    def __init__(self, slop, anneal_start=0):
        self._initialized = False
        self._anneal_start = anneal_start
        self.l_sup, self.l_in, self.l_out = None, None, None
        self.slop = float(slop)
        self.epochs_seen = 0

    def __call__(self, l_sup, l_in, l_out, epochs_seen, to_update):
        """Updates the weight rate according to the exp schedule.
        Input:
            l_sup: Theano shared variable.
            l_in: Theano shared variable.
            l_out: Theano shared variable.
            epochs_seen: Int. The number of seen epcohs.
            to_update: dict. indicate if to update l_in, l_out.
                {"l_in": True, "l_out": True}.
        """
        s_w = l_sup.get_value() + l_in.get_value() + l_out.get_value()
        assert s_w - 1 < 1e-3
        self.epochs_seen = epochs_seen

        if not self._initialized:
            self.l_sup, self.l_in, self.l_out = l_sup, l_in, l_out
            self._initialized = True

        if (self.epochs_seen >= self._anneal_start):
            self.l_sup.set_value(np.cast[theano.config.floatX](
                self.get_current_weight_rate()))
            val = 1. - self.l_sup.get_value()
            if to_update["l_in"] and to_update["l_out"]:
                self.l_in.set_value(np.cast[theano.config.floatX](val/2.))
                self.l_out.set_value(np.cast[theano.config.floatX](val/2.))
            elif to_update["l_in"] and not to_update["l_out"]:
                self.l_in.set_value(np.cast[theano.config.floatX](val))
            elif not to_update["l_in"] and to_update["l_out"]:
                self.l_out.set_value(np.cast[theano.config.floatX](val))

        s_w = l_sup.get_value() + l_in.get_value() + l_out.get_value()
        assert s_w - 1 < 1e-3

    def get_current_weight_rate(self):
        """Calculate the current weight cost rate according to the a
        schedule.
        """
        return np.max([0., 1. - np.exp(float(-self.epochs_seen)/self.slop)])


class StaticExponentialDecayWeightRateSingle(object):
    """
    A callback to adjust the cost weight rate.
    The weight is used to weight the PRE-TRAINING sub-costs in the embedded
    pre-training.

    The weights are updated by: weight_sup = 1 - v, weight_in = v/2
    weight_out = v/2.
    v > 0, v = exp(-epochs_seen/slope)
    Parameters:
        anneal_start: int (default 0). The epoch when to start annealing.
        slope: float. The slope of the exp.
    """
    def __init__(self, slop, anneal_start=0):
        self._initialized = False
        self._anneal_start = anneal_start
        self.w = None
        self.slop = float(slop)
        self.epochs_seen = 0

    def __call__(self, w, epochs_seen):
        """Updates the weight rate according to the exp schedule.
        Input:
            w: Theano shared variable.
            epochs_seen: Int. The number of seen epcohs.
        """
        self.epochs_seen = epochs_seen

        if not self._initialized:
            self.w = w
            self._initialized = True

        if (self.epochs_seen >= self._anneal_start):
            self.w.set_value(np.cast[theano.config.floatX](
                self.get_current_weight_rate()))

    def get_current_weight_rate(self):
        """Calculate the current weight cost rate according to the a
        schedule.
        """
        return np.max([0., 1. - np.exp(float(-self.epochs_seen)/self.slop)])


class StaticAnnealedWeightRate(object):
    """
    A callback to adjust the cost weight rate.
    The weight is used to weight the PRE-TRAINING sub-costs in the embedded
    pre-training.

    The weight is annealed by: weight_sup = weight_sup + v,
    weight_in = 1 - v/2, weight_sup = 1 - v/2.
    v > 0.
    The annealing process starts from the epoch T0 = 0 (by defautl) and
    finishes at some epoch T.
    At the epoch T, the weight must be 0. Hince, v = weight/(T-T0).
    Parameters:
        anneal_end: int. The epoch when to end the annealing
        anneal_start: int (default 0). The epoch when to start annealing.
    """
    def __init__(self, anneal_end, anneal_start=0):
        self._initialized = False
        self._anneal_start = anneal_start
        self._anneal_end = anneal_end
        self.l_sup, self.l_in, self.l_out = None, None, None
        self.v = 0.

    def __call__(self, l_sup, l_in, l_out, epochs_seen, to_update):
        """Updates the weight rate according to the annealing schedule.
        l_sup: Theano shared variable.
            l_in: Theano shared variable.
            l_out: Theano shared variable.
            epochs_seen: Int. The number of seen epcohs.
            to_update: dict. indicate if to update l_in, l_out.
                {"l_in": True, "l_out": True}.
        """
        s_w = l_sup.get_value() + l_in.get_value() + l_out.get_value()
        assert -1 < 1e-3

        if not self._initialized:
            self.l_sup, self.l_in, self.l_out = l_sup, l_in, l_out

            distance = float(self._anneal_end - self._anneal_start)
            if distance == 0:
                self.v = 0
            else:
                self.v = float(1. - self.l_sup.get_value()) / distance

            self._initialized = True

        if (epochs_seen >= self._anneal_start) and \
                (epochs_seen <= self._anneal_end):
            self.l_sup.set_value(np.cast[theano.config.floatX](
                self.get_current_weight_rate()))
            if epochs_seen == self._anneal_end:
                self.l_sup.set_value(np.cast[theano.config.floatX](1.))

            val = 1. - self.l_sup.get_value()
            if to_update["l_in"] and to_update["l_out"]:
                self.l_in.set_value(np.cast[theano.config.floatX](val/2.))
                self.l_out.set_value(np.cast[theano.config.floatX](val/2.))
            elif to_update["l_in"] and not to_update["l_out"]:
                self.l_in.set_value(np.cast[theano.config.floatX](val))
            elif not to_update["l_in"] and to_update["l_out"]:
                self.l_out.set_value(np.cast[theano.config.floatX](val))

#        print l_sup.get_value(), l_in.get_value(), l_out.get_value()

        s_w = l_sup.get_value() + l_in.get_value() + l_out.get_value()
        assert s_w - 1 < 1e-3

    def get_current_weight_rate(self):
        """Calculate the current learning rate according to the annealing
        schedule.
        """
        return np.max([0., self.l_sup.get_value() + self.v])


class StaticAnnealedWeightRateSingle(object):
    """
    A callback to adjust the cost weight rate.
    The weight is used to weight the PRE-TRAINING sub-costs in the embedded
    pre-training.

    The weight is annealed by: weight_sup = weight_sup + v,
    weight_in = 1 - v/2, weight_sup = 1 - v/2.
    v > 0.
    The annealing process starts from the epoch T0 = 0 (by defautl) and
    finishes at some epoch T.
    At the epoch T, the weight must be 0. Hince, v = weight/(T-T0).
    Parameters:
        anneal_end: int. The epoch when to end the annealing
        anneal_start: int (default 0). The epoch when to start annealing.
    """
    def __init__(self, anneal_end, down, init_vl, end_vl, anneal_start=0):
        self._initialized = False
        self._anneal_start = anneal_start
        self._anneal_end = anneal_end
        self.w = None
        self.v = 0.
        self.epochs_seen = 0
        self.down = down
        self.init_vl = init_vl
        self.end_vl = end_vl

    def __call__(self, w, epochs_seen):
        """Updates the weight rate according to the annealing schedule.
        l_sup: Theano shared variable.
            l_in: Theano shared variable.
            l_out: Theano shared variable.
            epochs_seen: Int. The number of seen epcohs.
            to_update: dict. indicate if to update l_in, l_out.
                {"l_in": True, "l_out": True}.
        """
        self.epochs_seen = epochs_seen
        if not self._initialized:
            self.w = w

            distance = float(self._anneal_end - self._anneal_start)
            if distance == 0:
                self.v = 0
            else:
                if self.down:
                    self.v = float(self.end_vl - self.init_vl) / distance
                else:
                    self.v = float(self.init_vl - self.end_vl) / distance

            self._initialized = True

        if self.epochs_seen == self._anneal_start:
            self.w.set_value(np.cast[theano.config.floatX](self.init_vl))
        if (self.epochs_seen >= self._anneal_start) and \
                (self.epochs_seen <= self._anneal_end):
            self.w.set_value(np.cast[theano.config.floatX](
                self.get_current_weight_rate()))
            if epochs_seen == self._anneal_end:
                if self.down:
                    self.w.set_value(np.cast[theano.config.floatX](0.))
                else:
                    self.w.set_value(np.cast[theano.config.floatX](1.))

    def get_current_weight_rate(self):
        """Calculate the current learning rate according to the annealing
        schedule.
        """
        return np.max([0., self.w.get_value() + self.v])


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.

    Parameters
    ----------
    arr : array_like
        The data to be converted.

    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


def sharedX_mtx(mtx, name=None, borrow=None, dtype=None):
    """Share a matrix value with type theano.confgig.floatX.
    Parameters:
        value: matrix array
        name: variable name (str)
        borrow: boolean
        dtype: the type of the value when shared. default: theano.config.floatX
    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(
        np.array(mtx, dtype=dtype), name=name, borrow=borrow)


def contains_nan(array):
    """Check whether a 'numpy.ndarray' contains any 'numpy.nan' values.

    array: a numpy.ndarray array

    Returns:
    contains_nan: boolean
        `True` if there is at least one 'numpy.nan', `False` otherwise.
    """
    return np.isnan(np.min(array))


def contains_inf(array):
    """ Check whether a 'numpy.ndarray' contains any 'numpy.inf' values.

    array: a numpy.ndarray array

    Returns:
    contains_inf: boolean
        `True` if there is a least one 'numpy.inf', `False` otherwise.
    """
    return np.isinf(np.nanmin(array)) or np.isinf(np.nanmax(array))


def isfinite(array):
    """Check if 'numpy.ndarray' contains any 'numpy.inf' or 'numpy.nan' values.

    array: a numpy.ndarray array

    Returns:
    isfinite: boolean
        `True` if there is no 'numpy.inf' and 'numpy.nan', `False` otherwise.
    """
    return np.isfinite(np.min(array)) and np.isfinite(np.max(array))


def get_cost(aes, l, eye=True):
    """Get the sum of all the reconstruction costs of the AEs.
    Input:
        aes_in: list. List of all the aes.
        l: shared variable or a list of shared variables for the importance
            weights.
    """
    costs = []
    for ae, i in zip(aes, range(len(aes))):
        if isinstance(ae, ConvolutionalAutoencoder):
            costs.append(l[i] * ae.get_train_cost()[0])
        else:
            costs.append(l[i] * ae.get_train_cost(face=eye)[0])
    cost = None
    if costs not in [[], None]:
        cost = reduce(lambda x, y: x + y, costs)
    return cost


def get_net_pure_cost(model, cost_type, eye=True):
    """Get the train cost of the network."""
    cost = None
    if eye:
        d_eyes = (
            (model.trg[:, 37] - model.trg[:, 46])**2 +
            (model.trg[:, 37] - model.trg[:, 46])**2).T
        if cost_type == CostType.MeanSquared:
            cost = T.mean(
                T.sqr(model.output_dropout - model.trg), axis=1) / d_eyes
        elif cost_type == CostType.CrossEntropy:
            cost = T.mean(
                T.nnet.binary_crossentropy(
                    model.output_dropout, model.trg), axis=1)
        else:
            raise ValueError("cost type unknow.")
    else:
        if cost_type == CostType.MeanSquared:
            cost = T.mean(
                T.sqr(model.output_dropout - model.trg), axis=1)
        elif cost_type == CostType.CrossEntropy:
            cost = T.mean(
                T.nnet.binary_crossentropy(
                    model.output_dropout, model.trg), axis=1)
        else:
            raise ValueError("cost type unknow.")
    return cost


def get_net_cost(model, cost_type, eye=True):
    """Get the train cost of the network."""
    cost = None
    if eye:
        d_eyes = (
            (model.trg[:, 37] - model.trg[:, 46])**2 +
            (model.trg[:, 37] - model.trg[:, 46])**2).T
        if cost_type == CostType.MeanSquared:
            cost = T.mean(
                T.sqr(model.output_dropout - model.trg), axis=1) / d_eyes
        elif cost_type == CostType.CrossEntropy:
            cost = T.mean(
                T.nnet.binary_crossentropy(
                    model.output_dropout, model.trg), axis=1)
        else:
            raise ValueError("cost type unknow.")
    else:
        if cost_type == CostType.MeanSquared:
            cost = T.mean(
                T.sqr(model.output_dropout - model.trg), axis=1)
        elif cost_type == CostType.CrossEntropy:
            cost = T.mean(
                T.nnet.binary_crossentropy(
                    model.output_dropout, model.trg), axis=1)
        else:
            raise ValueError("cost type unknow.")

    if model.l1 != 0.:
        cost += model.l1
    if model.l2 != 0.:
        cost += model.l2
    return cost


def prepare_updates(model, aes_in, aes_out, cost, cost_in, cost_out, cost_sup,
                    cost_code, learning_rate, l_sup, l_code,
                    updaters={"all": None, "in": None, "out": None},
                    in3D=False):
    """
    Prepare the updates function of the training algorithm
    (calculating the gradient ...).
    """
    # grads
    params_sup = model.params
    params_until_code = []
    if cost_code is not None:
        params_until_code = model.params_until_code
    params_in, params_out = [], []
    for l in aes_in:
        params_in += l.params
    for l in aes_out:
        params_out += l.params

    # All params (unique)
    all_params = params_sup
    # Reviewer 1: Q1.
    for l in aes_in:
        all_params += [l.hidden.b_prime]

    for l in aes_out:
        all_params += [l.hidden.b]

    grads_all = T.grad(T.mean(cost), all_params)
    if cost_in is not None:
        grads_in = T.grad(T.mean(cost_in), params_in)
    if cost_out is not None:
        grads_out = T.grad(T.mean(cost_out), params_out)
    if cost_code is not None:
        grads_code = T.grad(T.mean(l_code * cost_code), params_until_code)
    lr = learning_rate

    # Updates
    lr_sc_all = list(
        [sharedX_value(1.) for i in xrange(len(all_params))])
    lr_sc_in = list(
        [sharedX_value(1.) for i in xrange(len(params_in))])
    lr_sc_out = list(
        [sharedX_value(1.) for i in xrange(len(params_out))])
    lr_sc_code = list(
        [sharedX_value(1.) for i in xrange(len(params_until_code))])

    # case 1: only supervised data
    if updaters["all"] is not None:
            updates_all = updaters["all"].get_updates(
                lr, all_params, grads_all, lr_sc_all)
    else:
        updates_all = [
            (param, param - lr * grad)
            for (param, grad) in zip(all_params, grads_all)]

    if cost_in is not None:
        # case 2: only unsupervised data in
        if updaters["in"] is not None:
            updates_in = updaters["in"].get_updates(
                lr, params_in, grads_in, lr_sc_in)
        else:
            updates_in = [
                (param, param - learning_rate * grad)
                for (param, grad) in zip(params_in, grads_in)]

    if cost_out is not None:
        # case 3: only unsupervised data out
        if updaters["out"] is not None:
            updates_out = updaters["out"].get_updates(
                lr, params_out, grads_out, lr_sc_out)
        else:
            updates_out = [
                (param, param - learning_rate * grad)
                for (param, grad) in zip(params_out, grads_out)]

    if cost_code is not None:
        # case 4: only supervised data
        if updaters["code"] is not None:
            updates_code = updaters["code"].get_updates(
                lr, params_until_code, grads_code, lr_sc_code)
        else:
            updates_code = [
                (param, param - learning_rate * grad)
                for (param, grad) in zip(params_until_code, grads_code)]

    if in3D:
        x_train = T.tensor4('x_train')
    else:
        x_train = T.matrix('x_train')

    state_train = T.scalar(dtype=theano.config.floatX)
    y_train = T.matrix('y_train')
    theano_args = [x_train, y_train, theano.In(state_train, value=1.)]

    # compile the update theano functions
    # case 1: only supervised data
    if (cost_in is None) and (cost_out is None):
        givens = {model.x: x_train, model.state_train: state_train,
                  model.trg: y_train}

    if (cost_in is not None) and (cost_out is None):
        givens = {
            model.x: x_train, model.state_train: state_train,
            model.trg: y_train, aes_in[0].input: x_train}

    if (cost_in is None) and (cost_out is not None):
        givens = {
            model.x: x_train, model.state_train: state_train,
            model.trg: y_train, aes_out[0].input: y_train}

    if (cost_in is not None) and (cost_out is not None):
        givens = {
            model.x: x_train, model.state_train: state_train,
            model.trg: y_train,
            aes_in[0].input: x_train, aes_out[0].input: y_train}

    sgd_update_all = theano.function(
        inputs=theano_args,
        outputs=[T.mean(cost)],
        updates=updates_all,
        givens=givens,
        on_unused_input='ignore')
    sgd_update_code = None
    if cost_code is not None:
        sgd_update_code = theano.function(
            inputs=theano_args,
            outputs=[T.mean(cost_code)],
            updates=updates_code,
            givens=givens,
            on_unused_input='ignore')
    sgd_update_in = None
    if cost_in is not None:
        # case 2: only unsupervised data in
        sgd_update_in = theano.function(
            inputs=[x_train],
            outputs=[T.mean(cost_in)],
            updates=updates_in,
            givens={aes_in[0].input: x_train},
            on_unused_input='ignore')

    sgd_update_out = None
    if cost_out is not None:
        # case 3: only unsupervised data out
        sgd_update_out = theano.function(
            inputs=[y_train],
            outputs=[T.mean(cost_out)],
            updates=updates_out,
            givens={aes_out[0].input: y_train}, on_unused_input='ignore')

    # Track the cost of the supervised task only (not weighted)
    cost_sup_fn = theano.function(
        inputs=[x_train, y_train],
        outputs=[T.mean(cost_sup)],
        givens={
            model.x: x_train,
            model.trg: y_train
            },
        on_unused_input='ignore')

    updates = {"all": sgd_update_all,
               "in": sgd_update_in,
               "out": sgd_update_out,
               "cost_sup": cost_sup_fn,
               "code": sgd_update_code}
    return updates


def get_cost_code(model, aes_out):
    print "warning!!!: we consider only one output ae. fix this in the future."
    ae_out = aes_out[-1]
    code = ae_out.encode(ae_out.input)
    output_link = model.layer_dropout_code.output
    cost = T.mean(T.sqr(output_link - code), axis=1)

    return cost


def theano_fns(model,
               aes_in, aes_out, l_in, l_out, l_sup, l_code,
               learning_rate, cost_type,
               updaters={"all": None, "in": None, "out": None},
               max_colm_norm=False, max_norm=15.0, eye=True, in3D=False,
               id_code=None, use_dice=False):
    """
    """
    # code cost
    cost_code = None
    if id_code is not None:
        cost_code = get_cost_code(model, aes_out)
    # Get cost in
    cost_in = get_cost(aes_in, l_in, eye=False)
    # Get cost out
    cost_out = get_cost(aes_out, l_out, eye=eye)
    # Get cost network
    cost_net = get_net_cost(model, cost_type, eye=eye)
    # Get pure cost net
    cost_net_pure = get_net_pure_cost(model, cost_type, eye=eye)
    # Get the model error
    if use_dice:
        insec = T.sum(model.trg * model.output, axis=1)
        tmp = 1 - 2.0 * insec/(T.sum(model.trg, axis=1) + T.sum(model.output,
                               axis=1))
        error = T.mean(tmp)
    else:
        error = T.mean(T.mean((model.output - model.trg)**2, axis=1))

    # Total cost
    embed_cost = sharedX_value(0.)
    # Reviwer 1: Q1.
    if cost_in is not None:
        embed_cost += cost_in
    if cost_out is not None:
        embed_cost += cost_out
    if cost_code is not None:
        embed_cost += l_code * cost_code
    embed_cost += l_sup * cost_net
    # embed_cost = l_sup * cost_net

    # Train fn: Updates
    train_updates = prepare_updates(
        model, aes_in, aes_out, embed_cost, cost_in, cost_out, cost_net_pure,
        cost_code, learning_rate, l_sup, l_code,
        updaters=updaters, in3D=in3D)
    # Evaluation fn
    if in3D:
        x = T.tensor4('x')
    else:
        x = T.fmatrix("x")
    y = T.fmatrix("y")
    state_train = T.scalar(dtype=theano.config.floatX)

    output_fn_vl = [error, model.output]

    eval_fn = theano.function(
        [x, y, theano.In(state_train, value=0.)], output_fn_vl,
        givens={model.x: x,
                model.state_train: state_train,
                model.trg: y},
        on_unused_input='ignore')

    return train_updates, eval_fn


def get_eval_fn(model, in3D=False, use_dice=False):
    """Compile the evaluation function of the model."""
    if use_dice:
        insec = T.sum(model.trg * model.output, axis=1)
        tmp = 1 - 2.0 * insec/(T.sum(model.trg, axis=1) + T.sum(model.output,
                               axis=1))
        error = T.mean(tmp)
    else:
        error = T.mean(T.mean(T.power(model.output - model.trg, 2), axis=1))
    if in3D:
        x = T.tensor4('x')
    else:
        x = T.fmatrix("x")
    y = T.fmatrix("y")

    theano_arg_vl = [x, y]
    output_fn_vl = [error, model.output]

    eval_fn = theano.function(
        theano_arg_vl, output_fn_vl,
        givens={model.x: x,
                model.trg: y})

    return eval_fn


def evaluate_model(list_minibatchs_vl, eval_fn):
    """Evalute the model over a set."""
    error, output = None, None
    for mn_vl in list_minibatchs_vl:
        x = theano.shared(
            mn_vl['x'], borrow=True).get_value(borrow=True)
        y = theano.shared(
            mn_vl['y'], borrow=True).get_value(borrow=True)

        [error_mn, output_mn] = eval_fn(x, y)
        if error is None:
            error = error_mn
            output = output_mn
        else:
            error = np.vstack((error, error_mn))
            output = np.vstack((output, output_mn))
    return error, output


def evaluate_model_3D_unsup(list_minibatchs_vl, eval_fn):
    """Evalute the model over a set."""
    error, output, code = None, None, None
    for mn_vl in list_minibatchs_vl:
        x = theano.shared(
            mn_vl['x'], borrow=True).get_value(borrow=True)

        [error_mn, output_mn, code_mn] = eval_fn(x)
        if error is None:
            error = error_mn
            output = output_mn
            code = code_mn
        else:
            error = np.vstack((error, error_mn))
            output = np.vstack((output, output_mn))
            code = np.vstack((code, code_mn))

    return error, output, code


def plot_fig(values, title, x_str, y_str, path, best_iter, std_vals=None):
    """Plot some values.
    Input:
         values: list or numpy.ndarray of values to plot (y)
         title: string; the title of the plot.
         x_str: string; the name of the x axis.
         y_str: string; the name of the y axis.
         path: string; path where to save the figure.
         best_iter: integer. The epoch of the best iteration.
         std_val: List or numpy.ndarray of standad deviation values that
             corresponds to each value in 'values'.
    """
    floating = 6
    prec = "%." + str(floating) + "f"

    if best_iter >= 0:
        if isinstance(values, list):
            if best_iter >= len(values):
                best_iter = -1
        if isinstance(values, np.ndarray):
            if best_iter >= np.size:
                best_iter = -1

        v = str(prec % np.float(values[best_iter]))
    else:
        v = str(prec % np.float(values[-1]))
        best_iter = -1
    if best_iter == -1:
        best_iter = len(values)
    fig = plt.figure()
    plt.plot(
        values,
        label="lower val: " + v + " at " + str(best_iter) + " " +
        x_str)
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    plt.title(title, fontsize=8)
    plt.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': 8})
    plt.grid(True)
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    del fig


def plot_stats(train_stats, ax, fold_exp, tag):
    """Plot what happened during training.
    Input:
        ax: string. "mb": minbatch, "epoch": epoch.
    """
    if ax is "mb":
        fd = fold_exp + "/minibatches"
        extra1 = "_mb"
        extra2 = "minibatches"
        extra3 = train_stats["best_mb"]
    else:
        fd = fold_exp + "/epochs"
        extra1 = ""
        extra2 = "epochs"
        extra3 = train_stats["best_epoch"]
    if not os.path.exists(fd):
        os.makedirs(fd)

    if train_stats["all_cost" + extra1] != []:
        if train_stats["code_cost" + extra1] != []:
            plot_fig(train_stats["all_cost" + extra1],
                     "Train cost: sum(in+out+sup+code) [" + extra2 +
                     "]. Case: " + tag,
                     extra2, "cost: in+out+sup+code",
                     fd+"/train-cost-in+out+sup-code-" + extra2 + ".png",
                     extra3)
        else:
            plot_fig(train_stats["all_cost" + extra1],
                     "Train cost: sum(in+out+sup) [" + extra2 +
                     "]. Case: " + tag,
                     extra2, "cost: in+out+sup",
                     fd+"/train-cost-in+out+sup-" + extra2 + ".png",
                     extra3)

    if train_stats["code_cost" + extra1] != []:
        plot_fig(train_stats["code_cost" + extra1],
                 "Train cost: code [" + extra2 + "]. Case: " + tag,
                 extra2, "cost: code",
                 fd+"/train-cost-code-" + extra2 + ".png",
                 extra3)

    if train_stats["in_cost" + extra1] != []:
        plot_fig(train_stats["in_cost" + extra1],
                 "Train cost: in [" + extra2 + "]. Case: " + tag,
                 extra2, "cost: in",
                 fd+"/train-cost-in-" + extra2 + ".png",
                 extra3)

    if train_stats["out_cost" + extra1] != []:
        plot_fig(train_stats["out_cost" + extra1],
                 "Train cost: out [" + extra2 + "]. Case: " + tag,
                 extra2, "cost: out",
                 fd+"/train-cost-out-" + extra2 + ".png",
                 extra3)

    if train_stats["tr_pure_cost" + extra1] != []:
        plot_fig(train_stats["tr_pure_cost" + extra1],
                 "Train cost: pure sup [" + extra2 + "]. Case: " + tag,
                 extra2, "cost: pure sup",
                 fd+"/train-cost-pure-sup-" + extra2 + ".png",
                 extra3)

    if train_stats["error_tr" + extra1] != []:
        plot_fig(train_stats["error_tr" + extra1],
                 "Model train error: [" + extra2 + "]. Case: " + tag,
                 extra2, "Train error",
                 fd+"/train-error-" + extra2 + ".png",
                 extra3)

    if train_stats["error_vl" + extra1] != []:
        plot_fig(train_stats["error_vl" + extra1],
                 "Model valid error: [" + extra2 + "]. Case: " + tag,
                 extra2, "Valid error",
                 fd+"/valid-error-" + extra2 + ".png",
                 extra3)


def print_stats_train(train_stats, epoch, ext, mb):
    in_cost, out_cost, all_cost, tr_pure_cost = 0., 0., 0., 0.
    error_vl, error_tr, code_cost = 0., 0., 0.

    if train_stats["code_cost"+ext] != []:
        code_cost = str(prec2 % train_stats["code_cost"+ext][-1])
    if train_stats["in_cost"+ext] != []:
        in_cost = str(prec2 % train_stats["in_cost"+ext][-1])
    if train_stats["out_cost"+ext] != []:
        out_cost = str(prec2 % train_stats["out_cost"+ext][-1])
    if train_stats["tr_pure_cost"+ext] != []:
        tr_pure_cost = str(prec2 % train_stats["tr_pure_cost"+ext][-1])
    if train_stats["error_tr"+ext]:
        error_tr = str(prec2 % train_stats["error_tr"+ext][-1])
    all_cost = str(prec2 % train_stats["all_cost"+ext][-1])
    error_vl = str(prec2 % train_stats["error_vl"+ext][-1])
    min_vl_err = str(prec2 % min(train_stats["error_vl"+ext]))

    if ext is "":
        print "\r Epoch [", str(epoch), "] Train infos:",
    else:
        print "\r Epoch [", str(epoch), "] mb [", str(mb), "] Train infos:",
#    print "\t all cost:", all_cost, " \t tr pure cost:", tr_pure_cost
#    print "\t in cost:", in_cost, " \t out cost:", out_cost
#    print "\t error tr:", error_tr, " \t error vl:", error_vl, " min vl:",\
#        min_vl_err
#    print "\t all cost: %s ,\t tr pure cost: %s, \t in cost: %s,"\
#        "\t out cost: %s,\t error tr: %s, \t error vl: %s, \t min vl: %s"\
#        % (all_cost, tr_pure_cost, in_cost, out_cost, error_tr, error_vl,
#            min_vl_err)
    print "all cost:", all_cost, "\t tr pure cost:", tr_pure_cost,\
        "\t in cost:", in_cost, "\t  out cost:", out_cost, "\t error tr:",\
        error_tr, "\t code cost:", code_cost, "\t error vl:", error_vl,\
        "\t min vl:", min_vl_err,
    sys.stdout.flush()


def train_one_epoch(train_updates, eval_fn, list_minibatchs_tr,
                    l_in, l_out, l_sup, list_minibatchs_vl,
                    model, aes_in, aes_out, epoch, fold_exp,
                    train_stats, vl_error_begin, tag):
    """ train the model's parameters for 1 epoch.
    """
    sgd_update_all = train_updates["all"]
    sgd_update_in = train_updates["in"]
    sgd_update_out = train_updates["out"]
    tr_pure_cost_fn = train_updates["cost_sup"]

    in_cost, out_cost, all_cost, tr_pure_cost = [], [], [], []
    error_vl, error_tr = [], []
    nb_mb, best_epoch, best_mb = 0, 0, 0
    vl_freq = 1
    plot_freq = 10

    for minibatch in list_minibatchs_tr:
        t0 = DT.datetime.now()
        in_cost_mb, out_cost_mb, all_cost_mb = 0., 0., 0.
        # share data ONLY when needed (save the GPU memory).
        d_in = minibatch['in']
        d_out = minibatch['out']
        d_sup = minibatch['sup']
        # Update unsupervised task in
        if (l_in.get_value() != 0.) and (d_in['x'] is not None) and\
                (sgd_update_in is not None):
            xx = theano.shared(d_in['x'], borrow=True).get_value(borrow=True)
            in_cost_mb = sgd_update_in(xx)[0]
            in_cost.append(in_cost_mb)
            train_stats["in_cost_mb"].append(in_cost_mb)
            del xx

        # Update unsupervised task out
        if (l_out.get_value() != 0.) and (d_out['y'] is not None) and\
                (sgd_update_out is not None):
            xx = theano.shared(d_out['y'], borrow=True).get_value(borrow=True)
            out_cost_mb = sgd_update_out(xx)[0]
            out_cost.append(out_cost_mb)
            train_stats["out_cost_mb"].append(out_cost_mb)
            del xx

        # Update the supervised task.
        if (l_sup.get_value() != 0.) and (d_sup['x'] is not None):
            x = theano.shared(
                d_sup['x'], borrow=True).get_value(borrow=True)
            y = theano.shared(
                d_sup['y'], borrow=True).get_value(borrow=True)

            # track the pure tr cost: do it before the update
            tr_pure_cost.append(tr_pure_cost_fn(x, y)[0])
            train_stats["tr_pure_cost_mb"].append(tr_pure_cost[-1])
            # Update the params.
            print x.shape, y.shape
            all_cost_mb = sgd_update_all(x, y)[0]

            # Evaluate over train set
            error_mn, _ = eval_fn(x, y)
            error_tr.append(np.mean(error_mn))
            train_stats["error_tr_mb"].append(error_tr[-1])
            del x
            del y

        # Collect costs
        all_cost.append(in_cost_mb + out_cost_mb + all_cost_mb)
        train_stats["all_cost_mb"].append(all_cost[-1])

        # Validation
        if nb_mb % vl_freq == 0:
            error_mn, _ = evaluate_model(list_minibatchs_vl, eval_fn)
            error_vl.append(np.mean(error_mn))
            train_stats["error_vl_mb"].append(error_vl[-1])
            # Pick the best model accordonly to the validation
            # error of the model.
            if len(train_stats["error_vl_mb"]) > 1:
                min_vl = np.min(train_stats["error_vl_mb"][:-1])
            else:
                min_vl = vl_error_begin
            if error_vl[-1] < min_vl:
                min_vl = error_vl[-1]
                train_stats["best_epoch"] = epoch
                train_stats["best_mb"] = epoch * len(list_minibatchs_tr) +\
                    nb_mb
                model.save_params(fold_exp + "/model.pkl")
                if model.ft_extractor is not None:
                        model.ft_extractor.model.save(fold_exp + "/cnn.pkl")
#                print "\t\t\tFound better model: SAVED. min vl:",\
#                    str(prec2 % min_vl)
                if aes_in != []:
                    for l, k in zip(aes_in, range(len(aes_in))):
                        l.save_params(fold_exp + "/ae_in_" + str(k) + ".pkl")
                if aes_out != []:
                    for l, k in zip(aes_out, range(len(aes_out))):
                        l.save_params(fold_exp + "/ae_out_" + str(k) + ".pkl")

        nb_mb += 1
        # Plot stats: mb
        if nb_mb % plot_freq == 0:
            plot_stats(train_stats, "mb", fold_exp, tag)
        # Print train infos mb
        print_stats_train(train_stats, epoch, "_mb", nb_mb-1)
#        print "Time mini-batch:", DT.datetime.now() - t0

#    # check if the params do not have bad values (inf, nan)
#    for param in self.params:
#        if not isfinite(param.get_value(borrow=True)):
#            raise Exception('Non finite values (inf or nan) in paramaters.')
#
#    get_info_params()
    # Stats
    stats = {"in_cost": in_cost,
             "out_cost": out_cost,
             "all_cost": all_cost,
             "tr_pure_cost": tr_pure_cost,
             "error_tr": error_tr,
             "error_vl": error_vl,
             "best_epoch": best_epoch,
             "best_mb": best_mb}

    return stats


def train_one_epoch_chuncks(train_updates, eval_fn, l_ch_tr,
                            l_in, l_out, l_sup, l_code, list_minibatchs_vl,
                            model, aes_in, aes_out, epoch, fold_exp,
                            train_stats, vl_error_begin, tag, tr_batch_size,
                            stop=False):
    """ train the model's parameters for 1 epoch.
    """
    sgd_update_all = train_updates["all"]
    sgd_update_in = train_updates["in"]
    sgd_update_out = train_updates["out"]
    sgd_update_code = train_updates["code"]
    tr_pure_cost_fn = train_updates["cost_sup"]

    in_cost, out_cost, all_cost, tr_pure_cost, code_cost = [], [], [], [], []
    error_vl, error_tr = [], []
    nb_mb, best_epoch, best_mb = 0, 0, 0
    if epoch <= 1000:
        freq_vl = 2000
    elif epoch <= 1900 and epoch > 1000:
        freq_vl = 2000
    elif epoch > 1900:
        freq_vl = 1
    plot_freq = 20
    catchted_once = False

    for ch in l_ch_tr:
        if isinstance(ch, str):
            print "\n Chunk:", ch
            with open(ch, 'r') as f:
                l_samples = pkl.load(f)
        else:
            l_samples = ch
        sharpe = False
        if model.ft_extractor is not None:
            sharpe = True
        list_minibatchs_tr = split_data_to_minibatchs_embed(
            l_samples, tr_batch_size, share=False, sharpe=sharpe)
#        for ts in xrange(100):
#            shuffle(list_minibatchs_tr)
        for minibatch in list_minibatchs_tr:
            t0 = DT.datetime.now()
            in_cost_mb, out_cost_mb, all_cost_mb = 0., 0., 0.
            # share data ONLY when needed (save the GPU memory).
            d_in = minibatch['in']
            d_out = minibatch['out']
            d_sup = minibatch['sup']
            # Update unsupervised task in
            check_in = [l.get_value() != 0. for l in l_in]
            if (any(check_in)) and (d_in['x'] is not None) and\
                    (sgd_update_in is not None):
                xx = theano.shared(
                    d_in['x'], borrow=True).get_value(borrow=True)
                in_cost_mb = sgd_update_in(xx)[0]
                in_cost.append(in_cost_mb)
                train_stats["in_cost_mb"].append(in_cost_mb)
                del xx

            # Update unsupervised task out
            if (l_out[0].get_value() != 0.) and (d_out['y'] is not None) and\
                    (sgd_update_out is not None):
                xx = theano.shared(
                    d_out['y'], borrow=True).get_value(borrow=True)
                out_cost_mb = sgd_update_out(xx)[0]
                out_cost.append(out_cost_mb)
                train_stats["out_cost_mb"].append(out_cost_mb)
                del xx

            # Update the code:
            if (sgd_update_code is not None) and (l_code.get_value() != 0.)\
                    and (d_sup['x'] is not None) and\
                    (l_out[0].get_value() != 0.):
                x = theano.shared(
                    d_sup['x'], borrow=True).get_value(borrow=True)
                y = theano.shared(
                    d_sup['y'], borrow=True).get_value(borrow=True)
                code_cost.append(sgd_update_code(x, y)[0])
                train_stats["code_cost_mb"].append(code_cost[-1])
                del x
                del y

            # Update the supervised task.
            if (l_sup.get_value() != 0.) and (d_sup['x'] is not None):
                x = theano.shared(
                    d_sup['x'], borrow=True).get_value(borrow=True)
                y = theano.shared(
                    d_sup['y'], borrow=True).get_value(borrow=True)
                # track the pure tr cost: do it before the update
                tr_pure_cost.append(tr_pure_cost_fn(x, y)[0])
                train_stats["tr_pure_cost_mb"].append(tr_pure_cost[-1])
                # Update the params.
                all_cost_mb = sgd_update_all(x, y)[0]

                # Evaluate over train set
                error_mn, _ = eval_fn(x, y)
                error_tr.append(np.mean(error_mn))
                train_stats["error_tr_mb"].append(error_tr[-1])
                del x
                del y
            # Collect costs
            all_cost.append(in_cost_mb + out_cost_mb + all_cost_mb)
            train_stats["all_cost_mb"].append(all_cost[-1])

            # Validation
            # on somme server the access to the disc costs so much
            # time because the disc is not in the same machine
            # nor the same vlan!!!!!!!!!!!.
            if (nb_mb % freq_vl == 0):
                error_mn, _ = evaluate_model(list_minibatchs_vl, eval_fn)
                error_vl.append(np.mean(error_mn))
                train_stats["error_vl_mb"].append(error_vl[-1])
                # Pick the best model accordonly to the validation error of
                # the model.
                if len(train_stats["error_vl_mb"]) > 1:
                    min_vl = np.min(train_stats["error_vl_mb"][:-1])
                else:
                    min_vl = vl_error_begin
                print "vl error", error_vl[-1], epoch, "best:", min_vl,\
                    "min vl:", np.min(train_stats["error_vl_mb"])
                if error_vl[-1] < min_vl:
                    min_vl = error_vl[-1]
                    train_stats["best_epoch"] = epoch
                    train_stats["best_mb"] = len(
                        train_stats["error_vl_mb"]) - 1
                    model.catch_params()
                    catchted_once = True
                    if model.ft_extractor is not None:
                        model.ft_extractor.model.save(fold_exp + "/cnn.pkl")
#                    print "\t\t\tFound better model: SAVED. min vl:",\
#                        str(prec2 % min_vl)
                    if aes_in != []:
                        for l, k in zip(aes_in, range(len(aes_in))):
                            l.catch_params()

                    if aes_out != []:
                        for l, k in zip(aes_out, range(len(aes_out))):
                            l.catch_params()

            nb_mb += 1

            # Print train infos mb
#            print_stats_train(train_stats, epoch, "_mb", nb_mb-1)
    # save o disc best models.
    if (epoch % 100 == 0) or stop:
        if stop:
            print "Going to end the training, but before, I'm gonna go ahead "\
                " and save the model params."
        # Plot stats: mb
        plot_stats(train_stats, "mb", fold_exp, tag)
        if catchted_once:
#            if epoch > 1:
#                with open(fold_exp + "/vl_error.txt", "w") as f:
#                    f.write(repr(np.min(train_stats["error_vl"])))
            model.save_params(fold_exp + "/model.pkl", catched=True)
            if aes_in != []:
                for l, k in zip(aes_in, range(len(aes_in))):
                    if (epoch % 100 == 0):
                        l.save_params(
                            fold_exp + "/ae_in_" + str(k) + ".pkl",
                            catched=True)
                if aes_out != []:
                    for l, k in zip(aes_out, range(len(aes_out))):
                        if (epoch % 100 == 0):
                            l.save_params(
                                fold_exp + "/ae_out_" + str(k) + ".pkl",
                                catched=True)
#            print "Time mini-batch:", DT.datetime.now() - t0

#    # check if the params do not have bad values (inf, nan)
#    for param in self.params:
#        if not isfinite(param.get_value(borrow=True)):
#            raise Exception('Non finite values (inf or nan) in paramaters.')
#
#    get_info_params()
    # Stats
    stats = {"in_cost": in_cost,
             "out_cost": out_cost,
             "all_cost": all_cost,
             "tr_pure_cost": tr_pure_cost,
             "error_tr": error_tr,
             "error_vl": error_vl,
             "best_epoch": best_epoch,
             "best_mb": best_mb,
             "code_cost": code_cost}

    return stats


def collect_stats_epoch(stats, train_stats):
    if stats["in_cost"] != []:
        train_stats["in_cost"].append(np.mean(stats["in_cost"]))
    if stats["out_cost"] != []:
        train_stats["out_cost"].append(np.mean(stats["out_cost"]))
    if stats["code_cost"] != []:
        train_stats["code_cost"].append(np.mean(stats["code_cost"]))
    train_stats["all_cost"].append(np.mean(stats["all_cost"]))
    train_stats["tr_pure_cost"].append(np.mean(stats["tr_pure_cost"]))
    train_stats["error_tr"].append(np.mean(stats["error_tr"]))
    train_stats["error_vl"].append(np.mean(stats["error_vl"]))

    return train_stats


def split_mini_batch(x_train, y_train, share=True, borrow=True):
    x, y, xx, yy = None, None, None, None
    # split the supervised from the unsupervised data
    # Sup: (x, y)
    if len(x_train.shape) == 2:
        col_tr_x = x_train[:, 0]
    else:
        col_tr_x = x_train[:, 0, 0, 0]
    col_tr_y = y_train[:, 0]

    ind_sup = np.where(
        np.logical_and(np.invert(np.isnan(col_tr_x)),
                       np.invert(np.isnan(col_tr_y))))
    if len(ind_sup[0] > 0):
        x = x_train[ind_sup[0]]
        y = y_train[ind_sup[0]]

    # In: x_sup + x_nosup
    ind_xx = np.where(
        np.logical_and(np.invert(np.isnan(col_tr_x)), np.isnan(col_tr_y)))
    if len(ind_xx[0]) > 0:
        xx = x_train[ind_xx[0]]
    if x is not None:
        if xx is not None:
            xx = np.vstack((x, xx))
        else:
            xx = x_train[ind_sup[0]]
    # Out: y_sup + y_nonsup
    ind_yy = np.where(
        np.logical_and(np.isnan(col_tr_x), np.invert(np.isnan(col_tr_y))))
    if len(ind_yy[0] > 0):
        yy = y_train[ind_yy[0]]
    if y is not None:
        if yy is not None:
            yy = np.vstack((y, yy))
        else:
            yy = y_train[ind_sup[0]]

    if share:
        if xx is not None:
            c_batch_in = {
                'x': theano.shared(
                    np.asarray(xx, dtype=theano.config.floatX),
                    borrow=borrow)}
        else:
            c_batch_in = {'x': None}
        if yy is not None:
            c_batch_out = {
                'y': theano.shared(
                    np.asarray(yy, dtype=theano.config.floatX),
                    borrow=borrow)}
        else:
            c_batch_out = {'y': None}
        if x is not None:
            c_batch_sup = {
                'x': theano.shared(np.asarray(x, dtype=theano.config.floatX),
                                   borrow=borrow),
                'y': theano.shared(np.asarray(y, dtype=theano.config.floatX),
                                   borrow=borrow)}
        else:
            c_batch_sup = {'x': None, 'y': None}
    else:
        c_batch_in = {'x': xx}
        c_batch_out = {'y': yy}
        c_batch_sup = {'x': x, 'y': y}

    return c_batch_in, c_batch_out, c_batch_sup


def split_data_to_minibatchs_embed(data, batch_size, share=False, borrow=True,
                                   sharpe=False):
    """Split a dataset to minibatchs to:
        1. Control the case where the size of the set is not divided by the
            btahc_size.
        2. Allows to share batch by batch when the size of data is too big to
            fit in the GPU shared memory.

    data: dictionnary with two keys:
        x: x (numpy.ndarray) for the supervised data
            may contains rows with only numpy.nan as values.
        y: y (numpy.ndarray) for the superrvised data
            may contains rows with only numpy.nan as values.
    batch_size: the size of the batch
    sharpe: Boolean. If True, ignore the rest of data with size less than
        batch size.

    Returns:
        list_minibatchs: list
            a list of dic of mini-batchs, each one has the 3 keys: "in",
            "out", "sup".
    """
    nbr_samples = data['x'].shape[0]
    if batch_size > nbr_samples:
        batch_size = nbr_samples

    nbr_batchs = int(nbr_samples / batch_size)
    list_minibatchs = []
    for index in xrange(nbr_batchs):
        x_train = data['x'][index * batch_size:(index+1) * batch_size]
        y_train = data['y'][index * batch_size:(index+1) * batch_size]
        c_batch_in, c_batch_out, c_batch_sup = split_mini_batch(
            x_train.astype(theano.config.floatX),
            y_train.astype(theano.config.floatX), share=share, borrow=borrow)

        list_minibatchs.append(
            {'in': c_batch_in, 'out': c_batch_out, 'sup': c_batch_sup})
    if not sharpe:
        # in case some samples are left
        if (nbr_batchs * batch_size) < nbr_samples:
            x_train = data['x'][(index+1) * batch_size:]
            y_train = data['y'][(index+1) * batch_size:]
            c_batch_in, c_batch_out, c_batch_sup = split_mini_batch(
                x_train.astype(theano.config.floatX),
                y_train.astype(theano.config.floatX), share=share,
                borrow=borrow)
            list_minibatchs.append(
                {'in': c_batch_in, 'out': c_batch_out, 'sup': c_batch_sup})

    return list_minibatchs


def split_data_to_minibatchs_eval(data, batch_size, sharpe=False):
    """Split a dataset to minibatchs to:
        1. Control the case where the size of the set is not divided by the
            btahc_size.
        2. Allows to share batch by batch when the size of data is too big to
            fit in the GPU shared memory.

    data: dictionnary with two keys:
        x: x (numpy.ndarray) for the supervised data
            may contains rows with only numpy.nan as values.
        y: y (numpy.ndarray) for the superrvised data
            may contains rows with only numpy.nan as values.
    batch_size: the size of the batch

    Returns:
        list_minibatchs: list
            a list of dic of mini-batchs, each one has the 2 keys: "x", "y".
    """
    nbr_samples = data['x'].shape[0]
    if batch_size > nbr_samples:
        batch_size = nbr_samples

    nbr_batchs = int(nbr_samples / batch_size)
    list_minibatchs = []
    for index in xrange(nbr_batchs):
        x_train = data['x'][index * batch_size:(index+1) * batch_size]
        y_train = data['y'][index * batch_size:(index+1) * batch_size]
        list_minibatchs.append({'x': x_train.astype(theano.config.floatX),
                                'y': y_train.astype(theano.config.floatX)})
    if not sharpe:
        # in case some samples are left
        if (nbr_batchs * batch_size) < nbr_samples:
            x_train = data['x'][(index+1) * batch_size:]
            y_train = data['y'][(index+1) * batch_size:]
            list_minibatchs.append({'x': x_train.astype(theano.config.floatX),
                                    'y': y_train.astype(theano.config.floatX)})

    return list_minibatchs


def grab_train_data(x, y, nbr_sup, nbr_xx, nbr_yy):
    """Create a mixed train dataset.
    Input:
        x: numpy.ndarray. X matrix (all data)
        y: numpy.ndarray. Y matrix (all data)
        nbr_sup: int. The number of the supervised examples.
        nbr_xx: int. The number of the unsupervised x.
        nbr_yy: int. The number of the y without x.
    """
    assert nbr_sup + nbr_xx + nbr_yy <= x.shape[0]
    xtr = x[:nbr_sup]
    ytr = y[:nbr_sup]
    xxtr = x[nbr_sup:nbr_sup+nbr_xx]
    yytr = y[nbr_sup+nbr_xx: nbr_sup+nbr_xx+nbr_yy]
    # Mix them randomly.
    # Combine
    index_sup = range(nbr_sup)
    index_xx = range(nbr_sup, nbr_sup + nbr_xx)
    index_yy = range(nbr_sup + nbr_xx, nbr_sup + nbr_xx + nbr_yy)
    index = index_sup + index_xx + index_yy
    index_arr = np.array(index)
    for i in range(10000):
        np.random.shuffle(index_arr)
    if len(x.shape) == 2:
        mega_x = np.empty((len(index), x.shape[1]), dtype=np.float32)
    else:
        mega_x = np.empty((len(index), x.shape[1], x.shape[2], x.shape[3]),
                          dtype=np.float32)
    mega_y = np.empty((len(index), y.shape[1]), dtype=np.float32)
    for i, j in zip(index_arr, xrange(len(index))):
        if i in index_sup:
            mega_x[j] = xtr[i]
            mega_y[j] = ytr[i]
        elif i in index_xx:
            mega_x[j] = xxtr[i-nbr_sup]
            mega_y[j] = np.float32(np.nan)
        elif i in index_yy:
            mega_x[j] = np.float32(np.nan)
            mega_y[j] = yytr[i-nbr_sup-nbr_xx]
        else:
            raise ValueError("Something wrong.")

    return mega_x, mega_y


def reshape_data(y, s):
    """Reshape the output (debug).
    Input:
        y: numpy.ndarray. Matrix: row are samples.
        s: int. The new shape of the output samples (squared).
    """
    out = np.empty((y.shape[0], s*s), dtype=y.dtype)
    for i in xrange(y.shape[0]):
        im = imresize(y[i, :].reshape(28, 28), (s, s), 'bilinear').flatten()
        out[i, :] = im

    return out


def plot_x_y_yhat(x, y, y_hat, xsz, ysz, binz=False):
    """Plot x, y and y_hat side by side."""
    plt.close("all")
    f = plt.figure(figsize=(15, 10.8), dpi=300)
    gs = gridspec.GridSpec(1, 3)
    if binz:
        y_hat = (y_hat > 0.5) * 1.
    ims = [x, y, y_hat]
    tils = [
        "x:" + str(xsz) + "x" + str(xsz),
        "y:" + str(ysz) + "x" + str(ysz),
        "yhat:" + str(ysz) + "x" + str(ysz)]
    for n, ti in zip([0, 1, 2], tils):
        f.add_subplot(gs[n])
        if n == 0:
            plt.imshow(ims[n], cmap=cm.Greys_r)
        else:
            plt.imshow(ims[n], cmap=cm.Greys_r)
        plt.title(ti)

    return f


def plot_x_x_yhat(x, x_hat):
    """Plot x, y and y_hat side by side."""
    plt.close("all")
    f = plt.figure()  # figsize=(15, 10.8), dpi=300
    gs = gridspec.GridSpec(1, 2)
    ims = [x, x_hat]
    tils = [
        "xin:" + str(x.shape[0]) + "x" + str(x.shape[1]),
        "xout:" + str(x.shape[1]) + "x" + str(x_hat.shape[1])]
    for n, ti in zip([0, 1], tils):
        f.add_subplot(gs[n])
        plt.imshow(ims[n], cmap=cm.Greys_r)
        plt.title(ti)
        ax = f.gca()
        ax.set_axis_off()

    return f


def plot_all_x_xhat(X, Xhat, nbr, path):
    if nbr > X.shape[0]:
        nbr = X.shape[0]
    for i in range(nbr):
        x, x_hat = X[i], Xhat[i]
        x = x.reshape((x.shape[1], x.shape[2]))
        x_hat = x_hat.reshape((x_hat.shape[1], x_hat.shape[2]))
        f = plot_x_x_yhat(x, x_hat)
        f.savefig(path + "/" + str(i) + ".png", bbox_inches='tight')


def plot_all_x_y_yhat(x, y, yhat, xsz, ysz, fd, binz=False):
    for i in xrange(x.shape[0]):
        if len(x[i].shape) <= 2:
            imx = x[i, :].reshape(xsz, xsz)
        else:
            imx = x[i].transpose((1, 2, 0))
        imy = y[i, :].reshape(ysz, ysz)
        imyhat = yhat[i, :].reshape(ysz, ysz)
        fig = plot_x_y_yhat(imx, imy, imyhat, xsz, ysz, binz)
        fig.savefig(fd+"/"+str(i)+".png", bbox_inches='tight')
        del fig


if __name__ == "__main__":
    from keras import regularizers
    W_regularizer = regularizers.l2(1e-3)
    activity_regularizer = regularizers.activity_l2(0.0001)
    cnn = FeatureExtractorCNN("alexnet",
                              weights_path="../inout/weights/alexnet_weights.h5",
                              trainable=True, W_regularizer=W_regularizer,
                              activity_regularizer=activity_regularizer,
                              trained=False, dense=False,
                              just_the_features=True)