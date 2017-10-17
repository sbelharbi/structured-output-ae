from theano import tensor as T
import theano
import numpy
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import conv
from theano.tensor.nnet import abstract_conv

from sop_embed.layer import HiddenLayer
from sop_embed.extra import get_non_linearity_fn


def relu(x):
    return T.switch(x > 0, x, 0)


def sharedX_value(value, name=None, borrow=None, dtype=None):
    """Share a single value after transforming it to floatX type.
    value: a value
    name: variable name (str)
    borrow: boolean
    dtype: the type of the value when shared. default: theano.config.floatX
    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(
        theano._asarray(value, dtype=dtype), name=name, borrow=borrow)


#class HiddenLayer(object):
#    def __init__(self, rng, input, n_in, n_out, W=None, b=None, b_v=0.,
#                 activation=None):
#        """
#        Intput:
#            is_train: theano.iscalar.
#            b_v: float. The initial value of the bias.
#        """
#        self.input = input
#        if W is None:
#            W_values = numpy.asarray(
#                rng.uniform(
#                    low=-numpy.sqrt(6. / (n_in + n_out)),
#                    high=numpy.sqrt(6. / (n_in + n_out)),
#                    size=(n_in, n_out)
#                ),
#                dtype=theano.config.floatX
#            )
#            if activation == theano.tensor.nnet.sigmoid:
#                W_values *= 4
#
#            W = theano.shared(value=W_values, name='W', borrow=True)
#
#        if b is None:
#            b_values = (
#                numpy.ones((n_out,)) * b_v).astype(theano.config.floatX)
#            b = theano.shared(value=b_values, name='b', borrow=True)
#
#        self.W = W
#        self.b = b
#
#        lin_output = T.dot(input, self.W) + self.b
#
#        self.output = (
#            lin_output if activation is None
#            else activation(lin_output)
#        )
#
#        self.params = [self.W, self.b]


class IdentityHiddenLayer(object):
    """
    This is the identity layer. It takes the input and give it back as output.
    We will be using this layer just after the last convolution layer to applay
    a dropout.
    """
    def __init__(self, rng, input):
        self.input = input
        self.W = None
        self.b = None
        self.params = []
        self.output = input


def dropout_from_layer(rng, layer_output, p):
    """
    p: float. The probablity of dropping a unit.
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(99999))
    one = T.constant(1)
    retain_prob = one - p
    mask = srng.binomial(n=1, p=retain_prob, size=layer_output.shape,
                         dtype=layer_output.dtype)
    output = layer_output * mask

    return output


def localResponseNormalizationCrossChannel(incoming, alpha=1e-4,
                                           k=2, beta=0.75, n=5):
    """
    Implement the local response normalization cross the channels described
    in <ImageNet Classification with Deep Convolutional Neural Networks>,
    A.Krizhevsky et al. sec.3.3.
    Reference of the code:
    https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/
    normalization.py
    https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py
    Parameters:
    incomping: The feature maps. (output of the convolution layer).
    alpha: float scalar
    k: float scalr
    beta: float scalar
    n: integer: number of adjacent channels. Must be odd.
    """
    if n % 2 == 0:
        raise NotImplementedError("Works only with odd n")

    input_shape = incoming.shape
    half_n = n // 2
    input_sqr = T.sqr(incoming)
    b, ch, r, c = input_shape
    extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
    input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                input_sqr)
    scale = k
    for i in range(n):
        scale += alpha * input_sqr[:, i:i+ch, :, :]
    scale = scale ** beta

    return incoming / scale


class LRNCCIdentityLayer(IdentityHiddenLayer):
    def __init__(self, input, alpha=1e-4, k=2, beta=0.75, n=5):
        super(LRNCCIdentityLayer, self).__init__(rng=None, input=input)
        self.output = localResponseNormalizationCrossChannel(
            incoming=self.output, alpha=alpha, k=k, beta=beta, n=n)


class DropoutIdentityHiddenLayer(IdentityHiddenLayer):
    def __init__(self, rng, input, dropout_rate, rescale):
        """
        rescale: Boolean. Can be only used when applying dropout.
        """
        if rescale:
            one = T.constant(1)
            retain_prob = one - dropout_rate
            input /= retain_prob

        super(DropoutIdentityHiddenLayer, self).__init__(rng=rng, input=input)
        if dropout_rate > 0.:
            self.output = dropout_from_layer(rng, self.output, p=dropout_rate)


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, dropout_rate, rescale,
                 W=None, b=None, b_v=0., activation=None):
        """
        rescale: Boolean. Can be only used when applying dropout.
        """
        if rescale:
            one = T.constant(1)
            retain_prob = one - dropout_rate
            input /= retain_prob

        super(DropoutHiddenLayer, self).__init__(
            input=input, n_in=n_in, n_out=n_out, W=W, b=b,
            activation=activation, rng=rng)
        if dropout_rate > 0.:
            self.output = dropout_from_layer(rng, self.output, p=dropout_rate)


class ConvLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, activation,
                 padding, W=None, b=None, b_v=0., stride=(1, 1)):
        """Implement a convolution layer. No pooling."""
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.x = input
        print filter_shape, "***********"
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if rng is None:
            rng = numpy.random.RandomState(23455)
        if W is None:
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                name="w_conv",
                borrow=True
            )
        if b is None:
            b_v = (
                numpy.ones(
                    (filter_shape[0],)) * b_v).astype(theano.config.floatX)
            b = theano.shared(value=b_v, name="b_conv", borrow=True)

        self.W = W
        self.b = b
        conv_out = conv2d(
            input=self.x,
            filters=self.W,
            input_shape=image_shape,
            filter_shape=filter_shape,
            border_mode=padding,
            subsample=stride
        )
        linear = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        if activation is not None:
            self.output = activation(linear)
        else:
            self.output = linear
        self.params = [self.W, self.b]


class PoolingLayer(object):
    def __init__(self, input, poolsize, ignore_border, mode="max"):
        """Implement a pooling layer."""
        self.input = input
        self.x = input
        self.output = pool.pool_2d(
            input=self.x,
            ws=poolsize,
            ignore_border=ignore_border,
            mode=mode)
        self.params = []


class UnPoolingLayer(object):
    def __init__(self, input, ratio, use_1D_kernel=False):
        self.input = input
        self.x = input
        self.output = abstract_conv.bilinear_upsampling(
            input=self.x,
            ratio=ratio,
            use_1D_kernel=use_1D_kernel)
        self.params = []


class DeepConvolutionLayer(object):
    """A deep convolution network: convolution, pooling, convolution, pooling,
    ... . Can use it to build Transposed convolution by using unpooling
    instead of pooling."""
    def __init__(self, input, layers, crop_size):
        """crope_size = [Din, hight_img, width_img]"""
        self.input = input
        self.x = input
        self.input_features_dim = crop_size
        self.layers = []
        self.params = []
        tmp_in = self.x
        nbr_f_in, h, w = crop_size[0], crop_size[1], crop_size[2]
        image_shape = (None, crop_size[0], h, w)

        for i in range(len(layers)):
            layer = layers[i]
            if layer["type"] == "conv":
                print "output filter shape", layer["filter_shape"][0]
                self.layers.append(ConvLayer(
                    rng=layer["rng"],
                    input=tmp_in,
                    filter_shape=(layer["filter_shape"][0],
                                  nbr_f_in,
                                  layer["filter_shape"][1],
                                  layer["filter_shape"][2]),
                    image_shape=image_shape,
                    activation=get_non_linearity_fn(layer["activation"]),
                    padding=layer["padding"],
                    W=layer["W"],
                    b=layer["b"],
                    b_v=layer["b_v"],
                    stride=layer["stride"]))
                self.params.extend(self.layers[-1].params)
                tmp_in = self.layers[-1].output
                nbr_f_in = layer["filter_shape"][0]

                h = (h-layer["filter_shape"][1] + 2 * layer["padding"][0] + 1)
                w = (w-layer["filter_shape"][2] + 2 * layer["padding"][1] + 1)
#                if i < (len(layers) - 1):
#                    assert layers[i+1]["type"] in ["upsample", "downsample"]
            if layer["type"] == "downsample":
                self.layers.append(PoolingLayer(
                    input=tmp_in,
                    poolsize=layer["poolsize"],
                    ignore_border=layer["ignore_border"],
                    mode=layer["mode"]))
                self.params.extend(self.layers[-1].params)
                tmp_in = self.layers[-1].output
                # assert layers[i-1]["type"] in ["conv"]
                h = h / layer["poolsize"][0]
                w = w / layer["poolsize"][1]
            if layer["type"] == "upsample":
                self.layers.append(UnPoolingLayer(
                    input=tmp_in,
                    ratio=layer["ratio"],
                    use_1D_kernel=layer["use_1D_kernel"]))
                self.params.extend(self.layers[-1].params)
                tmp_in = self.layers[-1].output
                h = h * layer["ratio"]
                w = w * layer["ratio"]
            image_shape = (None, nbr_f_in, h, w)
        # End building.
        self.output_features_dim = [nbr_f_in, h, w]
        self.output = self.layers[-1].output
        self.output_size_flatten = nbr_f_in * h * w


class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape,
                 poolsize=(2, 2), maxout=False, poolmaxoutfactor=2,
                 W=None, b=None, b_v=0., stride=(1, 1), LRN={
                     "app": False, "before": False, "alpha": 1e-4, "k": 2,
                     "beta": 0.75, "n": 5}):
        """
        Input:
            maxout: Boolean. Indicates if to do or not a maxout.
            poolmaxoutfactor: How many feature maps to maxout. The number of
                input feature maps must be a multiple of poolmaxoutfactor.
            allow_dropout_conv: Boolean. Allow or not the dropout in conv.
                layer. This maybe helpful when we want to use dropout only
                for fully connected layers.
            LRN: tuple (a, b) of booleans. a: apply or not the local response
                normalization. b: before (True) or after (False) the pooling.
            b_v: float. The initial value of the bias.
        """
        self.LRNCCIdentityLayer = None
        if maxout:
            assert poolmaxoutfactor == 2
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if W is None:
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                name="w_conv",
                borrow=True
            )
        if b is None:
            b_v = (
                numpy.ones(
                    (filter_shape[0],)) * b_v).astype(theano.config.floatX)
            b = theano.shared(value=b_v, name="b_conv", borrow=True)

        self.W = W
        self.b = b
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=stride
        )
        # Local reponse normalization
        if LRN["app"] and LRN["before"]:
            self.LRNCCIdentityLayer = LRNCCIdentityLayer(
                conv_out, alpha=LRN["alpha"], k=LRN["k"], beta=LRN["beta"],
                n=LRN["n"])
            conv_out = self.LRNCCIdentityLayer.output
            print "LRN BEFORE pooling ..."

        if maxout:
            z = T.add(conv_out, self.b.dimshuffle('x', 0, 'x', 'x'))
            s = None
            for i in range(filter_shape[0]/poolmaxoutfactor):
                t = z[:, i::poolmaxoutfactor, :, :]
                if s is None:
                    s = t
                else:
                    s = T.maximum(s, t)
            z = s
            if poolsize not in [None, (1, 1)]:
                pooled_out = pool.pool_2d(
                    input=z,
                    ds=poolsize,
                    ignore_border=True
                )
                self.output = pooled_out
            else:
                self.output = z
        else:
            if poolsize not in [None, (1, 1)]:
                pooled_out = pool.pool_2d(
                    input=conv_out,
                    ds=poolsize,
                    ignore_border=True
                    )
                self.output = relu(
                    pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
                print "RELU..."
            else:
                # simple relu
                term = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
                self.output = T.switch(term > 0, term, 0 * term)
                print "RELU..."

        # Local reponse normalization
        if LRN["app"] and not LRN["before"]:
            self.LRNCCIdentityLayer = LRNCCIdentityLayer(
                self.output, alpha=LRN["alpha"], k=LRN["k"], beta=LRN["beta"],
                n=LRN["n"])
            self.output = self.LRNCCIdentityLayer.output
            print "LRN AFTER activation(of pooling)..."

        self.params = [self.W, self.b]


class DropoutLeNetConvPoolLayer(LeNetConvPoolLayer):
    def __init__(self, rng, input, filter_shape, image_shape, dropout_rate,
                 rescale, poolsize=(2, 2), stride=(1, 1),
                 LRN={
                     "app": False, "before": False, "alpha": 1e-4, "k": 2,
                     "beta": 0.75, "n": 5},
                 maxout=False, poolmaxoutfactor=2, W=None, b=None, b_v=0.):
        if rescale:
            one = T.constant(1)
            retain_prob = one - dropout_rate
            input /= retain_prob
        super(DropoutLeNetConvPoolLayer, self).__init__(
            rng=rng, input=input, filter_shape=filter_shape,
            image_shape=image_shape, poolsize=poolsize, stride=stride,
            LRN=LRN, maxout=maxout, poolmaxoutfactor=poolmaxoutfactor,
            W=W, b=b, b_v=b_v)
        if dropout_rate > 0.:
            self.output = dropout_from_layer(rng, self.output, p=dropout_rate)
