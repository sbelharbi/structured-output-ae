import theano.tensor as T
import theano


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


class CostType:
    MeanSquared = "MeanSquaredCost"
    CrossEntropy = "CrossEntropy"


class NonLinearity:
    RELU = "rectifier"
    TANH = "tanh"
    SIGMOID = "sigmoid"


def get_non_linearity_fn(nonlinearity):
        if nonlinearity == NonLinearity.SIGMOID:
            return T.nnet.sigmoid
        elif nonlinearity == NonLinearity.RELU:
            return relu
        elif nonlinearity == NonLinearity.TANH:
            return T.tanh
        elif nonlinearity is None:
            return None
