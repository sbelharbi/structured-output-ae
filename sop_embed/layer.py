# -*- coding: utf-8 -*-

#    Copyright (c) 2016 Soufiane Belharbi, Clément Chatelain,
#    Romain Hérault, Sébastien Adam (LITIS - EA 4108).
#    All rights reserved.
#
#   This file is part of structured-output-ae.
#
#    structured-output-ae is free software: you can redistribute it and/or
#    modify it under the terms of the Lesser GNU General Public License as
#    published by the Free Software Foundation, either version 3 of the
#    License, or (at your option) any later version.
#
#    structured-output-ae is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with structured-output-ae.
#    If not, see <http://www.gnu.org/licenses/>.


# Based on: https://github.com/caglar/autoencoders.git
# http://www-etud.iro.umontreal.ca/~gulcehrc/
# Modified by: Soufiane Belharbi


from __future__ import division
import numpy as np
import theano
from theano import tensor as T


class Layer(object):
    """
    A general base layer class for neural network.
    """
    def __init__(self,
                 input,
                 n_in,
                 n_out,
                 activation=T.nnet.sigmoid,
                 sparse_initialize=False,
                 num_pieces=1,
                 non_zero_units=25,
                 rng=None):

        self.num_pieces = num_pieces
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.sparse_initialize = sparse_initialize
        self.non_zero_units = non_zero_units
        self.W = None
        self.b = None
        self.activation = activation

    def reset_layer(self):
        """
        initailize the layer's parameters to random.
        """
        if self.W is None:
            if self.sparse_initialize:
                W_values = self.sparse_initialize_weights()
            else:
                if self.activation == theano.tensor.tanh:
                    born = np.sqrt(6. / (self.n_in + self.n_out))
                else:
                    born = 4 * np.sqrt(6. / (self.n_in + self.n_out))
                W_values = np.asarray(self.rng.uniform(
                    low=-born,
                    high=born,
                    size=(self.n_in, self.n_out)),
                    dtype=theano.config.floatX)

            self.W = theano.shared(value=W_values, name='W', borrow=True)

        if self.b is None:
            b_values = np.zeros((self.n_out/self.num_pieces),
                                dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)

        # The layer parameters
        self.params = [self.W, self.b]

    def sparse_initialization_weights(self):
        """
        Implement the sparse initialization technique as described in
        J. Marten, 'Deep learning via Hessian-free optimization', ICML, 2010.
        http://icml2010.haifa.il.ibm.com/papers/458.pdf
        """
        W = []
        mu, sigma = 0, 1/self.non_zero_units

        for i in xrange(self.n_in):
            row = np.zeros(self.n_out)
            non_zeros = self.rng.normal(mu, sigma, self.non_zero_units)
            # non_zeros /= non_zeros.sum()
            non_zero_idxs = self.rng.permutation(
                self.n_out)[0:self.non_zero_units]
            for j in xrange(self.non_zero_units):
                row[non_zero_idxs[j]] = non_zeros[j]
            W.append(row)
        W = np.asarray(W, dtype=theano.config.floatX)
        return W


class HiddenLayer(Layer):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, rng=None):
        """
        Typical hidden layer of an MLP: units are fully connected and have
        tangente hyperbolic activation function. Weight matrix (W) is of shape
        (n_in, n_out) and the bias vector (b) is of shape (nout,).

        Hidden unit activation is given by: tanh(dot(input, w)+ b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initiaze the weights.

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimension of the input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation:  Non linearity to be applied in the hidden layer.
        """
        if rng is None:
            rng = np.random.RandomState()

        super(HiddenLayer, self).__init__(
            input, n_in, n_out, activation=activation, rng=rng)
        self.reset_layer()

        if W is not None:
            self.W = W

        if b is not None:
            self.b = b

        self.params = [self.W, self.b]
        self.setup_outputs(input)

    def setup_outputs(self, input):
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output))

    def get_outputs(self, input):
        self.setup_outputs(input)
        return self.output


class AEHiddenLayer(Layer):
    def __init__(self,
                 input,
                 n_in,
                 n_out,
                 n_in_dec=None,
                 n_out_dec=None,
                 W=None,
                 b=None,
                 num_pieces=1,
                 bhid=None,
                 activation=T.nnet.sigmoid,
                 sparse_initialize=False,
                 tied_weights=True,
                 rng=None):
        """
        Typical hidden layer for an auto-encoder: The units are fully connected
        and have sigmoidal activation function. Weight matrix (W) is of shape
        (n_in, n_out) and the bias vector (b) is of shape(n_out,).

        Hidden units activation is given by: sigmoid(dot(input, w)+ b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initiaze the weights.

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimension of the input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation:  Non linearity to be applied in the hidden layer.
        """
        if rng is None:
            rng = np.random.RandomState()

        super(AEHiddenLayer, self).__init__(
            input=input,
            n_in=n_in,
            n_out=n_out,
            num_pieces=num_pieces,
            activation=activation,
            sparse_initialize=sparse_initialize,
            rng=rng)

        self.reset_layer()

        if W is not None:
            self.W = W

        if b is not None:
            self.b = b

        if bhid is not None:
            self.b_prime = bhid
        else:
            if n_in_dec is not None:
                b_values = np.zeros((n_out_dec), dtype=theano.config.floatX)
            else:
                b_values = np.zeros(
                    (self.n_in/num_pieces), dtype=theano.config.floatX)

            self.b_prime = theano.shared(value=b_values, name="b_prime")

        if tied_weights:
            self.W_prime = self.W.T
        else:
            if n_in_dec is not None and n_out_dec is not None:
                W_values = np.asarray(
                    self.rng.normal(loc=0.,
                                    scale=0.005,
                                    size=(n_out_dec, n_in_dec)),
                    dtype=theano.config.floatX)
            else:
                if self.activation == theano.tensor.tanh:
                    born = np.sqrt(6. / (self.n_in + self.n_out))
                else:
                    born = 4 * np.sqrt(6. / (self.n_in + self.n_out))
                W_values = np.asarray(
                    self.rng.uniform(
                        low=-born,
                        high=born,
                        size=(self.n_out, self.n_in)),
                    dtype=theano.config.floatX)

            self.W_prime = theano.shared(value=W_values, name='W_prime',
                                         borrow=True)
            self.params += [self.W_prime]

        self.params += [self.b_prime]
        self.setup_outputs(input)

    def setup_outputs(self, input):
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output))

    def get_outputs(self, input):
        self.setup_outputs(input)
        return self.output


class AEOutputLayer(AEHiddenLayer):
    def __init__(self,
                 input,
                 n_in,
                 n_out,
                 n_in_dec=None,
                 n_out_dec=None,
                 W=None,
                 b=None,
                 num_pieces=1,
                 bhid=None,
                 activation=T.nnet.sigmoid,
                 sparse_initialize=False,
                 tied_weights=True,
                 rng=None):
        """
        Typical hidden layer for an auto-encoder: The units are fully connected
        and have sigmoidal activation function. Weight matrix (W) is of shape
        (n_in, n_out) and the bias vector (b) is of shape(n_out,).
        The only difference between this AE and the AEHiddeLayer is:
        W_prime = W (real values)
        W = W_prime.T (tensor)

        Hidden units activation is given by: sigmoid(dot(input, w)+ b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initiaze the weights.

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimension of the input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation:  Non linearity to be applied in the hidden layer.
        """
        if rng is None:
            rng = np.random.RandomState()

        super(AEOutputLayer, self).__init__(
            input=input,
            n_in=n_in,
            n_out=n_out,
            n_in_dec=n_in_dec,
            n_out_dec=n_out_dec,
            W=W,
            b=b,
            num_pieces=num_pieces,
            bhid=bhid,
            activation=activation,
            sparse_initialize=sparse_initialize,
            tied_weights=tied_weights,
            rng=rng)
        # Reverse the weights
        if tied_weights:
            W_val = self.W.get_value()
            self.W_prime = theano.shared(value=np.transpose(W_val),
                                         name='W_prime', borrow=True)
            self.W = self.W_prime.T
            self.params = [self.W_prime, self.b, self.b_prime]


class LogisticRegressionLayer(Layer):
    """
    Multi-class logistic regression layer.
    The logistic regression is fully described by a weight matrix ::math:`W`
    and a bias vector ::math: `b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probablity.
    """
    def __init__(self, input, n_in, n_out, is_binary=False, threshold=0.4,
                 rng=None):
        """
        Initialize the parameters of the logistic regression.
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which
        the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie (number of classes)
        """
        self.activation = T.nnet.sigmoid
        self.threshold = threshold
        super(LogisticRegressionLayer, self).__init__(
            input,
            n_in,
            n_out,
            self.activation,
            rng)

        self.reset_layer()

        self.is_binary = is_binary
        if n_out == 1:
            self.is_binary = True
        # The number of classes
        self.n_classes_seen = np.zeros(n_out)
        # The number of the wrong classification madefor the class i
        self.n_wrong_classif_made = np.zeros(n_out)

        self.reset_conf_mat()

        # Compute vector class-membership probablities in symbolic form
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+ self.b)
        self.p_y_given_x = self.get_class_memberships(self.input)

        if not self.is_binary:
            # Compute prediction as class whose probability is maximal
            # in symbolic form
            self.y_decision = T.argmax(self.p_y_given_x, axis=1)
        else:
            # If the probability is greater than the specified threshold
            # assign to the class 1, otherwise it is 0. Which alos can be
            # checked if p(y=1|x) > threshold.
            self.y_decision = T.gt(T.flatten(self.p_y_given_x), self.threshold)

        self.params = [self.W, self.b]

    def reset_conf_mat(self):
        """
        Reset the confusion matrix.
        """
        self.conf_mat = np.zeros(shape=(self.n_out, self.n_out),
                                 dtype=np.dtype(int))

    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
            \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                    \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
            the correct label.
        Note: We use the mean instead of the sum so that the learning rate
            is less dependent of the batch size.
        """
        if self.is_binary:
            return -T.mean(T.log(self.p_y_given_x))
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def crossentropy_categorical(self, y):
        """
        Find the categorical cross entropy.
        """
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))

    def crossentropy(self, y):
        """
        use the theano nnet cross entropy function. Return the mean.
        Note: self.p_y_given_x is (batch_size, 1) but y is (batch_size,).
        In order to establish the compliance, we should flatten the
        p_y_given_x.
        """
        return T.mean(
            T.nnet.binary_crossentropy(T.flatten(self.p_y_given_x), y))

    def get_class_memberships(self, x):
        lin_activation = T.dot(x, self.W) + self.b
        if self.is_binary:
            # return the sigmoid value
            return T.nnet.sigmoid(lin_activation)
        # else retunr the softmax
        return T.nnet.softmax(lin_activation)

    def update_conf_mat(self, y, p_y_given_x):
        """
        Update the confusion matrix with the given true labels and estimated
        labels.
        """
        if self.n_out == 1:
            y_decision = (p_y_given_x > self.threshold)
        else:
            y_decision = np.argmax(p_y_given_x, axis=1)
        for i in xrange(y.shape[0]):
            self.conf_mat[y[i]][y_decision[i]] += 1

    def errors(self, y):
        """
        returns a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch. Zero one loss
        over the size of the minibatch.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
        correct label.
        """
        if y.ndim != self.y_decision.ndim:
            raise TypeError("y should have the same shape as self.y_decision",
                            ('y', y.type, "y_decision", self.y_decision.type))
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # The T.neq operator returns a vector of 0s and 1s, where:
            # 1 represents a mistake in classification
            return T.mean(T.neq(self.y_decision, y))
        else:
            raise NotImplementedError()

    def raw_prediction_errors(self, y):
        """
        Returns a binary array where each each element indicates if the
        corresponding sample has been correctly classified (0) or not (1) in
        the minibatch.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
        correct label.
        """
        if y.ndim != self.y_decision.ndim:
            raise TypeError("y should have the same shape as self.y_decision",
                            ('y', y.type, "y_decision", self.y_decision.type))
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # The T.neq operator returns a vector of 0s and 1s, where:
            # 1 represents a mistake in classification
            return T.neq(self.y_decision, y)
        else:
            raise NotImplementedError()

    def error_per_calss(self, y):
        """
        Return an array where each value is the error for the corresponding
        classe in the minibatch.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
        correct label.
        """
        if y.ndim != self.y_decision.ndim:
            raise TypeError("y should have the same shape as self.y_decision",
                            ('y', y.type, "y_decision", self.y_decision.type))
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            y_decision_res = T.neq(self.y_decision, y)
            for (i, y_decision_r) in enumerate(y_decision_res):
                self.n_classes_seen[y[i]] += 1
                if y_decision_r:
                    self.n_wrong_classif_made[y[i]] += 1
            pred_per_class = self.n_wrong_classif_made / self.n_classes_seen
            return T.mean(y_decision_res), pred_per_class
        else:
            raise NotImplementedError()
