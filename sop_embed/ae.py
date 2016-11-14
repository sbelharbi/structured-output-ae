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


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from layer import AEHiddenLayer, AEOutputLayer
from tools import NonLinearity, CostType, relu
import numpy as np
import cPickle as pkl

from collections import OrderedDict

theano.config.warn.subtensor_merge_bug = False


class Autoencoder(object):
    """
    Typical implementation of an autoencoder.
    """
    def __init__(
            self,
            input,
            nvis,
            nhid=None,
            nvis_dec=None,
            nhid_dec=None,
            rnd=None,
            bhid=None,
            cost_type=CostType.MeanSquared,
            momentum=1,
            num_pieces=1,
            L2_reg=-1,
            L1_reg=-1,
            sparse_initialize=False,
            nonlinearity=NonLinearity.TANH,
            bvis=None,
            tied_weights=True,
            reverse=False):

        self.input = input
        self.nvis = nvis
        self.nhid = nhid
        self.bhid = bhid
        self.bvis = bvis
        self.momentum = momentum
        self.nonlinearity = nonlinearity
        self.tied_weights = tied_weights
        self.gparams = None
        self.reverse = reverse
        self.activation = self.get_non_linearity_fn()
        self.catched_params = {}

        if cost_type == CostType.MeanSquared:
            self.cost_type = CostType.MeanSquared
        elif cost_type == CostType.CrossEntropy:
            self.cost_type = CostType.CrossEntropy

        if rnd is None:
            self.rnd = np.random.RandomState(1231)
        else:
            self.rnd = rnd

        self.srng = RandomStreams(seed=1231)

        if not reverse:
            self.hidden = AEHiddenLayer(input=input,
                                        n_in=nvis,
                                        n_out=nhid,
                                        num_pieces=num_pieces,
                                        n_in_dec=nvis_dec,
                                        n_out_dec=nhid_dec,
                                        activation=self.activation,
                                        tied_weights=tied_weights,
                                        sparse_initialize=sparse_initialize,
                                        rng=rnd)
        else:
            self.hidden = AEOutputLayer(input=input,
                                        n_in=nvis,
                                        n_out=nhid,
                                        num_pieces=num_pieces,
                                        n_in_dec=nvis_dec,
                                        n_out_dec=nhid_dec,
                                        activation=self.activation,
                                        tied_weights=tied_weights,
                                        sparse_initialize=sparse_initialize,
                                        rng=rnd)

        self.params = self.hidden.params

        self.sparse_initialize = sparse_initialize

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.L1 = 0
        self.L2 = 0

        if L1_reg != -1:
            if not reverse:
                self.L1 += abs(self.hidden.W).sum()
                if not tied_weights:
                    self.L1 += abs(self.hidden.W_prime).sum()
            else:
                self.L1 += abs(self.hidden.W_prime).sum()
                if not tied_weights:
                    self.L1 += abs(self.hidden.W).sum()

        if L2_reg != -1:
            if not reverse:
                self.L2 += (self.hidden.W_prime**2).sum()
                if not tied_weights:
                    self.L2 += (self.hidden.W**2).sum()
            else:
                self.L2 += (self.hidden.W**2).sum()
                if not tied_weights:
                    self.L2 += (self.hidden.W_prime**2).sum()

        if input is not None:
            self.x = input
        else:
            self.x = T.matrix('x_input', dtype=theano.config.floatX)

    def catch_params(self):
        for param in self.params:
            self.catched_params[param.name] = param.get_value()

    def nonlinearity_fn(self, d_in=None, recons=False):
        if self.nonlinearity == NonLinearity.SIGMOID:
            return T.nnet.sigmoid(d_in)
        elif self.nonlinearity == NonLinearity.RELU and not recons:
            return T.maximum(d_in, 0)
        elif self.nonlinearity == NonLinearity.RELU and recons:
            return T.nnet.softplus(d_in)
        elif self.nonlinearity == NonLinearity.TANH:
            return T.tanh(d_in)

    def get_non_linearity_fn(self):
        if self.nonlinearity == NonLinearity.SIGMOID:
            return T.nnet.sigmoid
        elif self.nonlinearity == NonLinearity.RELU:
            return relu
        elif self.nonlinearity == NonLinearity.TANH:
            return T.tanh

    def encode(self, x_in=None, center=True):
        if x_in is None:
            x_in = self.x

        act = self.nonlinearity_fn(T.dot(x_in, self.hidden.W) + self.hidden.b)
        if center:
            act = act - act.mean(0)
        return act

    def encode_linear(self, x_in=None):
        if x_in is None:
            x_in = self.x_in

        lin_out = T.dot(x_in, self.hidden.W) + self.hidden.b
        return self.nonlinearity_fn(lin_out), lin_out

    def decode(self, h):
        return self.nonlinearity_fn(
            T.dot(h, self.hidden.W_prime) + self.hidden.b_prime)

    def get_rec_cost(self, x_rec, eyes=False):
        """
        Returns the reconstruction cost.
        """
        if self.cost_type == CostType.MeanSquared:
            return T.mean(((self.x - x_rec)**2).sum(axis=1))
        elif self.cost_type == CostType.CrossEntropy:
            return T.mean(
                (T.nnet.binary_crossentropy(x_rec, self.x)).mean(axis=1))

    def get_rec_cost_face(self, x_rec):
        """
        Returns the reconstruction cost.
        """
        d_eyes = (
            (self.x[:, 37] - self.x[:, 46])**2 +
            (self.x[:, 37] - self.x[:, 46])**2).T
        if self.cost_type == CostType.MeanSquared:
            return T.mean(((self.x - x_rec)**2).sum(axis=1) / d_eyes)
        elif self.cost_type == CostType.CrossEntropy:
            return T.mean(
                (T.nnet.binary_crossentropy(
                    x_rec, self.x)).mean(axis=1) / d_eyes)

    def kl_divergence(self, p, p_hat):
        return p * T.log(p) - T.log(p_hat) + (1-p) * T.log(1-p) -\
            (1-p_hat) * T.log(1-p_hat)

    def sparsity_penality(self, h, sparsity_level=0.05, sparse_reg=1e-3,
                          batch_size=-1):
        if batch_size == -1 or batch_size == 0:
            raise Exception("Invalid batch size")

        sparsity_level = T.extra_ops.repeat(sparsity_level, self.nhid)
        sparsity_penality = 0
        avg_act = h.mean(axis=0)
        kl_div = self.kl_divergence(sparsity_level, avg_act)
        sparsity_penality = sparse_reg * kl_div.sum()
        return sparsity_penality

    def act_grads(self, inputs):
        h, acts = self.encode_linear(inputs)
        h_grad = T.grad(h.sum(), acts)
        return (h, h_grad)

    def jacobian_h_x(self, inputs):
        h, act_grad = self.act_grads(inputs)
        jacobian = self.hidden.W * act_grad.dimshuffle(0, 'x', 1)
        return (h, T.reshape(jacobian, newshape=(self.nhid, self.nvis)))

    def compute_jacobian_h_x(self, inputs):
        inputs = theano.shared(inputs.flatten())
        h = self.encode(inputs)
        # see later
        # h = h.faltten()
        # inputs = inputs.flatten()
        # inputs = T.reshape(inputs, newshape=(self.nvis))
        J = theano.gradient.jacobian(h, inputs)
        return h, J

    def sample_one_step(self, x, sigma):
        # h, J_t = self.jacobian_h_x(x)
        h, J_t = self.compute_jacobian_h_x(x)
        eps = self.srng.normal(avg=0, size=(self.nhid, 1), std=sigma)
        jacob_w_eps = T.dot(J_t.T, eps)
        delta_h = T.dot(J_t, jacob_w_eps)
        perturbed_h = h + delta_h.T
        x = self.decode(perturbed_h)
        return x

    def sample_scan(self, x, sigma, n_steps, samples):
        # Enable on-the-fly graph computations
        #  theano.config.compute_test_value = "raise"
        in_val = T.fmatrix("input_values")
        # in_val.tag.test_value = np.asarray(
        #    np.random.rand(1, 784), dtype=theano.config.floatX)
        s_sigma = T.fscalr("sigma_values")
        # s_sigma = np.asarray(
        #    np.random.rand(1), dtype=theano.config.floatX)
        mode = "FAST_RUN"
        values, updates = theano.scan(fn=self.sample_one_step,
                                      outputs_info=in_val,
                                      non_sequences=s_sigma,
                                      n_steps=n_steps,
                                      mode=mode)
        ae_sampler = theano.function(inputs=[in_val, s_sigma],
                                     outputs=values[-1],
                                     updates=updates)
        samples = ae_sampler(x, sigma)
        return samples

    def sample_old(self, x, sigma, n_steps):
        # Enable on-the-fly graph computations
        # theano.config.compute_test_value = "raise"
        # in_val = T.fmatrix('input_values")
        # in_val.tag.test_value = np.asarray(
        #   np.random.rand(1, 784), dtype=theano.config.floatX)
        # s_sigma = T.fscalar("sigma_value")
        # s_sigma = np.asarray(
        #   np.random.rand(1), dtype=theano.config.floatX)
        # mode = "FAST_RUN"
        samples = []
        sample = x
        samples.append(x)
        for i in xrange(n_steps):
            print "Sample %d ..." % i
            sampler = self.sample_one_step(sample, sigma)
            sample = sampler.eval()
            samples.append(sample)
        return samples

    def get_sgd_updates(self, learning_rate, lr_scaler=1.0, batch_size=1,
                        sparsity_level=-1, sparse_reg=-1, x_in=None):
        h = self.encode(x_in)
        x_rec = self.decode(h)
        cost = self.get_rec_cost(x_rec)

        if self.L1_reg != -1 and self.L1_reg is not None:
            cost += self.L1_reg * self.L1

        if self.L2_reg != -1 and self.L2_reg is not None:
            cost += self.L2_reg * self.L2

        if sparsity_level != -1 and sparse_reg != -1:
            sparsity_penal = self.sparsity_penality(
                h, sparsity_level, sparse_reg, batch_size)
            cost += sparsity_penal

        self.gparams = T.grad(cost, self.params)
        updates = OrderedDict({})
        for param, gparam in zip(self.params, self.gparams):
            updates[param] = self.momentum * param - lr_scaler * \
                learning_rate * gparam
        return (cost, updates, h, x_rec)

    def get_train_cost(self, batch_size=1, sparsity_level=-1, sparse_reg=-1,
                       x_in=None, face=False):
        h = self.encode(x_in)
        x_rec = self.decode(h)
        if face:
            cost = self.get_rec_cost_face(x_rec)
        else:
            cost = self.get_rec_cost(x_rec)

        if self.L1_reg != -1 and self.L1_reg is not None:
            cost += self.L1_reg * self.L1

        if self.L2_reg != -1 and self.L2_reg is not None:
            cost += self.L2_reg * self.L2

        if sparsity_level != -1 and sparse_reg != -1:
            sparsity_penal = self.sparsity_penality(
                h, sparsity_level, sparse_reg, batch_size)
            cost += sparsity_penal

        return (cost, h, x_rec)

    def save_params(self, weights_file, catched=False):
        """Save the model's parameters."""
        f_dump = open(weights_file, "w")
        params_vls = {}
        if catched:
            if self.catched_params != {}:
                params_vls = self.catched_params
            else:
                raise ValueError(
                    "You asked to save catched params," +
                    "but you didn't catch any!!!!!!!")
        else:
            for param in self.params:
                params_vls[param.name] = param.get_value()
        pkl.dump(params_vls, f_dump, protocol=pkl.HIGHEST_PROTOCOL)
        f_dump.close()

    def set_params_vals(self, weights_file):
        """Set the values of the parameters."""
        with open(weights_file, 'r') as f:
            params_vls = pkl.load(f)
            for param in self.params:
                param.set_value(params_vls[param.name])

    def fit(self,
            data=None,
            learning_rate=0.1,
            batch_size=100,
            n_epochs=20,
            lr_scalar=0.998,
            weights_file="out/ae_weights_mnist.npy"):
        """
        Fit the data to the autoencoder (training).
        """
        if data is None:
            raise Exception("Data can't be empty.")

        index = T.lscalar("index")
        data_shared = theano.shared(
            np.asarray(data, dtype=theano.config.floatX))
        n_batches = data.shape[0] / batch_size
        (cost, updates) = self.get_sgd_updates(
            learning_rate, lr_scalar, batch_size)
        train_ae = theano.function(
            [index], cost, updates=updates,
            givens={
                self.x: data_shared[index*batch_size: (index+1)*batch_size]})

        print "Start training the ae."
        ae_costs = []

        for epoch in xrange(n_epochs):
            print "Training at epoch %d" % epoch
            cost_one_epoch = []
            for batch_index in xrange(n_batches):
                cost_one_epoch.append(train_ae(batch_index))
            print "Training at epoch %d, %f" % (epoch, np.mean(cost_one_epoch))
            ae_costs.append(np.mean(cost_one_epoch))

        print "Saving files ..."
        self.save_params(weights_file)
        return ae_costs
