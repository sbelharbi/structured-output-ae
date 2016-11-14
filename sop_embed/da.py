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
import numpy as np
import cPickle as pkl

from ae import Autoencoder, CostType, NonLinearity


class DenoisingAutoencoder(Autoencoder):
    """
    Basic class for the denoising autoencoder.
    """
    def __init__(self,
                 input,
                 nvis,
                 nhid,
                 rnd=None,
                 theano_rng=None,
                 bhid=None,
                 cost_type=CostType.MeanSquared,
                 momentum=1,
                 L1_reg=-1,
                 L2_reg=-1,
                 sparse_initialize=False,
                 nonlinearity=NonLinearity.TANH,
                 bvis=None,
                 tied_weights=True,
                 reverse=False,
                 corruption_level=0.):
        super(DenoisingAutoencoder, self).__init__(
            input=input,
            nvis=nvis,
            nhid=nhid,
            rnd=rnd,
            bhid=bhid,
            cost_type=cost_type,
            momentum=momentum,
            L1_reg=L1_reg,
            L2_reg=L2_reg,
            sparse_initialize=sparse_initialize,
            nonlinearity=nonlinearity,
            bvis=bvis,
            tied_weights=tied_weights,
            reverse=reverse)
        self.corruption_level = corruption_level

        if not theano_rng:
            theano_rng = RandomStreams(rnd.randint(2 ** 30))
        self.theano_rng = theano_rng

    # Overrite this function:
    def encode(self, x_in=None, center=True):
        print "Overriten ...."
        if x_in is None:
            if self.corruption_level != 0.:
                x_in = self.theano_rng.binomial(
                    self.x.shape, n=1, p=1-self.corruption_level,
                    dtype=theano.config.floatX) * self.x
            else:
                x_in = self.x

        act = self.nonlinearity_fn(T.dot(x_in, self.hidden.W) + self.hidden.b)
        if center:
            act = act - act.mean(0)
        return act

    def corrupt_input(self, in_data, corruption_level):
        return self.theano_rng.binomial(self.x.shape, n=1,
                                        p=1-corruption_level,
                                        dtype=theano.config.floatX) * self.x

    def get_reconstructed_images(self, data):
        h = self.encode(x_in=data)
        x_rec = self.decode(h)
        return x_rec

    def debug_grads(self, data):
        gfn = theano.function([self.x], self.gparams[0])
        print "gradients:"
        print gfn(data)
        print "params:"
        if not self.reverse:
            print self.hidden.W.get_value()
        else:
            print self.hidden.W_prime.get_value()

    def fit(self,
            data=None,
            learning_rate=0.1,
            learning_decay=None,
            batch_size=100,
            n_epochs=60,
            corruption_level=0.5,
            weights_file=None,
            sparsity_level=-1,
            sparse_reg=-1,
            shuffle_data=True,
            lr_scaler=1.0,
            recons_img_file="out/dae_reconstructed_pento.npy"):
        if data is None:
            raise Exception("Data can not be empty.")

        index = T.iscalar("index")
        data_shared = theano.shared(np.asarray(data.tolist(),
                                               dtype=theano.config.floatX))
        n_batches = data.shape[0] / batch_size

        corrupted_input = self.corrupt_input(data_shared, corruption_level)

        (cost, updates, h, x_rec) = self.get_sgd_updates(
            learning_rate, lr_scaler=lr_scaler, batch_size=batch_size,
            sparsity_level=sparsity_level, sparse_reg=sparse_reg,
            x_in=corrupted_input)

        train_ae = theano.function(
            [index], cost, updates=updates,
            givens={
                self.x: data_shared[index*batch_size: (index+1) * batch_size]})

        print "Start training DAE."
        ae_costs = []
        batch_index = 0
        for epoch in xrange(n_epochs):
            idxs = np.arange(n_batches)
            np.random.shuffle(idxs)
            print "Training at epoch %d" % epoch
            cost_one_epoch = []
            for batch_index in idxs:
                cost_one_epoch.append(train_ae(batch_index))
                if False:
                    print "Cost: ", ae_costs[-1]
                    self.debug_grads(
                        data_shared.get_value()
                        [batch_index * batch_size: (batch_index+1)*batch_size])
            print "Training at epohc %d, %f" % (epoch, np.mean(cost_one_epoch))
            ae_costs.append(np.mean(cost_one_epoch))

        if weights_file is not None:
            print "Saving the weights ..."
            self.save_params(weights_file)
        if recons_img_file is not None:
            print "Saving reconstructed images ..."
            x_rec = self.get_reconstructed_images(data_shared)
            np.save(recons_img_file, x_rec)
        return ae_costs
