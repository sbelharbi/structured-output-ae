# Based on: https://github.com/caglar/autoencoders.git
# http://www-etud.iro.umontreal.ca/~gulcehrc/
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

from ae import Autoencoder, CostType


class SparseAutoencoder(Autoencoder):
    """
    Typical class for the sparse encoder.
    """
    def __init__(self,
                 input,
                 nvis,
                 nhid,
                 rnd=None,
                 theano_rng=None,
                 bhid=None,
                 cost_type=CostType.CrossEntropy,
                 bvis=None):
        super(SparseAutoencoder, self).__init__(
            input=input,
            nvis=nvis,
            nhid=nhid,
            rnd=rnd,
            bhid=bhid,
            cost_type=cost_type,
            bvis=bvis)
        if not theano_rng:
            theano_rng = RandomStreams(rnd.randint(2 ** 30))
        self.theano_rng = theano_rng

    def kl_divergence(self, p, p_hat):
        term1 = p * T.log(p)
        term2 = p * T.log(p_hat)
        term3 = (1-p) * T.log(1-p)
        term4 = (1-p) * T.log(1-p_hat)
        return term1 - term2 + term3 - term4

    def sparsity_penality(self, h, sparsity_level=0.05, sparsity_reg=1e-3,
                          batch_size=-1):
        if batch_size == -1 or batch_size == 0:
            raise Exception("Invalid batch size.")
        sparsity_level = T.extra_ops.repeat(sparsity_level, self.nhid)
        sparsity_penality = 0
        avg_act = h.mean(axis=0)
        kl_div = self.kl_divergence(sparsity_level, avg_act)
        sparsity_penality = sparsity_reg * kl_div.sum()
        return sparsity_penality

    def get_sa_sgd_updates(self, learning_rate, sparsity_level, sparse_reg,
                           batch_size, x_in=None):
        h = self.encode(x_in)
        x_rec = self.decode(h)
        cost = self.get_rec_cost(x_rec)
        sparsity_penal = self.sparsity_penality(h, sparsity_level, sparse_reg,
                                                batch_size)
        cost = cost + sparsity_penal

        gparams = T.grad(cost, self.params)
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - learning_rate * gparam
        return (cost, updates)

    def fit(self,
            data=None,
            learning_rate=0.08,
            batch_size=100,
            n_epochs=22,
            sparsity_penality=0.001,
            sparsity_level=0.05,
            weights_file="out/sa_weights_mnists.npy"):
        if data is None:
            raise Exception("Data can not be empty.")

        index = T.lscalar("index")
        data_shared = theano.shared(np.asarray(data.tolist(),
                                               dtype=theano.config.floatX))

        n_batches = data.shape[0] / batch_size
        (cost, updates) = self.get_sa_sgd_updates(
            learning_rate, sparsity_level, sparsity_penality, batch_size)

        train_ae = theano.function(
            [index], updates, givens={
                self.x: data_shared[index*batch_size: (index+1)*batch_size]})

        print "Started training SA."
        ae_costs = []
        for epoch in xrange(n_epochs):
            print "Training at epoch %d" % epoch
            cost_one_epoch = []
            for batch_index in xrange(n_batches):
                cost_one_epoch.append(train_ae(batch_index))
            print "Training at epoch %d, %f" % (epoch, np.mean(cost_one_epoch))
            ae_costs.append(np.mean(cost_one_epoch))

        print "Saving files ..."
        np.save(weights_file, self.params[0].get_value())
        return ae_costs
