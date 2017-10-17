# Based on: https://github.com/caglar/autoencoders.git
# http://www-etud.iro.umontreal.ca/~gulcehrc/
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import cPickle as pkl



from ae import Autoencoder, CostType, NonLinearity

from sop_embed.layers import DeepConvolutionLayer
from sop_embed.extra import sharedX_value


class ConvolutionalAutoencoder(object):
    """A demo of a DEEP convAE. It is two parts:
    Encoder: a deep convolutional network.
    Decoder: a deep transposed convolutional network."""
    def __init__(self, input, encoder_config, decoder_config, crop_size,
                 cost_type=CostType.MeanSquared, rnd=None,
                 corruption_level=0., l1_reg=0., l2_reg=0., reg_bias=False):
        """crope_size = [Din, hight_img, width_img]"""
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.catched_params = []
        self.cost_type = cost_type
        self.corruption_level = corruption_level
        self.x = input
        self.input = input
        self.params = []
        self.layers = []
        if rnd is None:
            rnd = np.random.RandomState(1231)
        self.theano_rng = RandomStreams(rnd.randint(2 ** 30))
        # Encoder part
        self.layers.append(
            DeepConvolutionLayer(input=self.x,
                                 layers=encoder_config,
                                 crop_size=crop_size))
        self.encoder = self.layers[0]
        self.params.extend(self.layers[0].params)
        code = self.layers[0].output
        output_features_dim = self.layers[0].output_features_dim
        # Decoder
        self.layers.append(
            DeepConvolutionLayer(input=code,
                                 layers=decoder_config,
                                 crop_size=output_features_dim))
        self.decoder = self.layers[1]
        self.output = self.layers[1].output
        self.params.extend(self.layers[1].params)
        self.L1 = 0
        self.L2 = 0
        self.L1_reg = l1_reg
        self.L2_reg = l2_reg
        for param in self.params:
            if ("w" in param.name) or ("W" in param.name):
                if l1_reg != 0.:
                    self.L1 += abs(param).sum()
                if l2_reg != 0.:
                    self.L2 += (param**2).sum()
            elif ("b" in param.name) or ("B" in param.name):
                if l1_reg != 0. and reg_bias:
                    self.L1 += abs(param).sum()
                if l2_reg != 0. and reg_bias:
                    self.L2 += (param**2).sum()
        if self.L1 != 0.:
            self.L1 *= sharedX_value(self.L1_reg, "l1_reg")
            print "Performing l1 reg. Lambda=", l1_reg
        if self.L2 != 0.:
            self.L2 *= sharedX_value(self.L2_reg, "l2_reg")
            print "Performing l2 reg. Lambda=", l2_reg

    def catch_params(self):
        self.catched_params = []
        for param in self.params:
            self.catched_params.append(param.get_value())

    def encode(self, x_in=None, train=False):
        if x_in is None:
            x_in = self.x
        self.encoder.x = x_in
        return self.encoder.output

    def decode(self, h):
        self.decoder.x = h
        return self.decoder.output

    def get_reconstructed_image(self, data):
        h = self.encode(x_in=data)
        x_rec = self.decode(h)
        return x_rec

    def get_rec_cost(self, x_rec, eyes=False):
        """
        Returns the reconstruction cost.
        """
        if self.cost_type == CostType.MeanSquared:
            return T.mean(((self.x.flatten(2) - x_rec)**2).sum(axis=1))
        elif self.cost_type == CostType.CrossEntropy:
            return T.mean(
                (T.nnet.binary_crossentropy(x_rec, self.x.flatten(2))).mean(axis=1))

    def get_reconstruction_error(self, x_in=None):
        if x_in is None:
            x_in = self.x
        h = self.encode(x_in, train=False)
        x_rec = self.decode(h)
        x_rec_faltten = x_rec.flatten(2)
        if self.cost_type == CostType.MeanSquared:
            return T.mean(((self.x.flatten(2) - x_rec_faltten)**2).sum(axis=1))
        elif self.cost_type == CostType.CrossEntropy:
            return T.mean(
                (T.nnet.binary_crossentropy(x_rec_faltten, self.x.flatten(2))).mean(axis=1))

    def get_train_cost(self, x_in=None):
        if x_in is None:
            if self.corruption_level != 0.:
                x_in = self.theano_rng.binomial(
                    x_in.shape, n=1, p=1-self.corruption_level,
                    dtype=theano.config.floatX) * self.x
            else:
                x_in = self.x
        h = self.encode(x_in)
        x_rec = self.decode(h)
        x_rec_faltten = x_rec.flatten(2)
        cost = self.get_rec_cost(x_rec_faltten)
        if self.L1_reg != 0.:
            cost += self.L1_reg * self.L1
        if self.L2_reg != 0.:
            cost += self.L2_reg * self.L2
        return (cost, h, x_rec)

    def save_params(self, weights_file, catched=False):
        """Save the model's parameters."""
        f_dump = open(weights_file, "w")
        params_vls = []
        if catched:
            if self.catched_params != []:
                params_vls = self.catched_params
            else:
                raise ValueError(
                    "You asked to save catched params," +
                    "but you didn't catch any!!!!!!!")
        else:
            for param in self.params:
                params_vls.append(param.get_value())
        pkl.dump(params_vls, f_dump, protocol=pkl.HIGHEST_PROTOCOL)
        f_dump.close()

    def set_params_vals(self, weights_file):
        """Set the values of the parameters."""
        with open(weights_file, 'r') as f:
            params_vls = pkl.load(f)
            for param, val in zip(self.params, params_vls):
                param.set_value(val)

    def switch_to_catched_params(self):
        """Set the model params to the catched ones."""
        if self.catched_params != []:
            for param, val in zip(self.params, self.catched_params):
                param.set_value(val)


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
