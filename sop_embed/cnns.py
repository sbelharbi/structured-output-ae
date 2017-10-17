from theano import tensor as T
import theano
import cPickle
import os
import numpy as np

from sop_embed.layers import HiddenLayer, LeNetConvPoolLayer
from sop_embed.layers import DropoutIdentityHiddenLayer
from sop_embed.layers import DropoutHiddenLayer
from sop_embed.layers import DropoutLeNetConvPoolLayer
from sop_embed.layers import dropout_from_layer
from sop_embed.tools import sharedX_value
from sop_embed.learning_rule import norm_constraint


def relu(x):
    return T.switch(x > 0, x, 0)


class base(object):
    def __init__(self, input):
        self.input = input
        self.params = []
        self.output = None
        self.output_dim = 1
        self.infos = None

    def save(self, filename):
        model_params = []
        with open(filename, 'w') as f:
            for layer in self.layers_dropout:
                model_params.append([p.get_value() for p in layer.params])

            params_to_dump = {"cnn_params": model_params, "infos": self.infos}
            cPickle.dump(params_to_dump, f, cPickle.HIGHEST_PROTOCOL)

    def set_params(self, list_params):
        for (layer, params) in zip(self.layers_dropout, list_params):
                for (p, pvl) in zip(layer.params, params):
                    p.set_value(pvl)

    def load(self, filename):
        with open(filename) as f:
            params_dumped = cPickle.load(f)
            model_params = params_dumped["cnn_params"]
            self.infos = params_dumped["infos"]
            for (layer, params) in zip(
                    self.layers_dropout, model_params):
                for (p, pvl) in zip(layer.params, params):
                    p.set_value(pvl)

#********************************** NEW


class CNN5Layers(base):
    def __init__(self, input, rng, cropsize, batch_size,
                 dropout_rates=[0., 0., 0., 0., 0., 0., 0., 0.],
                 nkerns=[10, 10, 10, 10, 10], filters=[11, 6, 6, 6, 6], inD=1,
                 poolsizes=[(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                 strides=[(4, 4), (1, 1), (1, 1), (1, 1), (1, 1)],
                 maxout=False, LRN=[None, None, None, None, None],
                 b_vs=[0., 0., 0., 0., 0., 0., 0., 0.]):
        """
        Input:
            cropsize: list. [height, width]
            inD: Integer. The 3th dimension of the input data. For example:
                1 means 1 image with size cropsize + [1].
            poolsizes: List of tuples. Each tuple: (h,w).
            thickness: Boolean. Indicates if an extra-output is needed for
                the slice thickness (True).
            xij: Boolean. Indicates if an extra-output is needed for the
                raw of the Xiphisternal joint.
            googlebox: Boolean. If True, we set an extra output using a sigmoid
                function to indicate if the L3 is present in the current
                window or not.
            maxout: Boolean. Indicates if to use or not a maxout activation.
            p: Float [0, 1]. The probablity that a unit is NOT dropped out.
            dense: Boolean. Use dense full connex layer.
            s_in: Int. The size of the hidden layer when dense is True.
            eyel3present: Boolean. If True, we create a full connex layer to
                detect if the l3 is prsent or not.
        """
        base.__init__(self, input)
        print cropsize, batch_size
        # Local response normalization
        default_LRN_none = {
            "app": False, "before": False, "alpha": 1e-4, "k": 2, "beta": 0.75,
            "n": 5}
        new_LRN = [default_LRN_none if e is None else e for e in LRN]
        poolmaxoutfactor = 1

        if maxout:
            poolmaxoutfactor = 2  # We pool every 2 maps.

        self.layers, self.params = [], []
        self.layers_dropout = []
        next_layer_input = self.input
        layer_ident_in = DropoutIdentityHiddenLayer(rng, next_layer_input,
                                                    dropout_rates[0],
                                                    False)
        next_dropout_layer_input = layer_ident_in.output

        layer0_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, inD, cropsize[0], cropsize[1]),
            filter_shape=(nkerns[0], inD, filters[0], filters[0]),
            dropout_rate=dropout_rates[1],
            rescale=False,
            poolsize=poolsizes[0],
            stride=strides[0],
            LRN=new_LRN[0],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[0]
        )

        layer0 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, inD, cropsize[0], cropsize[1]),
            filter_shape=(nkerns[0], inD, filters[0], filters[0]),
            poolsize=poolsizes[0],
            stride=strides[0],
            LRN=new_LRN[0],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer0_dropout.W * (1 - dropout_rates[1]),
            b=layer0_dropout.b
        )

        self.layers_dropout += [layer0_dropout]
        self.layers += [layer0]
        self.params += layer0_dropout.params
        next_dropout_layer_input = layer0_dropout.output
        next_layer_input = layer0.output
        # filtering reduce to: 400 - 11 + 1= 390 .
        # Max pooling reduce to: 390 / 2 = 195
        # 300 - 11 + 1 = 290 / 2 = 145
        s = strides[0]
        map_size_h = (cropsize[0] - filters[0] + 1)/s[0] / poolsizes[0][0]
        map_size_w = (cropsize[1] - filters[0] + 1)/s[1] / poolsizes[0][1]

        layer1_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[0]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[1], nkerns[0]/poolmaxoutfactor,
                          filters[1], filters[1]),
            dropout_rate=dropout_rates[2],
            rescale=False,
            poolsize=poolsizes[1],
            stride=strides[1],
            LRN=new_LRN[1],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[1]
        )

        layer1 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[0]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[1], nkerns[0]/poolmaxoutfactor,
                          filters[1], filters[1]),
            poolsize=poolsizes[1],
            stride=strides[1],
            LRN=new_LRN[1],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer1_dropout.W * (1 - dropout_rates[2]),
            b=layer1_dropout.b
        )
        self.layers_dropout += [layer1_dropout]
        self.layers += [layer1]
        self.params += layer1_dropout.params
        next_dropout_layer_input = layer1_dropout.output
        next_layer_input = layer1.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[1]
        map_size_h = (map_size_h - filters[1] + 1)/s[0] / poolsizes[1][0]
        map_size_w = (map_size_w - filters[1] + 1)/s[1] / poolsizes[1][1]

        layer2_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[1]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[2], nkerns[1]/poolmaxoutfactor,
                          filters[2], filters[2]),
            dropout_rate=dropout_rates[3],
            rescale=False,
            poolsize=poolsizes[2],
            stride=strides[2],
            LRN=new_LRN[2],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[2]
        )

        layer2 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[1]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[2], nkerns[1]/poolmaxoutfactor,
                          filters[2], filters[2]),
            poolsize=poolsizes[2],
            stride=strides[2],
            LRN=new_LRN[2],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer2_dropout.W * (1 - dropout_rates[3]),
            b=layer2_dropout.b
        )
        self.layers_dropout += [layer2_dropout]
        self.layers += [layer2]
        self.params += layer2_dropout.params
        next_dropout_layer_input = layer2_dropout.output
        next_layer_input = layer2.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[2]
        map_size_h = (map_size_h - filters[2] + 1)/s[0] / poolsizes[2][0]
        map_size_w = (map_size_w - filters[2] + 1)/s[1] / poolsizes[2][1]

        layer3_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[2]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[3], nkerns[2]/poolmaxoutfactor,
                          filters[3], filters[3]),
            dropout_rate=dropout_rates[4],
            rescale=False,
            poolsize=poolsizes[3],
            stride=strides[3],
            LRN=new_LRN[3],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[3]
        )

        layer3 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[2]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[3], nkerns[2]/poolmaxoutfactor,
                          filters[3], filters[3]),
            poolsize=poolsizes[3],
            stride=strides[3],
            LRN=new_LRN[3],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer3_dropout.W * (1 - dropout_rates[4]),
            b=layer3_dropout.b
        )
        self.layers_dropout += [layer3_dropout]
        self.layers += [layer3]
        self.params += layer3_dropout.params
        next_dropout_layer_input = layer3_dropout.output
        next_layer_input = layer3.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[3]
        map_size_h = (map_size_h - filters[3] + 1)/s[0] / poolsizes[3][0]
        map_size_w = (map_size_w - filters[3] + 1)/s[1] / poolsizes[3][1]

        layer4_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[3]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[4], nkerns[3]/poolmaxoutfactor,
                          filters[4], filters[4]),
            dropout_rate=dropout_rates[5],
            rescale=False,
            poolsize=poolsizes[4],
            stride=strides[4],
            LRN=new_LRN[4],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[4]
        )

        layer4 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[3]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[4], nkerns[3]/poolmaxoutfactor,
                          filters[4], filters[4]),
            poolsize=poolsizes[4],
            stride=strides[4],
            LRN=new_LRN[4],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer4_dropout.W * (1 - dropout_rates[5]),
            b=layer4_dropout.b
        )
        self.layers_dropout += [layer4_dropout]
        self.layers += [layer4]
        self.params += layer4_dropout.params
        next_dropout_layer_input = layer4_dropout.output
        next_layer_input = layer4.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[4]
        map_size_h = (map_size_h - filters[4] + 1)/s[0] / poolsizes[4][0]
        map_size_w = (map_size_w - filters[4] + 1)/s[1] / poolsizes[4][1]
        next_dropout_layer_input = next_dropout_layer_input.flatten(2)
        next_layer_input = next_layer_input.flatten(2)
        self.output_dim = (
            nkerns[4] / poolmaxoutfactor) * map_size_h * map_size_w

        self.output_dropout = next_dropout_layer_input
        self.output = next_layer_input
        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "CNN model contains %i parameters" % nparams


class CNN4Layers(base):
    def __init__(self, rng, cropsize, batch_size, output_activation,
                 dropout_rates=[0., 0., 0., 0., 0., 0., 0.],
                 nkerns=[10, 10, 10, 10], filters=[11, 6, 6, 6], inD=1,
                 poolsizes=[(2, 2), (2, 2), (2, 2), (2, 2)],
                 strides=[(4, 4), (1, 1), (1, 1), (1, 1)],
                 maxout=False, LRN=[None, None, None, None],
                 b_vs=[0., 0., 0., 0., 0., 0., 0.],
                 dense=0, dense_layers=[1]):
        """
        Input:
            cropsize: list. [height, width]
            inD: Integer. The 3th dimension of the input data. For example:
                1 means 1 image with size cropsize + [1].
            poolsizes: List of tuples. Each tuple: (h,w).
            thickness: Boolean. Indicates if an extra-output is needed for
                the slice thickness (True).
            xij: Boolean. Indicates if an extra-output is needed for the
                raw of the Xiphisternal joint.
            googlebox: Boolean. If True, we set an extra output using a sigmoid
                function to indicate if the L3 is present in the current
                window or not.
            maxout: Boolean. Indicates if to use or not a maxout activation.
            p: Float [0, 1]. The probablity that a unit is NOT dropped out.
            dense: Boolean. Use dense full connex layer.
            s_in: Int. The size of the hidden layer when dense is True.
            eyel3present: Boolean. If True, we create a full connex layer to
                detect if the l3 is prsent or not.
        """
        base.__init__(self)
        # Local response normalization
        default_LRN_none = {
            "app": False, "before": False, "alpha": 1e-4, "k": 2, "beta": 0.75,
            "n": 5}
        new_LRN = [default_LRN_none if e is None else e for e in LRN]
        poolmaxoutfactor = 1

        if maxout:
            poolmaxoutfactor = 2  # We pool every 2 maps.

        self.X_batch, self.y_batch = T.tensor4('x'), T.matrix('y')
        self.X_batch_eyel3 = T.tensor4('x')
        self.layers, self.params = [], []
        self.layers_dropout = []
        next_layer_input = self.X_batch
        layer_ident_in = DropoutIdentityHiddenLayer(rng, next_layer_input,
                                                    dropout_rates[0],
                                                    False)
        next_dropout_layer_input = layer_ident_in.output

        layer0_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, inD, cropsize[0], cropsize[1]),
            filter_shape=(nkerns[0], inD, filters[0], filters[0]),
            dropout_rate=dropout_rates[1],
            rescale=False,
            poolsize=poolsizes[0],
            stride=strides[0],
            LRN=new_LRN[0],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[0]
        )

        layer0 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, inD, cropsize[0], cropsize[1]),
            filter_shape=(nkerns[0], inD, filters[0], filters[0]),
            poolsize=poolsizes[0],
            stride=strides[0],
            LRN=new_LRN[0],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer0_dropout.W * (1 - dropout_rates[1]),
            b=layer0_dropout.b
        )

        self.layers_dropout += [layer0_dropout]
        self.layers += [layer0]
        self.params += layer0_dropout.params
        next_dropout_layer_input = layer0_dropout.output
        next_layer_input = layer0.output
        # filtering reduce to: 400 - 11 + 1= 390 .
        # Max pooling reduce to: 390 / 2 = 195
        # 300 - 11 + 1 = 290 / 2 = 145
        s = strides[0]
        map_size_h = (cropsize[0] - filters[0] + 1)/s[0] / poolsizes[0][0]
        map_size_w = (cropsize[1] - filters[0] + 1)/s[1] / poolsizes[0][1]

        layer1_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[0]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[1], nkerns[0]/poolmaxoutfactor,
                          filters[1], filters[1]),
            dropout_rate=dropout_rates[2],
            rescale=False,
            poolsize=poolsizes[1],
            stride=strides[1],
            LRN=new_LRN[1],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[1]
        )

        layer1 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[0]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[1], nkerns[0]/poolmaxoutfactor,
                          filters[1], filters[1]),
            poolsize=poolsizes[1],
            stride=strides[1],
            LRN=new_LRN[1],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer1_dropout.W * (1 - dropout_rates[2]),
            b=layer1_dropout.b
        )
        self.layers_dropout += [layer1_dropout]
        self.layers += [layer1]
        self.params += layer1_dropout.params
        next_dropout_layer_input = layer1_dropout.output
        next_layer_input = layer1.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[1]
        map_size_h = (map_size_h - filters[1] + 1)/s[0] / poolsizes[1][0]
        map_size_w = (map_size_w - filters[1] + 1)/s[1] / poolsizes[1][1]

        layer2_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[1]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[2], nkerns[1]/poolmaxoutfactor,
                          filters[2], filters[2]),
            dropout_rate=dropout_rates[3],
            rescale=False,
            poolsize=poolsizes[2],
            stride=strides[2],
            LRN=new_LRN[2],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[2]
        )

        layer2 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[1]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[2], nkerns[1]/poolmaxoutfactor,
                          filters[2], filters[2]),
            poolsize=poolsizes[2],
            stride=strides[2],
            LRN=new_LRN[2],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer2_dropout.W * (1 - dropout_rates[3]),
            b=layer2_dropout.b
        )
        self.layers_dropout += [layer2_dropout]
        self.layers += [layer2]
        self.params += layer2_dropout.params
        next_dropout_layer_input = layer2_dropout.output
        next_layer_input = layer2.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[2]
        map_size_h = (map_size_h - filters[2] + 1)/s[0] / poolsizes[2][0]
        map_size_w = (map_size_w - filters[2] + 1)/s[1] / poolsizes[2][1]

        layer3_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[2]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[3], nkerns[2]/poolmaxoutfactor,
                          filters[3], filters[3]),
            dropout_rate=dropout_rates[4],
            rescale=False,
            poolsize=poolsizes[3],
            stride=strides[3],
            LRN=new_LRN[3],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[3]
        )

        layer3 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[2]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[3], nkerns[2]/poolmaxoutfactor,
                          filters[3], filters[3]),
            poolsize=poolsizes[3],
            stride=strides[3],
            LRN=new_LRN[3],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer3_dropout.W * (1 - dropout_rates[4]),
            b=layer3_dropout.b
        )
        self.layers_dropout += [layer3_dropout]
        self.layers += [layer3]
        self.params += layer3_dropout.params
        next_dropout_layer_input = layer3_dropout.output
        next_layer_input = layer3.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[3]
        map_size_h = (map_size_h - filters[3] + 1)/s[0] / poolsizes[3][0]
        map_size_w = (map_size_w - filters[3] + 1)/s[1] / poolsizes[3][1]
        next_dropout_layer_input = next_dropout_layer_input.flatten(2)
        next_layer_input = next_layer_input.flatten(2)
        # Size: (nkerns[3]/poolmaxoutfactor) * map_size_h * map_size_w,

        self.output_dropout = self.layers_dropout[-1].output
        self.output = self.layers[-1].output
        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "model contains %i parameters" % nparams


class CNN3Layers(base):
    def __init__(self, rng, cropsize, batch_size, output_activation,
                 dropout_rates=[0., 0., 0., 0., 0., 0.],
                 nkerns=[10, 10, 10], filters=[11, 6, 6], inD=1,
                 poolsizes=[(2, 2), (2, 2), (2, 2)],
                 strides=[(4, 4), (1, 1), (1, 1)],
                 maxout=False, LRN=[None, None, None],
                 b_vs=[0., 0., 0., 0., 0., 0.],
                 dense=0, dense_layers=[1]):
        """
        Input:
            cropsize: list. [height, width]
            inD: Integer. The 3th dimension of the input data. For example:
                1 means 1 image with size cropsize + [1].
            poolsizes: List of tuples. Each tuple: (h,w).
            thickness: Boolean. Indicates if an extra-output is needed for
                the slice thickness (True).
            xij: Boolean. Indicates if an extra-output is needed for the
                raw of the Xiphisternal joint.
            googlebox: Boolean. If True, we set an extra output using a sigmoid
                function to indicate if the L3 is present in the current
                window or not.
            maxout: Boolean. Indicates if to use or not a maxout activation.
            p: Float [0, 1]. The probablity that a unit is NOT dropped out.
            dense: Boolean. Use dense full connex layer.
            s_in: Int. The size of the hidden layer when dense is True.
            eyel3present: Boolean. If True, we create a full connex layer to
                detect if the l3 is prsent or not.
        """
        base.__init__(self)
        # Local response normalization
        default_LRN_none = {
            "app": False, "before": False, "alpha": 1e-4, "k": 2, "beta": 0.75,
            "n": 5}
        new_LRN = [default_LRN_none if e is None else e for e in LRN]
        poolmaxoutfactor = 1

        if maxout:
            poolmaxoutfactor = 2  # We pool every 2 maps.

        self.X_batch, self.y_batch = T.tensor4('x'), T.matrix('y')
        self.X_batch_eyel3 = T.tensor4('x')
        self.layers, self.params = [], []
        self.layers_dropout = []
        next_layer_input = self.X_batch
        layer_ident_in = DropoutIdentityHiddenLayer(rng, next_layer_input,
                                                    dropout_rates[0],
                                                    False)
        next_dropout_layer_input = layer_ident_in.output

        layer0_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, inD, cropsize[0], cropsize[1]),
            filter_shape=(nkerns[0], inD, filters[0], filters[0]),
            dropout_rate=dropout_rates[1],
            rescale=False,
            poolsize=poolsizes[0],
            stride=strides[0],
            LRN=new_LRN[0],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[0],
        )

        layer0 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, inD, cropsize[0], cropsize[1]),
            filter_shape=(nkerns[0], inD, filters[0], filters[0]),
            poolsize=poolsizes[0],
            stride=strides[0],
            LRN=new_LRN[0],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer0_dropout.W * (1 - dropout_rates[1]),
            b=layer0_dropout.b
        )

        self.layers_dropout += [layer0_dropout]
        self.layers += [layer0]
        self.params += layer0_dropout.params
        next_dropout_layer_input = layer0_dropout.output
        next_layer_input = layer0.output
        # filtering reduce to: 400 - 11 + 1= 390 .
        # Max pooling reduce to: 390 / 2 = 195
        # 300 - 11 + 1 = 290 / 2 = 145
        s = strides[0]
        map_size_h = (cropsize[0] - filters[0] + 1)/s[0] / poolsizes[0][0]
        map_size_w = (cropsize[1] - filters[0] + 1)/s[1] / poolsizes[0][1]

        layer1_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[0]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[1], nkerns[0]/poolmaxoutfactor,
                          filters[1], filters[1]),
            dropout_rate=dropout_rates[2],
            rescale=False,
            poolsize=poolsizes[1],
            stride=strides[1],
            LRN=new_LRN[1],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[1],
        )

        layer1 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[0]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[1], nkerns[0]/poolmaxoutfactor,
                          filters[1], filters[1]),
            poolsize=poolsizes[1],
            stride=strides[1],
            LRN=new_LRN[1],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer1_dropout.W * (1 - dropout_rates[2]),
            b=layer1_dropout.b
        )
        self.layers_dropout += [layer1_dropout]
        self.layers += [layer1]
        self.params += layer1_dropout.params
        next_dropout_layer_input = layer1_dropout.output
        next_layer_input = layer1.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[1]
        map_size_h = (map_size_h - filters[1] + 1)/s[0] / poolsizes[1][0]
        map_size_w = (map_size_w - filters[1] + 1)/s[1] / poolsizes[1][1]

        layer2_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[1]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[2], nkerns[1]/poolmaxoutfactor,
                          filters[2], filters[2]),
            dropout_rate=dropout_rates[3],
            rescale=False,
            poolsize=poolsizes[2],
            stride=strides[2],
            LRN=new_LRN[2],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[2],
        )

        layer2 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[1]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[2], nkerns[1]/poolmaxoutfactor,
                          filters[2], filters[2]),
            poolsize=poolsizes[2],
            stride=strides[2],
            LRN=new_LRN[2],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer2_dropout.W * (1 - dropout_rates[3]),
            b=layer2_dropout.b
        )
        self.layers_dropout += [layer2_dropout]
        self.layers += [layer2]
        self.params += layer2_dropout.params
        next_dropout_layer_input = layer2_dropout.output
        next_layer_input = layer2.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[2]
        map_size_h = (map_size_h - filters[2] + 1)/s[0] / poolsizes[2][0]
        map_size_w = (map_size_w - filters[2] + 1)/s[1] / poolsizes[2][1]
        next_dropout_layer_input = next_dropout_layer_input.flatten(2)
        next_layer_input = next_layer_input.flatten(2)

        n_out = 1
        activation = output_activation
        if dense > 0:
            n_out = dense_layers[0]
            activation = relu

        layer3_dropout = HiddenLayer(
            rng,
            input=next_dropout_layer_input,
            n_in=(nkerns[2]/poolmaxoutfactor) * map_size_h * map_size_w,
            n_out=n_out,
            W=None,
            b=None,
            b_v=b_vs[3],
            activation=activation
        )

        layer3 = HiddenLayer(
            rng,
            input=next_layer_input,
            n_in=(nkerns[2]/poolmaxoutfactor) * map_size_h * map_size_w,
            n_out=n_out,
            W=layer3_dropout.W,
            b=layer3_dropout.b,
            activation=activation
        )
        self.layers_dropout += [layer3_dropout]
        self.layers += [layer3]
        self.params += layer3_dropout.params

        next_dropout_layer_input = layer3_dropout.output
        next_layer_input = layer3.output

        activation = output_activation
        if dense in [1, 2]:
            n_out = dense_layers[1]
            if dense == 2:
                activation = relu
            layer4_dropout = HiddenLayer(
                rng,
                input=next_dropout_layer_input,
                n_in=dense_layers[0],
                n_out=n_out,
                W=None,
                b=None,
                b_v=b_vs[4],
                activation=activation
            )

            layer4 = HiddenLayer(
                rng,
                input=next_layer_input,
                n_in=dense_layers[0],
                n_out=n_out,
                W=layer4_dropout.W,
                b=layer4_dropout.b,
                activation=activation
            )
            self.layers_dropout += [layer4_dropout]
            self.layers += [layer4]
            self.params += layer4_dropout.params

            next_dropout_layer_input = layer4_dropout.output
            next_layer_input = layer4.output

        activation = output_activation
        if dense == 2:
            n_out = dense_layers[2]
            layer5_dropout = HiddenLayer(
                rng,
                input=next_dropout_layer_input,
                n_in=dense_layers[1],
                n_out=n_out,
                W=None,
                b=None,
                b_v=b_vs[5],
                activation=activation
            )

            layer5 = HiddenLayer(
                rng,
                input=next_layer_input,
                n_in=dense_layers[1],
                n_out=n_out,
                W=layer5_dropout.W,
                b=layer5_dropout.b,
                activation=activation
            )
            self.layers_dropout += [layer5_dropout]
            self.layers += [layer5]
            self.params += layer5_dropout.params

        self.output_dropout = self.layers_dropout[-1].output
        self.output = self.layers[-1].output
        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "model contains %i parameters" % nparams


class CNN2Layers(base):
    def __init__(self, rng, cropsize, batch_size, output_activation,
                 dropout_rates=[0., 0., 0., 0., 0.],
                 nkerns=[10, 10], filters=[11, 6], inD=1,
                 poolsizes=[(2, 2), (2, 2)], strides=[(4, 4), (1, 1)],
                 maxout=False, LRN=[None, None], b_vs=[0., 0., 0., 0., 0.],
                 dense=0, dense_layers=[1]):
        """
        Input:
            cropsize: list. [height, width]
            inD: Integer. The 3th dimension of the input data. For example:
                1 means 1 image with size cropsize + [1].
            poolsizes: List of tuples. Each tuple: (h,w).
            thickness: Boolean. Indicates if an extra-output is needed for
                the slice thickness (True).
            xij: Boolean. Indicates if an extra-output is needed for the
                raw of the Xiphisternal joint.
            googlebox: Boolean. If True, we set an extra output using a sigmoid
                function to indicate if the L3 is present in the current
                window or not.
            maxout: Boolean. Indicates if to use or not a maxout activation.
            p: Float [0, 1]. The probablity that a unit is NOT dropped out.
            dense: Boolean. Use dense full connex layer.
            s_in: Int. The size of the hidden layer when dense is True.
            eyel3present: Boolean. If True, we create a full connex layer to
                detect if the l3 is prsent or not.
        """
        base.__init__(self)
        # Local response normalization
        default_LRN_none = {
            "app": False, "before": False, "alpha": 1e-4, "k": 2, "beta": 0.75,
            "n": 5}
        new_LRN = [default_LRN_none if e is None else e for e in LRN]
        poolmaxoutfactor = 1

        if maxout:
            poolmaxoutfactor = 2  # We pool every 2 maps.

        self.X_batch, self.y_batch = T.tensor4('x'), T.matrix('y')
        self.X_batch_eyel3 = T.tensor4('x')
        self.layers, self.params = [], []
        self.layers_dropout = []
        next_layer_input = self.X_batch
        layer_ident_in = DropoutIdentityHiddenLayer(rng, next_layer_input,
                                                    dropout_rates[0],
                                                    False)
        next_dropout_layer_input = layer_ident_in.output

        layer0_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, inD, cropsize[0], cropsize[1]),
            filter_shape=(nkerns[0], inD, filters[0], filters[0]),
            dropout_rate=dropout_rates[1],
            rescale=False,
            poolsize=poolsizes[0],
            stride=strides[0],
            LRN=new_LRN[0],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[0]
        )

        layer0 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, inD, cropsize[0], cropsize[1]),
            filter_shape=(nkerns[0], inD, filters[0], filters[0]),
            poolsize=poolsizes[0],
            stride=strides[0],
            LRN=new_LRN[0],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer0_dropout.W * (1 - dropout_rates[1]),
            b=layer0_dropout.b
        )

        self.layers_dropout += [layer0_dropout]
        self.layers += [layer0]
        self.params += layer0_dropout.params
        next_dropout_layer_input = layer0_dropout.output
        next_layer_input = layer0.output
        # filtering reduce to: 400 - 11 + 1= 390 .
        # Max pooling reduce to: 390 / 2 = 195
        # 300 - 11 + 1 = 290 / 2 = 145
        s = strides[0]
        map_size_h = (cropsize[0] - filters[0] + 1)/s[0] / poolsizes[0][0]
        map_size_w = (cropsize[1] - filters[0] + 1)/s[1] / poolsizes[0][1]

        layer1_dropout = DropoutLeNetConvPoolLayer(
            rng,
            input=next_dropout_layer_input,
            image_shape=(batch_size, nkerns[0]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[1], nkerns[0]/poolmaxoutfactor,
                          filters[1], filters[1]),
            dropout_rate=dropout_rates[2],
            rescale=False,
            poolsize=poolsizes[1],
            stride=strides[1],
            LRN=new_LRN[1],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=None,
            b=None,
            b_v=b_vs[1]
        )

        layer1 = LeNetConvPoolLayer(
            rng,
            input=next_layer_input,
            image_shape=(batch_size, nkerns[0]/poolmaxoutfactor,
                         map_size_h, map_size_w),
            filter_shape=(nkerns[1], nkerns[0]/poolmaxoutfactor,
                          filters[1], filters[1]),
            poolsize=poolsizes[1],
            stride=strides[1],
            LRN=new_LRN[1],
            maxout=maxout,
            poolmaxoutfactor=poolmaxoutfactor,
            W=layer1_dropout.W * (1 - dropout_rates[2]),
            b=layer1_dropout.b
        )
        self.layers_dropout += [layer1_dropout]
        self.layers += [layer1]
        self.params += layer1_dropout.params
        next_dropout_layer_input = layer1_dropout.output
        next_layer_input = layer1.output
        # 195 - 6 + 1 = 190 / 2 = 95
        # 145 - 6 + 1 = 140 / 2 = 70
        s = strides[1]
        map_size_h = (map_size_h - filters[1] + 1)/s[0] / poolsizes[1][0]
        map_size_w = (map_size_w - filters[1] + 1)/s[1] / poolsizes[1][1]
        next_dropout_layer_input = next_dropout_layer_input.flatten(2)
        next_layer_input = next_layer_input.flatten(2)

        n_out = 1
        activation = output_activation
        if dense > 0:
            n_out = dense_layers[0]
            activation = relu
        layer2_dropout = HiddenLayer(
            rng,
            input=next_dropout_layer_input,
            n_in=(nkerns[1]/poolmaxoutfactor) * map_size_h * map_size_w,
            n_out=n_out,
            W=None,
            b=None,
            b_v=b_vs[2],
            activation=activation
        )

        layer2 = HiddenLayer(
            rng,
            input=next_layer_input,
            n_in=(nkerns[1]/poolmaxoutfactor) * map_size_h * map_size_w,
            n_out=n_out,
            W=layer2_dropout.W * (1 - dropout_rates[4]),
            b=layer2_dropout.b,
            activation=activation
        )
        self.layers_dropout += [layer2_dropout]
        self.layers += [layer2]
        self.params += layer2_dropout.params

        next_dropout_layer_input = layer2_dropout.output
        next_layer_input = layer2.output

        activation = output_activation
        if dense in [1, 2]:
            n_out = dense_layers[1]
            if dense == 2:
                activation = relu
            layer3_dropout = HiddenLayer(
                rng,
                input=next_dropout_layer_input,
                n_in=dense_layers[0],
                n_out=n_out,
                W=None,
                b=None,
                b_v=b_vs[3],
                activation=activation
            )

            layer3 = HiddenLayer(
                rng,
                input=next_layer_input,
                n_in=dense_layers[0],
                n_out=n_out,
                W=layer3_dropout.W * (1 - dropout_rates[5]),
                b=layer3_dropout.b,
                activation=activation
            )
            self.layers_dropout += [layer3_dropout]
            self.layers += [layer3]
            self.params += layer3_dropout.params

            next_dropout_layer_input = layer3_dropout.output
            next_layer_input = layer3.output

        activation = output_activation
        if dense == 2:
            n_out = dense_layers[2]
            layer4_dropout = HiddenLayer(
                rng,
                input=next_dropout_layer_input,
                n_in=dense_layers[1],
                n_out=n_out,
                W=None,
                b=None,
                b_v=b_vs[4],
                activation=activation
            )

            layer4 = HiddenLayer(
                rng,
                input=next_layer_input,
                n_in=dense_layers[1],
                n_out=n_out,
                W=layer4_dropout.W,
                b=layer4_dropout.b,
                activation=activation
            )
            self.layers_dropout += [layer4_dropout]
            self.layers += [layer4]
            self.params += layer4_dropout.params

        self.output_dropout = self.layers_dropout[-1].output
        self.output = self.layers[-1].output
        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "model contains %i parameters" % nparams
#********************************** NEW*
