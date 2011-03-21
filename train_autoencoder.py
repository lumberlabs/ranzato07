#!/usr/bin/env python

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

floatX = theano.config.floatX

class Encoder(object):

    def __init__(self, image, filter_shape, image_shape):
        self.numpy_rng = numpy.random.RandomState() #732)
        self.filter_shape = filter_shape
        self.image_shape = image_shape

        assert image_shape[1] == filter_shape[1]
        self.image = image
        self.init_encoder()

    def init_encoder(self):
        batch = self.image.reshape((1, 1, self.image_shape[2], self.image_shape[3])) # batch size, num inp filters, h, w
        fan_in = numpy.prod(self.filter_shape[1:])
        W_bound = numpy.sqrt(6.0 / fan_in)

        W_values = numpy.asarray(self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape), dtype=floatX)
        self.W = theano.shared(value=W_values)

        self.conv_out = conv.conv2d(input=batch, filters=self.W, filter_shape=self.filter_shape, image_shape=self.image_shape)

        conv_out_image_x, conv_out_image_y = conv.ConvOp.getOutputShape(self.image_shape[2:4], self.filter_shape[2:4])

        conv_out_rasterized = self.conv_out.reshape((self.filter_shape[0], -1))
        self.max, argmax_raveled = T.max_and_argmax(conv_out_rasterized, axis=-1)
        argmax_x = argmax_raveled / conv_out_image_x
        argmax_y = argmax_raveled % conv_out_image_y
        self.argmax = T.stack(argmax_x, argmax_y).T
        self.encode = theano.function(inputs=[self.image], outputs=[self.max, self.argmax])

def train(training_data):
    input_image = T.matrix("input_image")
    filter_shape = (4, 1, 7, 7) # num filters, num inp filters, h, w
    image_shape = (1, 1, 17, 17) # batch size, num inp filters, h, w
    encoder = Encoder(input_image, filter_shape, image_shape)
    encoded_features, encoded_locations = encoder.encode(training_data[0])
    print encoded_features
    print encoded_locations

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_file",
                        help="input"
                       )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        training_data = pickle.load(f)

    train(training_data)


if __name__ == '__main__':
    sys.exit(main())
