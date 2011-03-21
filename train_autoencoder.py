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

    def __init__(self, inp, filter_shape, image_shape):
        numpy_rng = numpy.random.RandomState() #732)

        assert image_shape[1] == filter_shape[1]
        self.inp = inp

        fan_in = numpy.prod(filter_shape[1:])
        W_bound = numpy.sqrt(6.0 / fan_in)

        W_values = numpy.asarray(numpy_rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=floatX)
        # print W_values.shape
        self.W = theano.shared(value=W_values)

        # convolve input feature maps with filters
        self.conv_out = conv.conv2d(input=inp, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)

        self.conv_out_shape = conv.ConvOp.getOutputShape(image_shape[2:4], filter_shape[2:4])
        # print post_conv_size
        # print filter_shape
        # print image_shape
        # 
        # return conv_out

        conv_out_rasterized = self.conv_out.reshape((filter_shape[0], -1))

        # y_max, self.argmax_y = T.max_and_argmax(self.conv_out, axis=-1)
        # self.max, self.argmax_x = T.max_and_argmax(y_max, axis=-1)
        self.max, self.argmax = T.max_and_argmax(conv_out_rasterized, axis=-1)

def train(training_data):
    # print training_data
    # print training_data.shape
    input_image = T.matrix("input_image")
    batch = input_image.reshape((1, 1, 17, 17)) # batch size, num inp filters, h, w
    filter_shape = (4, 1, 7, 7) # num filters, num inp filters, h, w
    image_shape = (1, 1, 17, 17) # batch size, num inp filters, h, w
    encoder = Encoder(batch, filter_shape, image_shape)
    out = encoder.conv_out
    g = theano.function(inputs=[input_image], outputs=[out, encoder.max, encoder.argmax])
    conv, conv_max, conv_argmax = g(training_data[0])
    print conv.shape
    print conv_max
    print conv_argmax
    argmax_unraveled = [numpy.unravel_index(i, encoder.conv_out_shape) for i in conv_argmax]
    print argmax_unraveled
    

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
