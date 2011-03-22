#!/usr/bin/env python

from __future__ import division

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

import numpy
import theano
import theano.tensor as T
import theano.tensor.signal.conv
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

        batch = self.image.reshape((1, 1, self.image_shape[2], self.image_shape[3])) # batch size, num inp filters, h, w

        fan_in = numpy.prod(self.filter_shape[1:])
        W_bound = numpy.sqrt(6.0 / fan_in)
        W_values = numpy.asarray(self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape), dtype=floatX)
        self.W = theano.shared(name="W_c", value=W_values)

        self.conv_out = conv.conv2d(input=batch, filters=self.W, filter_shape=self.filter_shape, image_shape=self.image_shape)

        conv_out_image_x, conv_out_image_y = conv.ConvOp.getOutputShape(self.image_shape[2:4], self.filter_shape[2:4])

        conv_out_rasterized = self.conv_out.reshape((self.filter_shape[0], -1))
        max_vals, argmax_raveled = T.max_and_argmax(conv_out_rasterized, axis=-1)
        argmax_x = argmax_raveled // conv_out_image_x
        argmax_y = argmax_raveled % conv_out_image_y
        argmax = T.cast(T.stack(argmax_x, argmax_y).T, "int32") # I don't fully understand why this cast is necessary
        self.encode = theano.function(inputs=[self.image], outputs=[max_vals, argmax])

def zeros_with_submatrix(submatrix, location, offset, submatrix_shape, destination_shape):
    assert all((d - 1) % 2 == 0 for d in submatrix_shape)
    pre_convolve_shape = tuple(d - s + 1 for s, d in zip(submatrix_shape, destination_shape))
    dest = T.zeros(pre_convolve_shape)
    submatrix_offset = tuple((d - 1) // 2 for d in submatrix_shape)
    dest_with_one = T.set_subtensor(dest[location[0] + offset[0] - submatrix_offset[0],
                                         location[1] + offset[1] - submatrix_offset[1]], 1.0)
    convolved = theano.tensor.signal.conv.conv2d(dest_with_one, submatrix, border_mode="full")
    convolved_shape_fixed = convolved.reshape(destination_shape)
    return convolved_shape_fixed

class Decoder(object):

    def __init__(self, features, locations, filters, image_shape, filter_shape):
        self.numpy_rng = numpy.random.RandomState() #732)

        conv_out_image_x, conv_out_image_y = conv.ConvOp.getOutputShape(image_shape[2:4], filter_shape[2:4])
        conv_offset_x = (image_shape[2] - conv_out_image_x) // 2
        conv_offset_y = (image_shape[3] - conv_out_image_y) // 2
        conv_offset = (-conv_offset_x, -conv_offset_y)

        fan_in = numpy.prod(filter_shape[1:])
        W_bound = numpy.sqrt(6.0 / fan_in)
        W_values = numpy.asarray(self.numpy_rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=floatX)
        # W_values = numpy.ones(shape=filter_shape, dtype=floatX)
        self.W = theano.shared(name="W_d", value=W_values)

        def accumulate(feature_weight, location, template, acc_at_time_i):
            filter_to_add = template[0] * feature_weight
            addend = zeros_with_submatrix(filter_to_add, location, conv_offset, filter_shape[2:4], (17, 17))
            return acc_at_time_i + addend

        scan_result, scan_updates = theano.scan(fn=accumulate,
                                                outputs_info=T.zeros(image_shape[2:], dtype=floatX),
                                                sequences=[features, locations, self.W])

        self.decode = theano.function(inputs=[features, locations],
                                      outputs=[scan_result[-1]],
                                      updates=scan_updates)


def train(training_data):
    input_image = T.matrix("input_image")
    filter_shape = (4, 1, 7, 7) # num filters, num inp filters, h, w
    image_shape = (1, 1, 17, 17) # batch size, num inp filters, h, w
    encoder = Encoder(input_image, filter_shape, image_shape)
    encoded_features, encoded_locations = encoder.encode(training_data[0])
    print encoded_features
    print encoded_locations

    features = T.vector("features")
    locations = T.imatrix("locations")
    decoder = Decoder(features, locations, encoder.W, image_shape, filter_shape)
    print decoder.decode(encoded_features, encoded_locations)

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
