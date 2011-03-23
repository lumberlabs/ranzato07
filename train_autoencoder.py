#!/usr/bin/env python

from __future__ import division

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

import PIL.Image
import numpy
import theano
import theano.tensor as T
import theano.tensor.signal.conv
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from utils import tile_raster_images

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
        self.feature_weights, argmax_raveled = T.max_and_argmax(conv_out_rasterized, axis=-1)
        argmax_x = argmax_raveled // conv_out_image_x
        argmax_y = argmax_raveled % conv_out_image_y
        self.feature_locations = T.cast(T.stack(argmax_x, argmax_y).T, "int32") # I don't fully understand why this cast is necessary
        self.params = [self.W]

    def encoder_energy(self, base_feature_weights):
        return ((self.feature_weights - base_feature_weights) ** 2).sum()

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

        self.decoded = scan_result[-1]
        self.params = [self.W]

    def decoder_energy(self, image):
        return ((image - self.decoded) ** 2).sum()

def gradient_updates(score, params, learning_rate):
    gradient_params = [T.grad(score, param) for param in params]

    updates = {}
    for param, gradient_params in zip(params, gradient_params):
        updates[param] = param - learning_rate * gradient_params

    return updates

def train(training_data):
    input_image = T.matrix("input_image")
    filter_shape = (4, 1, 7, 7) # num filters, num inp filters, h, w
    image_shape = (1, 1, 17, 17) # batch size, num inp filters, h, w
    encoder = Encoder(input_image, filter_shape, image_shape)
    encode = theano.function(inputs=[input_image], outputs=[encoder.feature_weights, encoder.feature_locations])

    features = T.vector("features")
    locations = T.imatrix("locations")
    simple_decoder = Decoder(features, locations, encoder.W, image_shape, filter_shape)
    # decode = theano.function(inputs=[input_image, features, locations],
    #                          outputs=decoder.decoded)
    # calc_decoder_energy = theano.function(inputs=[input_image, features, locations],
    #                                       outputs=decoder_energy)

    # base_feature_weights = T.vector("base_feature_weights")
    # calc_encoder_energy = theano.function(inputs=[input_image, base_feature_weights],
    #                                       outputs=encoder_energy)

    ideal_weights = theano.shared(name="ideal_weights", value=numpy.zeros((4,), dtype=floatX))

    decoder_for_ideal_weights = Decoder(ideal_weights, locations, encoder.W, image_shape, filter_shape)
    decoder_energy = decoder_for_ideal_weights.decoder_energy(input_image)
    encoder_energy = encoder.encoder_energy(ideal_weights)

    total_energy = encoder_energy + decoder_energy

    energy_params = [ideal_weights]
    step_energy = theano.function(inputs=[input_image, locations],
                                  outputs=total_energy,
                                  updates=gradient_updates(total_energy, energy_params, learning_rate=0.05))

    decoder_params = decoder_for_ideal_weights.params
    step_decoder = theano.function(inputs=[input_image, locations],
                                   outputs=decoder_energy,
                                   updates=gradient_updates(decoder_energy, decoder_params, learning_rate=0.05))

    encoder_params = encoder.params
    step_encoder = theano.function(inputs=[input_image],
                                   outputs=encoder_energy,
                                   updates=gradient_updates(encoder_energy, encoder_params, learning_rate=0.05))


    for image_index, image in enumerate(training_data):
        encoded_features, encoded_locations = encode(image)
        # decoded_image = decode(encoded_features, encoded_locations)

        ideal_weights.set_value(encoded_features)

        prior_energy = float("inf")
        keep_going = True
        while keep_going:
            current_energy = step_energy(image, encoded_locations)
            keep_going = current_energy < prior_energy
            prior_energy = current_energy

        # print "TOTAL", current_energy
        # 
        decoder_energy = step_decoder(image, encoded_locations)
        # print "DECODER", decoder_energy
        encoder_energy = step_encoder(image)
        # print "ENCODER", encoder_energy

        if image_index % 25 == 0:
            print "Dumping images at image index {i}".format(i=image_index)
            image = PIL.Image.fromarray(tile_raster_images(X=encoder.W.get_value(),
                                        img_shape=(7, 7),
                                        tile_shape=(2, 2), 
                                        tile_spacing=(1, 1)))
            image.save("encoder_filters_{i}.png".format(i=image_index))
            image = PIL.Image.fromarray(tile_raster_images(X=decoder_for_ideal_weights.W.get_value(),
                                        img_shape=(7, 7),
                                        tile_shape=(2, 2), 
                                        tile_spacing=(1, 1)))
            image.save("decoder_filters_{i}.png".format(i=image_index))


        # for x in xrange(500):
        #     print step_down_energy(image, encoded_locations)

        # print decoded_image
        # 
        # print encoded_features
        # print encoded_locations
        # print decoded_image
        # print calc_decoder_energy(encoded_features, encoded_locations, image)
        # print calc_encoder_energy(image, encoded_features)

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
