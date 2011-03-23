#!/usr/bin/env python

from __future__ import division

import argparse
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

import PIL.Image
import numpy
import theano
import theano.tensor as T
import theano.tensor.signal.conv # for convolutions
import theano.tensor.nnet.conv # for getOutputShape

from utils import tile_raster_images

floatX = theano.config.floatX

class Encoder(object):

    def __init__(self, image_variable, filter_shape, image_shape, numpy_rng=None):
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState()

        num_filters = filter_shape[0]
        individual_filter_shape = filter_shape[1:]

        # TODO: This is completely unmotivated. It comes purely from the observation that
        # initializing W_bound = 1/100 works pretty well for a 7x7 filter, and 100 ~= 2 * 7 * 7. :/
        fan_in = numpy.prod(individual_filter_shape)
        filters_elem_bound = 1 / (2 * fan_in) 
        filters_value = numpy.asarray(numpy_rng.uniform(low=-filters_elem_bound, high=filters_elem_bound, size=filter_shape), dtype=floatX)

        self.filters = theano.shared(name="encoder_filters", value=filters_value)

        convolved = theano.tensor.signal.conv.conv2d(image_variable, self.filters)
        convolved_rows, convolved_cols = theano.tensor.nnet.conv.ConvOp.getOutputShape(image_shape, individual_filter_shape)

        # rasterize each convolved image, since max_and_argmax doesn't accept multiple axes
        convolved_rasterized = convolved.reshape((num_filters, -1))
        self.code, argmax_raveled = T.max_and_argmax(convolved_rasterized, axis=-1)

        # now unravel the argmax value to undo the rasterization
        argmax_row = argmax_raveled // convolved_rows
        argmax_col = argmax_raveled % convolved_rows
        locations_upcast = T.stack(argmax_row, argmax_col).T

        self.locations = T.cast(locations_upcast, "int32") # the // and % upcast from int32 to int64; cast back down

        self.params = [self.filters]

    def encoder_energy(self, wrt_code):
        return ((self.code - wrt_code) ** 2).sum()

def zeros_with_submatrix(submatrix, center_location, offset, submatrix_shape, destination_shape):
    """
    Helper function to fill a large matrix with zeros, and set a subportion of it to match a provided submatrix.
    """
    # allow no even dimensions in submatrix -- then there is no clean center
    assert all((dim - 1) % 2 == 0 for dim in submatrix_shape)

    # strategy: make a target destination that is smaller by just the right amount,
    # insert a 1 at the right point, then convolve
    pre_convolve_shape = tuple(d - s + 1 for s, d in zip(submatrix_shape, destination_shape))
    dest = T.zeros(pre_convolve_shape)
    submatrix_offset = tuple((dim - 1) // 2 for dim in submatrix_shape)
    dest_with_one = T.set_subtensor(dest[center_location[0] + offset[0] - submatrix_offset[0],
                                         center_location[1] + offset[1] - submatrix_offset[1]], 1.0)
    convolved = theano.tensor.signal.conv.conv2d(dest_with_one, submatrix, border_mode="full")
    convolved_shape_fixed = convolved.reshape(destination_shape)
    return convolved_shape_fixed

class Decoder(object):

    def __init__(self, code, locations, filters, image_shape, filter_shape, numpy_rng=None):
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState()

        # TODO: This is copied and pasted directly from the Encoder. Make it DRY. How to do so cleanly?

        num_filters = filter_shape[0]
        individual_filter_shape = filter_shape[1:]

        # TODO: This is completely unmotivated. It comes purely from the observation that
        # initializing W_bound = 1/100 works pretty well for a 7x7 filter, and 100 ~= 2 * 7 * 7. :/
        fan_in = numpy.prod(individual_filter_shape)
        filters_elem_bound = 1 / (2 * fan_in) 
        filters_value = numpy.asarray(numpy_rng.uniform(low=-filters_elem_bound, high=filters_elem_bound, size=filter_shape), dtype=floatX)

        self.filters = theano.shared(name="decoder_filters", value=filters_value)

        convolved_dims = theano.tensor.nnet.conv.ConvOp.getOutputShape(image_shape, individual_filter_shape)

        # calculate the offset induced by the fact that the convolved row/col offsets are smaller because
        # the convolution changed the matrix size
        conv_offset = tuple((convolved_dim - image_dim) // 2 for convolved_dim, image_dim in zip(convolved_dims, image_shape))

        def accumulate(code_elem, location, template, accumulated_so_far):
            filter_to_add = code_elem * template
            addend = zeros_with_submatrix(filter_to_add, location, conv_offset, individual_filter_shape, image_shape)
            return accumulated_so_far + addend

        scan_result, scan_updates = theano.scan(fn=accumulate,
                                                outputs_info=T.zeros(image_shape, dtype=floatX),
                                                sequences=[code, locations, self.filters])

        self.decoded = scan_result[-1]
        self.params = [self.filters]

    def decoder_energy(self, wrt_image):
        return ((self.decoded - wrt_image) ** 2).sum()

def gradient_updates(score, params, learning_rate):
    gradient_params = [T.grad(score, param) for param in params]

    updates = {}
    for param, gradient_params in zip(params, gradient_params):
        updates[param] = param - learning_rate * gradient_params

    return updates

def train(training_data,
          output_directory=None,
          filter_save_frequency=None):

    image_variable = T.matrix("image")
    filter_shape = (4, 7, 7) # num filters, r, c
    image_shape = (17, 17) # r, c
    encoder = Encoder(image_variable, filter_shape, image_shape)
    encode = theano.function(inputs=[image_variable], outputs=[encoder.code, encoder.locations])

    features = T.vector("features")
    locations = T.imatrix("locations")

    ideal_weights = theano.shared(name="ideal_weights", value=numpy.zeros((4,), dtype=floatX))

    decoder_for_ideal_weights = Decoder(ideal_weights, locations, encoder.filters, image_shape, filter_shape)
    decoder_energy = decoder_for_ideal_weights.decoder_energy(image_variable)
    encoder_energy = encoder.encoder_energy(ideal_weights)

    L1_code_penalty = abs(ideal_weights).sum()

    encoder_energy_weight = 5
    decoder_energy_weight = 1
    L1_code_penalty_weight = 2

    total_energy = encoder_energy_weight * encoder_energy + \
                   decoder_energy_weight * decoder_energy + \
                   L1_code_penalty_weight * L1_code_penalty

    energy_params = [ideal_weights]
    step_energy = theano.function(inputs=[image_variable, locations],
                                  outputs=total_energy,
                                  updates=gradient_updates(total_energy, energy_params, learning_rate=0.05))

    decoder_params = decoder_for_ideal_weights.params
    step_decoder = theano.function(inputs=[image_variable, locations],
                                   outputs=decoder_energy,
                                   updates=gradient_updates(decoder_energy, decoder_params, learning_rate=0.01))

    encoder_params = encoder.params
    step_encoder = theano.function(inputs=[image_variable],
                                   outputs=encoder_energy,
                                   updates=gradient_updates(encoder_energy, encoder_params, learning_rate=0.01))

    if output_directory is not None and not os.path.isdir(output_directory):
        print "Creating output directory {d}".format(d=output_directory)
        os.makedirs(output_directory)

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

        decoder_energy = step_decoder(image, encoded_locations)
        encoder_energy = step_encoder(image)

        if output_directory is not None and image_index % filter_save_frequency == 0:
            print "Saving filters at image index {i}".format(i=image_index)
            image = PIL.Image.fromarray(tile_raster_images(X=numpy.r_[encoder.filters.get_value(), decoder_for_ideal_weights.filters.get_value()],
                                        img_shape=(7, 7),
                                        tile_shape=(2, 4), 
                                        tile_spacing=(1, 1)))
            image_filename = os.path.join(output_directory, "filters_{i}.png".format(i=image_index))
            image.save(image_filename)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input_file",
                        help="input"
                       )
    parser.add_argument("-o", "--output-directory",
                        default="filters",
                        help="directory in which to write the filters during training; if None, filters will not be written"
                       )
    parser.add_argument("-f", "--filter-save-frequency",
                        type=int,
                        default=200,
                        help="save the filters to the output directory every n samples"
                       )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        training_data = pickle.load(f)

    train(training_data, output_directory=args.output_directory, filter_save_frequency=args.filter_save_frequency)


if __name__ == '__main__':
    sys.exit(main())
