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
        filters_elem_bound = 1 / fan_in
        filters_value = numpy.asarray(numpy_rng.uniform(low=0, high=filters_elem_bound, size=filter_shape), dtype=floatX)

        self.filters = theano.shared(name="encoder_filters", value=filters_value)

        convolved = theano.tensor.signal.conv.conv2d(image_variable, self.filters)
        convolved_rows, convolved_cols = theano.tensor.nnet.conv.ConvOp.getOutputShape(image_shape, individual_filter_shape)

        # rasterize each convolved image, since max_and_argmax doesn't accept multiple axes
        convolved_rasterized = convolved.reshape((num_filters, -1))
        raw_code, argmax_raveled = T.max_and_argmax(convolved_rasterized, axis=-1)

        self.code = T.tanh(raw_code)

        # now unravel the argmax value to undo the rasterization
        argmax_row = argmax_raveled // convolved_rows
        argmax_col = argmax_raveled % convolved_rows
        locations_upcast = T.stack(argmax_row, argmax_col).T

        self.locations = T.cast(locations_upcast, "int32") # the // and % upcast from int32 to int64; cast back down

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
    dest_with_one = T.set_subtensor(dest[center_location[0] + offset[0] + submatrix_offset[0],
                                         center_location[1] + offset[1] + submatrix_offset[1]], 1.0)
    convolved = theano.tensor.signal.conv.conv2d(dest_with_one, submatrix, border_mode="full")
    convolved_shape_fixed = convolved.reshape(destination_shape)
    return convolved_shape_fixed

class Decoder(object):

    def __init__(self, code, locations, image_shape, filter_shape, numpy_rng=None):
        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState()

        # TODO: This is copied and pasted directly from the Encoder. Make it DRY. How to do so cleanly?

        num_filters = filter_shape[0]
        individual_filter_shape = filter_shape[1:]

        # TODO: This is completely unmotivated. It comes purely from the observation that
        # initializing W_bound = 1/100 works pretty well for a 7x7 filter, and 100 ~= 2 * 7 * 7. :/
        fan_in = numpy.prod(individual_filter_shape)
        filters_elem_bound = 1 / fan_in
        filters_value = numpy.asarray(numpy_rng.uniform(low=0, high=filters_elem_bound, size=filter_shape), dtype=floatX)

        self.filters = theano.shared(name="decoder_filters", value=filters_value)

        convolved_dims = theano.tensor.nnet.conv.ConvOp.getOutputShape(image_shape, individual_filter_shape)

        # calculate the offset induced by the fact that the convolved row/col offsets are smaller because
        # the convolution changed the matrix size
        conv_offset = tuple((convolved_dim - image_dim) // 2 for convolved_dim, image_dim in zip(convolved_dims, image_shape))

        def accumulate(code_elem, location, template, accumulated_so_far):
            filter_to_add = code_elem * template
            # i'm not entirely clear on why this flipping is needed; i believe offhand
            # that it is due to a side-effect of the convolution.
            # everything trains up ok without it; this just makes the trained encoder and
            # decoder filters look similar, rather than being flipped relative to each other,
            # which is very satisfying. :)
            filter_flipped = filter_to_add[::-1,::-1]
            addend = zeros_with_submatrix(filter_flipped, location, conv_offset, individual_filter_shape, image_shape)
            return accumulated_so_far + addend

        scan_result, scan_updates = theano.scan(fn=accumulate,
                                                outputs_info=T.zeros(image_shape, dtype=floatX),
                                                sequences=[code, locations, self.filters])

        self.decoded = scan_result[-1]

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
          save_frequency=None,
          num_filters=None,
          image_shape=None):

    numpy_rng = numpy.random.RandomState(8912373)

    image_variable = T.matrix("image")
    individual_filter_shape = (7, 7)
    filter_shape = (num_filters, individual_filter_shape[0], individual_filter_shape[1]) # num filters, r, c
    encoder = Encoder(image_variable, filter_shape, image_shape, numpy_rng=numpy_rng)
    encode = theano.function(inputs=[image_variable], outputs=[encoder.code, encoder.locations])

    locations_variable = T.imatrix("locations")

    optimal_code = theano.shared(name="optimal_code", value=numpy.zeros((num_filters,), dtype=floatX))

    # to create a decoder using the encoder's produced code, you'd do something similar, but use
    # encoder.code instead of optimal_code
    decoder_using_optimal_code = Decoder(optimal_code, locations_variable, image_shape, filter_shape, numpy_rng=numpy_rng)

    decoder_energy = decoder_using_optimal_code.decoder_energy(image_variable) # compare with original input image
    encoder_energy = encoder.encoder_energy(optimal_code) # compare against a calculated optimal code
    L1_code_penalty = abs(optimal_code).sum()

    encoder_energy_weight = 5
    decoder_energy_weight = 1
    L1_code_penalty_weight = 2

    total_energy = encoder_energy_weight * encoder_energy + \
                   decoder_energy_weight * decoder_energy + \
                   L1_code_penalty_weight * L1_code_penalty

    energy_params = [optimal_code]
    step_energy = theano.function(inputs=[image_variable, locations_variable],
                                  outputs=total_energy,
                                  updates=gradient_updates(total_energy, energy_params, learning_rate=0.05))

    decoder_params = [decoder_using_optimal_code.filters]
    step_decoder = theano.function(inputs=[image_variable, locations_variable],
                                   outputs=None,
                                   updates=gradient_updates(decoder_energy, decoder_params, learning_rate=0.01))

    encoder_params = [encoder.filters]
    step_encoder = theano.function(inputs=[image_variable],
                                   outputs=None,
                                   updates=gradient_updates(encoder_energy, encoder_params, learning_rate=0.01))

    if output_directory is not None and not os.path.isdir(output_directory):
        print "Creating output directory {d}".format(d=output_directory)
        os.makedirs(output_directory)

    summed_energy_since_last_print = 0

    for image_index, image in enumerate(training_data):

        if image_index % save_frequency == 0:
            # print "Saving filters at image {i}".format(i=image_index)
            print "Average total energy at image {i} is {e:.2f}".format(i=image_index,
                                                                        e=summed_energy_since_last_print / save_frequency)
            summed_energy_since_last_print = 0
            encoder_filters = encoder.filters.get_value()
            decoder_filters = decoder_using_optimal_code.filters.get_value()
            # print "Encoder filter min {n}, max {x}".format(n=numpy.min(encoder_filters), x=numpy.max(encoder_filters))
            # print "Decoder filter min {n}, max {x}".format(n=numpy.min(decoder_filters), x=numpy.max(decoder_filters))
            if output_directory is not None:
                # make the output roughly square; encoders will be on the top half, decoders on the bottom
                # chop off the filters so that each set of filters perfectly fills its half-square
                rows_per_coder = int(math.floor(math.sqrt(num_filters / 2)))
                display_filters_per_coder = (num_filters // rows_per_coder) * rows_per_coder
                combined_display_filters = numpy.r_[encoder_filters[:display_filters_per_coder],
                                                    decoder_filters[:display_filters_per_coder]]
                filters_image = PIL.Image.fromarray(tile_raster_images(X=combined_display_filters,
                                                    img_shape=individual_filter_shape,
                                                    tile_shape=(2 * rows_per_coder, num_filters // rows_per_coder),
                                                    tile_spacing=(1, 1)))
                image_filename = os.path.join(output_directory, "filters_{i}.png".format(i=image_index))
                filters_image.save(image_filename)

        encoded_code, encoded_locations = encode(image)

        # copy the actual code to the optimal code, to be optimized
        optimal_code.set_value(encoded_code)

        # hack: search for optimal code only so long as we're making immediate progress
        # todo: incorporate patience here
        prior_energy = float("inf")
        keep_going = True
        while keep_going:
            current_energy = step_energy(image, encoded_locations)
            keep_going = current_energy < prior_energy
            prior_energy = current_energy

        summed_energy_since_last_print += current_energy

        # found the optimal code; now take a single gradient descent step for decoder and encoder
        step_decoder(image, encoded_locations)
        step_encoder(image)

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
    parser.add_argument("-s", "--save-frequency",
                        type=int,
                        default=200,
                        help="print status info and save filters every n samples"
                       )
    parser.add_argument("-f", "--num_filters",
                        type=int,
                        default=50,
                        help="number of filters to train"
                       )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        training_data = pickle.load(f)

    train(training_data,
          output_directory=args.output_directory,
          save_frequency=args.save_frequency,
          num_filters=args.num_filters,
          image_shape=training_data[0].shape)


if __name__ == '__main__':
    sys.exit(main())
