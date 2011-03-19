#!/usr/bin/env python

from __future__ import division

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import random
import sys

import numpy

PATCH_SIZE = 17
LINE_LENGTH = 7
NUM_LINES_PER_PATCH = 2
LINE_CENTER_SIZE = 5
LINE_PERIPHERY_SIZE = (PATCH_SIZE - LINE_CENTER_SIZE) // 2

HALF_LINE_LENGTH = (LINE_LENGTH - 1) // 2

CENTERED_ONE = numpy.zeros(LINE_LENGTH, dtype=numpy.float32)
CENTERED_ONE[HALF_LINE_LENGTH] = 1.0

LINE_TYPES = [numpy.eye(LINE_LENGTH, dtype=numpy.float32), # UL/BR diagonal
              numpy.rot90(numpy.eye(LINE_LENGTH, dtype=numpy.float32)), # BL/TR diagonal
              numpy.outer(CENTERED_ONE, numpy.ones(LINE_LENGTH, dtype=numpy.float32)), # horizontal, centered
              numpy.outer(numpy.ones(LINE_LENGTH, dtype=numpy.float32), CENTERED_ONE), # verical, centered
             ]

def generate_sample_patch():
    patch = numpy.zeros((PATCH_SIZE, PATCH_SIZE), dtype=numpy.float32)

    lines = random.sample(LINE_TYPES, NUM_LINES_PER_PATCH)

    for line in lines:
        line_center = [random.randint(LINE_PERIPHERY_SIZE, LINE_PERIPHERY_SIZE + LINE_CENTER_SIZE) for dim in xrange(2)]
        patch_area_slices = [slice(dim_center - HALF_LINE_LENGTH, dim_center + HALF_LINE_LENGTH + 1) for dim_center in line_center]
        patch[patch_area_slices] = numpy.maximum(patch[patch_area_slices], line)
    return patch

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Generate toy examples for reproducing Ranzato et al 07.")
    parser.add_argument("output_filename",
                        help="output filename"
                       )
    parser.add_argument("-n", "--num-samples",
                        type=int,
                        required=True,
                        help="number of samples to produce"
                       )
    args = parser.parse_args()

    sample_patches = numpy.asarray([generate_sample_patch() for sample_num in xrange(args.num_samples)])
    with open(args.output_filename, "wb") as outfile:
        pickle.dump(sample_patches, outfile, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    sys.exit(main())
