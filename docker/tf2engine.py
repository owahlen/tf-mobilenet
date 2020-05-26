#!/usr/bin/env python3

import os
import sys
import ctypes

import argparse
import uff
import tensorrt as trt
import graphsurgeon as gs
from config import model_ssd_mobilenet_v2 as model

ctypes.CDLL("lib/libflattenconcat.so")


def write_uff_file(pb_file, uff_file):
    dynamic_graph = model.add_plugin(gs.DynamicGraph(pb_file))
    uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), model.output_name, output_filename=uff_file)


def write_engine_file(uff_file, engine_file):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    runtime = trt.Runtime(TRT_LOGGER)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', model.dims)
        parser.register_output('MarkOutput_0')
        parser.parse(uff_file, network)
        engine = builder.build_cuda_engine(network)

        # serialize engine into engine_file
        buf = engine.serialize()
        with open(engine_file, 'wb') as f:
            f.write(buf)


def main(pb_file):
    if not os.path.isfile(pb_file):
        sys.exit("Unable to find input file: {}".format(pb_file))

    base_file = str(os.path.splitext(pb_file))
    uff_file = base_file + '.uff'
    engine_file = base_file + '.engine'

    write_uff_file(pb_file, uff_file)
    write_engine_file(uff_file, engine_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert frozen tensorflow model to TensorRT engine')
    parser.add_argument('pb_file', help='name of tensorflow "*.pb" file (e.g. frozen_graph.pb)')
    args = parser.parse_args()

    main(args.pb_file)
