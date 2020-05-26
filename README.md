# TF-MobileNet
This repository illustrates how to apply transfer learning with TensorFlow 2.
A convolutional network is trained and prepared for inference using TensorRT.

The repository contains the following notebooks:

1. train.ipynb: Instantiates, tunes, trains and saves a MobileNetV2 graph
2. freeze.ipynb: Freezes and saves the graph
3. show_graph.py: Creates logs of the frozen graph to view it in `tensorboard`
