import graphsurgeon as gs

output_name = ['NMS']
dims = [3,300,300]
layout = 7

def add_plugin(graph):
    # remove all assert nodes
    # https://www.tensorflow.org/api_docs/python/tf/debugging/Assert
    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    # Remove superfluous identity nodes from the graph
    # https://www.tensorflow.org/api_docs/python/tf/identity
    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)

    # Create TensorFlow NodeDef for an input placeholder
    # https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder
    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        shape=[1, 3, 300, 300]
    )

    # Create TensorFlow NodeDef for a GridAnchor_TRT operation:
    # https://github.com/NVIDIA/TensorRT/tree/master/plugin/gridAnchorPlugin
    PriorBox = gs.create_plugin_node(
        name="GridAnchor",
        op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1,0.1,0.2,0.2],
        featureMapShapes=[19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    # Create TensorFlow NodeDef for a NMS_TRT operation:
    # https://github.com/NVIDIA/TensorRT/tree/master/plugin/nmsPlugin
    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=1e-8,
        nmsThreshold=0.6,
        topK=100,
        keepTopK=100,
        numClasses=91,
        inputOrder=[1, 0, 2],
        confSigmoid=1,
        isNormalized=1
    )

    # Create ConcatV2 NodeDef
    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        axis=2
    )

    # Create FlattenConcat NodeDef using custom plugin in libflattenconcat
    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
    )

    # Create FlattenConcat NodeDef using custom plugin in libflattenconcat
    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
    )

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "ToFloat": Input,
        "image_tensor": Input,
        "Concatenate": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf # one input to NMS
    }

    # replace the operations/subgraphs in the namespaces according to the namespace_plugin_map
    graph.collapse_namespaces(namespace_plugin_map)
    # remove all output nodes
    graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    # The whole `Postprocessor` namespace has been replaced with a NMS_TRT op.
    # Its NodeDef has the following inputs: ['Input', 'concat_box_conf', 'Squeeze', 'concat_priorbox']
    # The 'Input' is the result of replacing the 'Preprocessor' which also was connected to the 'Postprocessor'
    # The input of 'Squeeze' is 'concat', which in turn was replaced by 'concat_box_loc'
    # Since NMS_TRT just has 3 inputs (loc_data, conf_data, prior_data), 'Input' must be removed.
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")

    return graph
