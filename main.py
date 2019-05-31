import keras
import numpy as np
import os
import tensorflow as tf
from CustomLayer import NetVLADLayer
from keras import backend as K


import TerminalColors
tcol = TerminalColors.bcolors()

def load_basic_model( ):
    K.set_learning_phase(0)


    input_img = keras.layers.Input( shape=(60, 80, 256 ) )
    out = NetVLADLayer(num_clusters = 16)( input_img )
    model = keras.models.Model( inputs=input_img, outputs=out )


    model.summary()
    return model


def write_kerasmodel_as_tensorflow_pb( model, LOG_DIR, output_model_name='output_model.pb' ):
    """ Takes as input a keras.models.Model() and writes out
        Tensorflow proto-binary.
    """
    print tcol.HEADER,'[write_kerasmodel_as_tensorflow_pb] Start', tcol.ENDC

    import tensorflow as tf
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    K.set_learning_phase(0)
    sess = K.get_session()



    # Make const
    print 'Make Computation Graph as Constant and Prune unnecessary stuff from it'
    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        [node.op.name for node in model.outputs])
    constant_graph = tf.graph_util.remove_training_nodes(constant_graph)


    #--- convert Switch --> Identity
    # I am doing this because TensorRT cannot process Switch operations.
    # # https://github.com/tensorflow/tensorflow/issues/8404#issuecomment-297469468
    # for node in constant_graph.node:
    #     if node.op == "Switch":
    #         node.op = "Identity"
    #         del node.input[1]
    # # END

    # Write .pb
    # output_model_name = 'output_model.pb'
    print tcol.OKGREEN, 'Write ', output_model_name, tcol.ENDC
    print 'model.outputs=', [node.op.name for node in model.outputs]
    graph_io.write_graph(constant_graph, LOG_DIR, output_model_name,
                     as_text=False)
    print tcol.HEADER, '[write_kerasmodel_as_tensorflow_pb] Done', tcol.ENDC


    # Write .pbtxt (for viz only)
    output_model_pbtxt_name = output_model_name+'.pbtxt' #'output_model.pbtxt'
    print tcol.OKGREEN, 'Write ', output_model_pbtxt_name, tcol.ENDC
    tf.train.write_graph(constant_graph, LOG_DIR,
                      output_model_pbtxt_name, as_text=True)

    # Write model.summary to file (to get info on input and output shapes)
    output_modelsummary_fname = LOG_DIR+'/'+output_model_name + '.modelsummary.log'
    print tcol.OKGREEN, 'Write ', output_modelsummary_fname, tcol.ENDC
    with open(output_modelsummary_fname,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))




# def verify_generated_uff_with_tensorrt_uffparser( ufffilename, uffinput, uffinput_dims, uff_output ):
def verify_generated_uff_with_tensorrt_uffparser( ufffilename ):
    """ Loads the UFF file with TensorRT (py). """
    assert os.path.isfile( ufffilename ), "ufffilename="+ ufffilename+ ' doesnt exist'
    import tensorrt as trt

    print tcol.HEADER, '[verify_generated_uff_with_tensorrt_uffparser] TensorRT version=', trt.__version__, tcol.ENDC

    try:
        uffinput = "input_1"
        # uffinput_dims = (3,240,320)
        uffinput_dims = (256, 80,60)
        # uffoutput = "conv_pw_5_relu/Relu6"
        # uffoutput = "net_vlad_layer_1/l2_normalize_1"
        uffoutput = "net_vlad_layer_1/add_1"
        # uffoutput = "net_vlad_layer_1/Reshape_1"

        TRT_LOGGER = trt.Logger( trt.Logger.WARNING)
        with trt.Builder( TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
            print 'ufffilename=', str( ufffilename)
            print 'uffinput=', str( uffinput), '\t', 'uffinput_dims=', str( uffinput_dims)
            print 'uffoutput=', str( uffoutput)
            parser.register_input( uffinput, uffinput_dims  )
            parser.register_output( uffoutput )
            parser.parse( ufffilename, network )
            pass

        print tcol.OKGREEN, '[verify_generated_uff_with_tensorrt_uffparser] Verified.....!', tcol.ENDC
    except:
        print tcol.FAIL, '[verify_generated_uff_with_tensorrt_uffparser] UFF file=', ufffilename, ' with uffinput=', uffinput , ' uffoutput=', uffoutput , ' cannot be parsed.'




if __name__ == '__main__':
    model = load_basic_model()
    write_kerasmodel_as_tensorflow_pb( model, './',  output_model_name='output_model.pb' )

    print '-----------'
    print 'NOW on your bash prompt do:\n\t$convert-to-uff output_model.pb'
    print '-----------'
    # quit()

    verify_generated_uff_with_tensorrt_uffparser( './output_model.uff' )
