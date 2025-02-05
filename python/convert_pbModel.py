from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import tf2onnx
from tensorflow.keras.layers import InputLayer

PATH = "C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\model\\"
# Load the frozen model
model_path = "C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\model\\model.h5"

def convertH5_to_ONNX(model_path):

    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Convert to ONNX
    onnx_model_path = PATH+"model.onnx"
    spec = (tf.TensorSpec(model.input_shape, tf.float32),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)

    # Save the ONNX model
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"ONNX model saved as {onnx_model_path}")


def convertH5_to_Frozen(model_path,output_path):
    
    # Load Keras model
    model = tf.keras.models.load_model(model_path)

    # Ensure model is built by calling it with dummy input
    dummy_input = tf.ones((1,) + model.input_shape[1:])  # Create dummy input tensor
    model(dummy_input)  # Run a forward pass

    # Convert model to a ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.input_shape, model.input[0].dtype))

    # Convert variables to constants
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # Save frozen graph
    tf.io.write_graph(frozen_func.graph, ".", output_path, as_text=False)

    print(f"Frozen model saved as {output_path}")

def convertSavedModel_to_Frozen():
    pb_model = PATH+"model_saved\\"

    loaded_model = tf.saved_model.load(pb_model)
    print("pb Model loaded successfully")

    concerte_func = loaded_model.signatures['serving_default']
    frozen_func = convert_variables_to_constants_v2(concerte_func)

    tf.io.write_graph(frozen_func.graph,".",PATH+"frozen_model.pb", as_text=False)
    print("Frozen model saved as ../model/frozen_model.pb")

    frozen_model = (PATH+"frozen_model.pb")
    print("Frozen Model loaded successfully")

# Load the model
model = tf.keras.models.load_model(PATH+"model.h5")

# Replace 'batch_shape' with 'input_shape' in the config
config = {'batch_shape': [None, 40], 'dtype': 'float32', 'sparse': False, 'name': 'input_layer'}
config['input_shape'] = config.pop('batch_shape')[1:]

# Reconstruct the InputLayer
input_layer = InputLayer.from_config(config)
print(input_layer)


# Get the ConcreteFunction from the Keras model
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Convert variables to constants
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

# Export the frozen graph
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name=PATH+"frozen_graph.pb",
                  as_text=False)
