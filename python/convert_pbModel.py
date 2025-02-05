from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import tf2onnx

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

# Load the graph
def load_frozen_model(model_path):
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

#convertH5_to_ONNX(model_path)
# Load the .h5 model
model = load_model(model_path, compile=False)

# Manually rebuild the model
inputs = tf.keras.Input(shape=(40,))  # Adjust shape as needed
outputs = model(inputs)
new_model = tf.keras.Model(inputs, outputs)

# Save the new model
new_model.save(PATH+"rebuilt_model.h5")

# Convert to ONNX
import tf2onnx
onnx_model_path = PATH+"model.onnx"
spec = (tf.TensorSpec(new_model.input_shape, tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(new_model, input_signature=spec)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"ONNX model saved as {onnx_model_path}")