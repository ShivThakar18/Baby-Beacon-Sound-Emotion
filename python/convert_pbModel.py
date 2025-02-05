import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import keras 
import numpy as np

PATH = "C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\model\\"
# Load the frozen model
model_path = PATH+"frozen_model.pb"

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

# Load the model
graph = load_frozen_model(model_path)

# Print available operations (useful for debugging)
for op in graph.get_operations():
    print(op.name)

print("Frozen model loaded successfully!")
