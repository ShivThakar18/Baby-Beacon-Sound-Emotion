import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

PATH = "C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\model\\"
pb_model = PATH+"model_saved\\"

loaded_model = tf.saved_model.load(pb_model)
print("pb Model loaded successfully")

concerte_func = loaded_model.signatures['serving_default']
frozen_func = convert_variables_to_constants_v2(concerte_func)

tf.io.write_graph(frozen_func.graph,".",PATH+"frozen_model.pb", as_text=False)
print("Frozen model saved as ../model/frozen_model.pb")