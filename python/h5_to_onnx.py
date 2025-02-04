import tensorflow as tf

model = tf.keras.models.load_model("/home/ghosttt/Baby-Beacon-Sound-Emotion/model/augmented_baby_emotions_model.h5")
model.save("augmented_baby_emotions_model.onnx",save_format="tf")

