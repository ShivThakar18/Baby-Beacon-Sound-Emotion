import tensorflow as tf
import tf2onnx

# Load the Keras model
model = tf.keras.models.load_model("C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\model\\augmented_baby_emotions_model.h5")  # or "your_model.keras"

# Convert to ONNX
onnx_model_path = "C:\\Users\\shivt\\Documents\\GitHub\\Baby-Beacon-Sound-Emotion\\model\\augmented_baby_emotions_model.onnx"  # Ensure the extension is .onnx

# Convert model
spec = (tf.TensorSpec((None, 40), tf.float32),)  # Adjust input shape to match your MFCC features
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the ONNX model
tf2onnx.save_model(onnx_model, onnx_model_path)

print(f"Model saved to {onnx_model_path}")
