import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Embedded custom CNN Classifier", page_icon="📷")

st.title("Air Pollution Mini-Model (ESP32-CAM)")
st.write("Upload an image to test the 398 KB INT8 Quantized Custom CNN.")

# Load Labels
@st.cache_resource
def load_labels():
    with open('labels.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]

# Load TFLite Model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path='model_quant.tflite')
    interpreter.allocate_tensors()
    return interpreter

labels = load_labels()
interpreter = load_model()

# Get input and output details
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Classifying...")
    
    # Preprocess
    img = image.resize((96, 96))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Quantize input (Convert Float32 to INT8)
    scale, zero_point = input_details['quantization']
    if scale != 0.0:
        img_array = (img_array / scale) + zero_point
    
    # Cast to INT8
    img_array = img_array.astype(input_details['dtype'])
    
    # Run Inference
    interpreter.set_tensor(input_details['index'], img_array)
    interpreter.invoke()
    
    # Get Output
    output_data = interpreter.get_tensor(output_details['index'])[0]
    
    # Dequantize Output (Convert INT8 back to Float32 probabilities)
    out_scale, out_zero_point = output_details['quantization']
    if out_scale != 0.0:
        output_data = (output_data.astype(np.float32) - out_zero_point) * out_scale
        
    scores = tf.nn.softmax(output_data).numpy() # Just in case it's logits, but model already has softmax. But wait!

    # Actually, the TFLite output comes from the Dense(num_classes, activation='softmax') layer!
    # So `output_data` IS the probability distribution, just quantized to INT8.
    
    prediction_idx = np.argmax(output_data)
    prediction_label = labels[prediction_idx]
    confidence = output_data[prediction_idx] * 100
    
    st.success(f"Prediction: **{prediction_label}** ({confidence:.2f}%)")
    
    with st.expander("Show detailed probabilities"):
        for i, label in enumerate(labels):
            st.write(f"- {label}: {output_data[i]*100:.2f}%")
