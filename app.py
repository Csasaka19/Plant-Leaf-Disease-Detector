import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
import os

# Set page config first!
st.set_page_config(layout="wide") # Use wider layout

# --- Configuration ---
MODEL_PATH = './models/plant_disease_model.h5'
CLASS_INDEX_PATH = './index/class_indices.json'
EXAMPLE_IMAGE_DIR = './example_images'
IMG_WIDTH, IMG_HEIGHT = 128, 128 # Should match the dimensions used during training

# --- Load Model and Class Indices ---
@st.cache_resource # Cache the model loading
def load_prediction_model():
    """Loads the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        st.stop()
    try:
        # Load the model without compiling for prediction
        loaded_model = load_model(MODEL_PATH, compile=False)
        loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data # Cache the class indices loading
def load_class_indices():
    """Loads the class indices from JSON file."""
    if not os.path.exists(CLASS_INDEX_PATH):
        st.error(f"Error: Class index file not found at {CLASS_INDEX_PATH}")
        st.stop()
    try:
        with open(CLASS_INDEX_PATH, 'r') as f:
            class_indices = json.load(f)
        # Create inverse mapping (index -> class name)
        index_to_class = {v: k for k, v in class_indices.items()}
        # Store class names in a list in the correct order (matching the indices)
        class_names_ordered = [None] * len(index_to_class)
        for idx, name in index_to_class.items():
            class_names_ordered[idx] = name
        if None in class_names_ordered:
             st.error("Error: Class index mapping is incomplete.")
             st.stop()
        return index_to_class, class_names_ordered
    except Exception as e:
        st.error(f"Error loading class indices: {e}")
        st.stop()

model = load_prediction_model()
index_to_class, class_names_ordered = load_class_indices()
num_classes = len(index_to_class)

# --- Image Preprocessing ---
def preprocess_image(image_pil):
    """Preprocesses the uploaded PIL image for the model."""
    try:
        # Resize image
        img = image_pil.resize((IMG_WIDTH, IMG_HEIGHT))
        # Convert to RGB if it's RGBA (like some PNGs)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Convert to NumPy array
        img_array = np.array(img)
        # Normalize pixel values ( crucial!)
        img_array = img_array / 255.0
        # Expand dimensions to create a batch of 1
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- Prediction Function ---
def predict(image_data):
    """Runs prediction on preprocessed image data."""
    try:
        prediction = model.predict(image_data)
        predicted_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]
        return prediction, predicted_index, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

# --- Initialize Session State ---
if 'image_to_predict' not in st.session_state:
    st.session_state.image_to_predict = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'display_image' not in st.session_state:
    st.session_state.display_image = None

# --- Streamlit UI ---
st.title("🌿 Plant Leaf Disease Detector")
st.write("Upload an image or select an example below to classify if a plant leaf is Healthy, Powdery, or Rust.")

# --- Sidebar --- #
st.sidebar.header("Options")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)", min_value=0, max_value=100, value=50, step=5
) / 100.0 # Convert to 0.0-1.0 scale

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Convolutional Neural Network (CNN) trained on plant leaf images "
    "to detect common diseases (Healthy, Powdery, Rust)."
)

# --- Main Area --- #
col1, col2 = st.columns([2, 1]) # Define columns for layout

with col1:
    st.subheader("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="fileuploader")

    if uploaded_file is not None:
        # If a new file is uploaded, process it
        if st.session_state.get('uploaded_file_name') != uploaded_file.name:
            st.session_state.image_to_predict = uploaded_file
            st.session_state.uploaded_file_name = uploaded_file.name # Store filename to detect changes
            st.session_state.prediction_result = None # Reset prediction
            st.session_state.display_image = Image.open(uploaded_file)

    st.subheader("Or Select an Example")
    example_files = {}
    if os.path.isdir(EXAMPLE_IMAGE_DIR):
        try:
             # Dynamically find example images based on expected class names
            for class_name in class_names_ordered:
                # Try common extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    potential_path = os.path.join(EXAMPLE_IMAGE_DIR, f"{class_name}{ext}")
                    if os.path.exists(potential_path):
                         example_files[class_name] = potential_path
                         break # Found one, move to next class

        except Exception as e:
             st.warning(f"Could not load example images: {e}")
    else:
        st.warning(f"Example image directory not found: {EXAMPLE_IMAGE_DIR}")

    if example_files:
        example_cols = st.columns(len(example_files))
        for i, (class_name, img_path) in enumerate(example_files.items()):
             with example_cols[i]:
                 try:
                     img = Image.open(img_path)
                     st.image(img, caption=f"Example: {class_name}", width=150)
                     if st.button(f"Predict {class_name}", key=f"btn_{class_name}"):
                         st.session_state.image_to_predict = img_path # Store path
                         st.session_state.prediction_result = None # Reset prediction
                         st.session_state.display_image = img
                         st.session_state.uploaded_file_name = None # Clear uploaded file state
                 except Exception as e:
                    st.error(f"Failed to load {class_name} example: {e}")
    else:
         st.info("Place example images (e.g., Healthy.jpg, Powdery.png) in the 'example_images' directory to enable this feature.")


with col2:
    st.subheader("Prediction Result")
    # Perform prediction if an image (uploaded or example) is ready
    if st.session_state.image_to_predict is not None and st.session_state.prediction_result is None:
        try:
            if isinstance(st.session_state.image_to_predict, str): # Example image path
                image_pil = Image.open(st.session_state.image_to_predict)
            else: # Uploaded file
                image_pil = Image.open(st.session_state.image_to_predict)

            # Preprocess and Predict
            processed_image = preprocess_image(image_pil)
            if processed_image is not None:
                raw_prediction, predicted_index, confidence = predict(processed_image)
                # Store results in session state
                st.session_state.prediction_result = {
                    "raw": raw_prediction,
                    "index": predicted_index,
                    "confidence": confidence
                }
            else:
                st.session_state.image_to_predict = None # Clear if preprocessing failed

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.session_state.image_to_predict = None # Clear on error
            st.session_state.prediction_result = None

    # Display the image that was processed
    if st.session_state.display_image is not None:
         st.image(st.session_state.display_image, caption='Image for Prediction', use_container_width=True)
         st.write("")

    # Display the prediction result from session state
    if st.session_state.prediction_result is not None:
        confidence = st.session_state.prediction_result['confidence']
        predicted_index = st.session_state.prediction_result['index']
        raw_prediction = st.session_state.prediction_result['raw']

        if confidence >= confidence_threshold:
            if predicted_index in index_to_class:
                predicted_class_name = index_to_class[predicted_index]
                st.success(f"Prediction: **{predicted_class_name}**")
                st.info(f"Confidence: **{confidence*100:.2f}%**")
            else:
                st.error(f"Error: Predicted index {predicted_index} not found in class mapping.")
        else:
            st.warning(f"Prediction Uncertain (Confidence {confidence*100:.2f}% is below threshold of {confidence_threshold*100:.0f}%)")

        # Optional: Display raw prediction probabilities
        with st.expander("Show Prediction Probabilities"):
            if raw_prediction is not None:
                probabilities = {name: f"{prob*100:.2f}%" for name, prob in zip(class_names_ordered, raw_prediction[0])}
                st.write(probabilities)
            else:
                st.write("Probabilities not available.")

    elif st.session_state.image_to_predict is not None:
         # If image is set but result is None (likely due to error handled elsewhere or ongoing processing)
         st.info("Processing image...")
    else:
        st.info("Upload an image or select an example to see the prediction.") 