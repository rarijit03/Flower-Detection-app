import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

# Define the path to your model
MODEL_PATH = 'E:\\Webel\\densenet121_best_model.keras'

# Check if the model file exists and load it
if not os.path.exists(MODEL_PATH):
    st.error(f"The model file does not exist at the specified path: {MODEL_PATH}")
else:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Define image size and class indices
IMAGE_SIZE = (224, 224)
class_indices = {'Dahlia': 0, 'Garbera': 1, 'Hibiscus': 2, 'Jasmine': 3, 'Lotus': 4, 
                 'Marigold': 5, 'Rose': 6, 'Sunflower': 7, 'Tulip': 8, 'Rajnigandha': 9}
class_names = list(class_indices.keys())

def load_and_preprocess_image(image):
    try:
        img = image.resize(IMAGE_SIZE)
        img_array = np.array(img)
        if img_array.shape[-1] == 4:  # If the image has an alpha channel, remove it
            img_array = img_array[:, :, :3]
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None


# Streamlit app title and description
st.title('🌸 Flower Detection Model 🌸')
st.write("""
    Upload an image of a flower, and the model will classify it into one of the following categories:
    **Dahlia, Garbera, Hibiscus, Jasmine, Lotus, Marigold, Rose, Sunflower, Tulip, Rajnigandha**
""")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write(f"Uploaded file name: {uploaded_file.name}")  # Debugging line
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Predict'):
        st.write("Classifying...")
        try:
            img_array = load_and_preprocess_image(image)
            predictions = model.predict(img_array)
            pred_class = np.argmax(predictions, axis=1)
            st.write(f'**Predicted Class:** {class_names[pred_class[0]]}')
        except Exception as e:
            st.error(f"Error in processing the image: {e}")

# Footer
st.write("---")
st.write("Developed by A")

#streamlit run flowerapp.py --server.enableXsrfProtection false
