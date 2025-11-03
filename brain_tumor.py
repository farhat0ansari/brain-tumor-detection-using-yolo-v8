import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import tempfile

# Set title and description
st.title('Brain Tumor Detection System')
st.write('Upload an MRI scan to detect brain tumors using YOLOv10')

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpeg", "jpg", "png"])

# Create a placeholder for the model
model = None

def load_model():
    # Load the YOLOv10 model
    from ultralytics import YOLOv10
    model = YOLOv10("best.pt")  # Path to your model weights
    return model

def predict_brain_tumor(image_path, model):
    # Make prediction
    results = model.predict(source=image_path, conf=0.25)
    
    # Get the annotated image
    result = results[0]
    annotated_img = result.plot()
    
    # Convert from BGR to RGB (matplotlib uses RGB)
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    # Get detection details
    detections = result.boxes.data
    class_names = []
    if len(detections) > 0:
        for det in detections:
            cls_id = int(det[5])
            class_names.append(model.names[cls_id])
    
    # Count occurrences of each class
    detection_results = {}
    for name in class_names:
        if name in detection_results:
            detection_results[name] += 1
        else:
            detection_results[name] = 1
    
    # Create detection string
    detection_str = ', '.join([f"{name}: {count}" for name, count in detection_results.items()])
    if not detection_str:
        detection_str = "No tumors detected"
    
    return annotated_img, detection_str

# Main process
if uploaded_file is not None:
    # Load model if not already loaded
    if model is None:
        with st.spinner('Loading model...'):
            model = load_model()
        st.success('Model loaded successfully!')

    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Make prediction
    with st.spinner('Analyzing image...'):
        try:
            result_img, detection_str = predict_brain_tumor(tmp_path, model)
            
            # Display original image
            st.subheader("Uploaded Image")
            st.image(uploaded_file, caption='Original Image', use_column_width=True)
            
            # Display result
            st.subheader("Detection Result")
            st.image(result_img, caption='Detection Result', use_column_width=True)
            
            # Display detection text with color
            has_tumor = "tumor" in detection_str.lower()
            color = "red" if has_tumor else "green"
            st.markdown(f"<h3 style='color: {color};'>{detection_str}</h3>", unsafe_allow_html=True)
            
            if has_tumor:
                st.warning("Recommendation: Please consult with a medical professional for further evaluation.")
            else:
                st.success("No brain tumors detected in the scan.")
                
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
            
    # Clean up the temporary file
    os.unlink(tmp_path)

else:
    st.info("Please upload an MRI scan to begin.")
    
# Add additional information
st.sidebar.title("About")
st.sidebar.info(
    """
    This application uses YOLOv10 to detect brain tumors in MRI scans.
    
    Note: This tool is for demonstration purposes only and should not be 
    used for actual medical diagnosis. Always consult with a healthcare 
    professional.
    """
)
