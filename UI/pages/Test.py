import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

st.set_page_config(
    page_title="Oral Lesion App", page_icon="😏"
)

st.title("Object Detection Page")

def load_model():
    model_url = "https://tfhub.dev/tensorflow/efficientdet/d7/1"
    model = hub.load(model_url)
    return model

@st.cache(allow_output_mutation=True)
def get_detector():
    return load_model()

def object_detection(image, model):
    # Preprocess the image for object detection
    image = tf.image.decode_image(image.read())
    image = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]

    # Run object detection on the image
    detections = model(image)

    # Display the detected image
    image_with_detections = tf.image.draw_bounding_boxes(
        image, detections["detection_boxes"], detections["detection_classes"]
    )

    return image_with_detections

def main():
    st.info(__doc__)

    # Load the object detection model
    detector = get_detector()

    file = st.file_uploader("Upload file", type=["png", "jpg"])
    show_file = st.empty()

    if not file:
        st.warning("Please upload an image file.")
        return

    content = file.getvalue()

    if isinstance(file, BytesIO):
        # Display the original image
        st.image(file, caption="Uploaded Image", use_column_width=True)

        # Perform object detection
        detected_image = object_detection(file, detector)

        # Display the image with detections
        st.image(detected_image, caption="Image with Detections", use_column_width=True)
    else:
        st.error("Unsupported file type. Please upload a valid image file.")

if __name__ == "__main__":
    main()
