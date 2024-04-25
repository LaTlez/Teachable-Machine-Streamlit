import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Function to load image and preprocess it for the model
def load_image(image):
    """Preprocesses an image for the model (resizing, normalization).

    Args:
        image: A file-like object containing the image data.

    Returns:
        A NumPy array representing the preprocessed image, ready for model input.
    """
    # Convert image to CV2 format (BGR)
    cv2_img = cv2.imdecode(np.frombuffer(image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    # Resize the image
    image = cv2.resize(cv2_img, (224, 224), interpolation=cv2.INTER_AREA)
    # Convert to float32
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # Normalize the image
    image = (image / 127.5) - 1
    return image

# Load the class labels from labels.txt and assign to a list
classes = [' '.join(x.split(' ')[1:]).replace('\n','') for x in open('labels.txt', 'r').readlines()]

# Load the Keras model
model = load_model('keras_model.h5', compile=False)

# Load pre-calculated metrics (assuming you have them)
try:
  # Load accuracy per class (replace 'accuracy_per_class.npy' with your file)
  accuracy_per_class = np.load('accuracy_per_class.npy')
except FileNotFoundError:
  st.warning("Accuracy per class data not found.")
  accuracy_per_class = None

try:
  # Load confusion matrix (replace 'confusion_matrix.npy' with your file)
  confusion_matrix = np.load('confusion_matrix.npy')
except FileNotFoundError:
  st.warning("Confusion matrix data not found.")
  confusion_matrix = None

try:
  # Load training history (replace 'training_history.npy' with your file)
  history = np.load('training_history.npy')
  epochs = range(len(history))
except FileNotFoundError:
  st.warning("Training history data not found.")
  history = None


# Streamlit app layout
st.title(f'Image Classifier - {", ".join(classes)}')

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
  # Load and preprocess the uploaded image
  image = load_image(uploaded_file)
  # Make predictions
  probabilities = model.predict(image)

  # Get the predicted class with highest probability
  predicted_class = classes[np.argmax(probabilities[0])]
  prob = round(np.max(probabilities[0]) * 100, 2)

  # Display the uploaded image
  col1, col2 = st.columns(2)
  with col1:
    st.image(uploaded_file, width=250)
  with col2:
    st.subheader("Classification Results")
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {prob}%")

    # Create a progress bar to visually represent confidence
    st.progress(prob / 100)

# Display dashboard after result
st.subheader("Evaluation Metrics")

# Display accuracy per class (if available)
if accuracy_per_class is not None:
  st.subheader("Accuracy per Class")
  for i, class_name in enumerate(classes):
      class_accuracy = accuracy_per_class[i]
      st.write(f"{class_name}: {class_accuracy:.2%}")

# Display confusion matrix (if available)
if confusion_matrix is not None:
  st.subheader("Confusion Matrix")
  st.dataframe(confusion_matrix, index=classes, columns=classes)

# Display accuracy and loss per epoch (if available)
if history is not None:
  st.subheader("Accuracy and Loss per Epoch")
  fig, (ax1, ax2) = st.subplots(1, 2, figsize=(12, 4))
  ax1.plot(epochs, history[:, 0], label='Training Accuracy')
  ax1.plot(epochs, history[:, 1], label='Validation Accuracy')
  ax1.set_title('Accuracy')
  ax1.legend()
  ax2.plot(epochs, history[:, 2], label='Training Loss')
  ax2.plot(epochs, history[:, 3], label='Validation Loss')
  ax2.set_title('Loss')
  ax2.legend()
  st.pyplot(fig)

st.balloons()
