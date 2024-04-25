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

# Streamlit app layout
st.title(f'Image Classifier - {", ".join(classes)}')

# Radio buttons for image selection (upload or webcam)
source = st.radio("Select image source .jpeg :", ("Upload Image", "Webcam"))

if source == "Upload Image":
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

    # Display the uploaded image and prediction results
    st.image(uploaded_file, width=250)
    st.write(f"I'm {prob}% sure this is a {predicted_class}.")
else:
  # Access webcam for real-time classification
  run = st.checkbox("Run Webcam")
  video_capture = cv2.VideoCapture(0)

  while run:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if not ret:
      # Handle potential video capture failure
      st.error("Error: Failed to capture frame from webcam.")
      break

    # Preprocess the frame
    frame = load_image(cv2.imencode('.jpg', frame)[1])
    # Make predictions
    probabilities = model.predict(frame)

    # Get the predicted class with highest probability
    predicted_class = classes[np.argmax(probabilities[0])]
    prob = round(np.max(probabilities[0]) * 100, 2)

    # Display the webcam frame and prediction results
    cv2.putText(frame, f"{predicted_class} - {prob}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    st.image(frame, channels="BGR")

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Release the capture and close all windows
  video_capture.release()
  cv2.destroyAllWindows()

st.balloons()
