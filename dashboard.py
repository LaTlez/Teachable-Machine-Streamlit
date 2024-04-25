import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score

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

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    """Calculates accuracy, confusion matrix, and optionally plots them.

    Args:
        y_true: True labels for the data.
        y_pred: Predicted labels by the model.

    Returns:
        A dictionary containing accuracy and confusion matrix values.
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Optionally plot confusion matrix (using Plotly.js)
    """
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        texttemplate="%s<br>%d%%"
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='True Class'
    )
    st.plotly_chart(fig)
    """

    return {'accuracy': accuracy, 'confusion_matrix': cm}

# Track predictions and true labels for evaluation (if sufficient data available)
predictions = []
true_labels = []

# Streamlit app layout
st.title(f'Image Classifier - {", ".join(classes)}')

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg|jpeg|png")

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

  # Track predictions and true labels (assuming ground truth available)
  if st.checkbox("Have ground truth label?"):
      true_label = st.text_input("Enter the correct class label:", key="true_label")
      if true_label:
          true_labels.append(classes.index(true_label))
          predictions.append(classes.index(predicted_class))

# Display dashboard after result (if sufficient data available)
if len(true_labels) > 0:
  # Evaluate model performance
  evaluation_results = evaluate_model(true_labels, predictions)

  # Display accuracy per class
  st.subheader("Accuracy per Class")
  class_accuracies = {}
  for i, class_name in enumerate(classes):
      true_count = sum(label == i for label in true_labels)
      predicted_count = sum(pred == i for pred in predictions)
      if true_count > 0:  # Avoid division by zero
          class_accuracy = round((predicted_count / true_count) * 100, 2)
      else:
          class_accuracy = 0.0
      class_accuracies[class_name] = class_accuracy
      st.write(f"{class_name}: {class_accuracy}%")

  # Display overall accuracy
  overall_accuracy = evaluation_results['accuracy']
  st.subheader(f"Overall Accuracy: {overall_accuracy:.2%}")

  # Optionally display confusion matrix (commented out)
  """
  # Display confusion matrix
  st.subheader("Confusion Matrix")
  cm = evaluation_results['confusion_matrix']
  st.dataframe(cm, index=classes, columns=classes)
  """

  # Load and display accuracy and loss history (if available)
  try:
      # Assuming history data is stored in a NumPy array named 'history'
      history = np.load('training_history.npy')
      epochs = range(len(history))
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
  except FileNotFoundError:
      st.write("Training history data not found.")

st.balloons()
