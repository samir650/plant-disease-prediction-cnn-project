# Plant Disease Classifier

## Description
This project is a **Plant Disease Classifier** that allows users to upload an image of a plant leaf and get predictions about possible plant diseases. The model uses a deep learning approach to classify diseases and is deployed as a simple web application using Streamlit.

## Features
- Upload plant leaf images for disease detection.
- Uses a **pre-trained deep learning model** for classification.
- Provides real-time results via a user-friendly **Streamlit** interface.

## Tech Stack
- **Python** - Main programming language.
- **TensorFlow/Keras** - Deep learning framework for model training and inference.
- **Streamlit** - Web framework for the user interface.
- **Pillow (PIL)** - Image processing library.
- **NumPy** - Numerical operations and array manipulations.
- **JSON** - Storing and retrieving class names.

## Installation
Ensure you have **Python 3.7+** installed, then install the required dependencies:
```sh
pip install tensorflow streamlit pillow numpy
```

## How to Run
To start the web application, execute:
```sh
streamlit run app.py
```

## Usage
1. Open the application in the browser.
2. Upload an image of a plant leaf.
3. Click the **Classify** button to get a prediction.
4. The application displays the predicted disease.

## Code Structure
- **Model Loading:**
  ```python
  model = tf.keras.models.load_model("trained_model/plant_disease_prediction_model.h5")
  ```
- **Image Preprocessing:**
  ```python
  def load_and_preprocess_image(image_path, target_size=(224, 224)):
      img = Image.open(image_path)
      img = img.resize(target_size)
      img_array = np.array(img) / 255.0
      img_array = np.expand_dims(img_array, axis=0)
      return img_array
  ```
- **Prediction Function:**
  ```python
  def predict_image_class(model, image_path, class_indices):
      img_array = load_and_preprocess_image(image_path)
      predictions = model.predict(img_array)
      predicted_class = np.argmax(predictions, axis=1)[0]
      return class_indices[str(predicted_class)]
  ```

## Future Improvements
- Improve model accuracy with additional training data.
- Deploy the application online for broader access.
- Add a feedback system to improve predictions over time.

## License
This project is licensed under the MIT License.

