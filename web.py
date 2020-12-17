import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = s_model = tf.keras.models.load_model("model.sav")

def predict_object(image_file):
    image = Image.open(image_file) 
    image = image.resize((32,32),Image.ANTIALIAS)
    img_array = np.asarray(image, dtype='int32')
    img_array = img_array.reshape(1, 32, 32, 3)
    prediction = model.predict(img_array)
    obj = np.argmax(prediction, axis=None, out=None)
    return obj


st.title("Welcome to the object detector program")
st.header("Please enter the image file for recognition such as aeroplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
result = ""
r = ""
if st.button("Predict"):
    result = predict_object(uploaded_file)
    if result == 0:
        r = 'aeroplane'
    elif result == 1:
        r = 'automobile'
    elif result == 2:
        r = 'bird'
    elif result == 3:
        r = 'cat'
    elif result == 4:
        r = 'deer'
    elif result == 5:
        r = 'dog'
    elif result == 6:
        r = 'frog'
    elif result == 7:
        r = 'horse'
    elif result == 8:
        r = 'ship'
    elif result == 9:
        r = 'truck'
st.success('The object detected is: {}'.format(r))

