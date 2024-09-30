import tensorflow as tf 
import cv2
from PIL import Image
import numpy as np
import streamlit as st


def make_prediction(img,model):
    # img=cv2.imread(img)
    img=Image.fromarray(img)
    img=img.resize((128,128))
    img=np.array(img)
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)
    if res:
        print("Tumor Detected")
    else:
        print("No Tumor")
    return res

model = tf.keras.models.load_model('cnn_tumor.h5')
st.title("Brain Tumor Detector")
img=st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
if img:
    img=Image.open(img)
    img=np.array(img)
    st.image(img,caption="Uploaded Image",use_column_width=True)
    if st.button("Predict"):
        res=make_prediction(img,model)
        if res:
            st.success("Tumor Detected")
        else:
            st.success("No Tumor")
    else:
        st.write("Click on Predict Button to make Prediction")
