import cv2
import tensorflow as tf
import streamlit as st
import numpy as np

label= ["C0:safe driving","C1:texting - right","C2:talking on the phone - right",
        "C3:texting - left",
        "C4:talking on the phone - left",
        "C5:operating the radio",
        "C6:drinking",
        "C7:reaching behind",
        "C8:hair and makeup",
        "C9:talking to passenger"]


def prepare(img):
    img = cv2.resize(img, (224, 224))
    img.reshape(-1, 224, 224, 3)
    img = np.array(img)
    img = np.array(img).reshape(-1, 224, 224, 3)
    return img

@st.cache
def load_model():

    model = tf.keras.models.load_model('vgg_model.h5')
    return model
model = load_model()

alert= st.empty()
st.title('Drowsiness Detection')
cam = cv2.VideoCapture(0)
start = st.checkbox('Start')
textside = st.sidebar.empty()

img = st.empty()

while start:
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prediction = model.predict([prepare(frame)])

    p = label[np.argmax(prediction)]
    textside.subheader("Label :" + str(p))
    if(p!= "C0:safe driving"):
        alert.warning('ALERT DRIVE PROPERLY')
    img.image(frame, "Cam")
img = st.empty()
cam.release()





