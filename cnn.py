import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score,confusion_matrix
from tensorflow.keras.utils import to_categorical
from PIL import Image
import pickle
import numpy as np


model=load_model('cnn.h5')
with open('names.pkl','rb') as f:
    names=pickle.load(f)

st.title('📷 CIFAR-10 Image Classifier')
st.write('uploade your cifar10 Image')
file=st.file_uploader('Uploade Image...',type=["jpg","jpeg","png"])
if file is not None:
    img=Image.open(file).convert('RGB')
    st.image(img,caption="Uploaded Image", use_column_width=True)

    img_size=img.resize((32,32), Image.Resampling.LANCZOS)
    img_array=np.asarray(img_size)/255.0
    img_dim=np.expand_dims(img_array,axis=0)

    mol=model.predict(img_dim)[0]
    mol_max=np.argmax(mol)
    class_names=names[mol_max]
    last_mol=mol[mol_max]


    st.success(st.success(f"### 🎯 Prediction: **{class_names}** (Confidence: {last_mol:.2f})")
)


