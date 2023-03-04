import pickle
import os
import numpy as np
from skimage.transform import resize
from PIL import Image
import streamlit as st
from train import train_model_, get_images
from skimage.io import imread
from skimage.transform import resize

# Web app UI
st.title('Image Classifier')
st.text('(Model is limited to classes given by the user)')


if 'page' not in st.session_state:
    st.session_state.page = 0


def nextPage(): st.session_state.page += 1
def firstPage(): st.session_state.page = 0


ph = st.empty()

if st.session_state.page == 0:
    with ph.container():
        # User Input
        st.text('Write The Class (eg, Train, Car, etc) ')

        select_box = st.selectbox('Select Total Classes', [2, 3, 4, 5])

        User_input = [st.text_input(f'Class {_}')
                      for _ in range(1, select_box+1)]  # arg1

        n = st.text_input('Enter Number Of Images (>10 recommended)')  # arg2

        if '' in User_input:
            st.warning('Input required')
        else:
            if st.button('Train Model'):
                if int(n) < 10:
                    st.warning('Number greater than 10 expected')
                else:
                    st.write('Downloading Images...')
                    get_images(User_input, int(n))
                    st.write('Training Model...')
                    train_model_()  # function train_model
                    nextPage()


if st.session_state.page == 1:
    with ph.container():
        # Prediction
        st.text('Upload the Image')
        uploaded_file = st.file_uploader('Choose an image')
        test_url = st.text_input('Enter the test image url')

        # results of uploads
        if st.button('PREDICT CLASS'):
            model = pickle.load(open('img_model.pkl', 'rb'))
            if test_url != "":
                img = imread(test_url)
            else:
                img = Image.open(uploaded_file)
            st.image(img, 'Uploaded Image')
            flatten_data = []
            DATA_DIR = 'Images/'
            CATEGORIES = os.listdir(DATA_DIR)
            st.write('Results...')

            img = np.array(img)
            img_resized = resize(img, (150, 150, 3))
            flatten_data.append(img_resized.flatten())
            flatten_data = np.array(flatten_data)
            y_out = model.predict(flatten_data)
            y_out = CATEGORIES[y_out[0]]
            st.write(f' Predicted Class is {y_out.upper()}')

        if st.button('Back'):
            firstPage()
