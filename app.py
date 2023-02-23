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

# User Input
st.text('Write The Class (eg, Train, Car, etc) ')

select_box = st.selectbox('Select Total Classes', [2, 3, 4, 5])

User_input = [st.text_input(f'Class {_}')
              for _ in range(1, select_box+1)]  # arg1

n = st.text_input('Enter Number Of Images (>10 recommended)')  # arg2

if '' in User_input:
    st.warning('Input required')
else:
    try:
        n = int(n)
    except:
        st.warning('Enter a valid number')

    if st.button('Train Model'):
        if n < 10:
            st.warning('Number greater than 10 expected')
            try:
                folder = (os.listdir('Images')[0])
                if os.listdir('Images') != User_input or (len(os.listdir(f'Images/{folder}'))):
                    get_images(User_input, int(n))  # func get_images
            except:
                get_images(User_input, int(n))
            st.write('Training Model...')
            train_model_()  # function tain_model

            # Prediction

            st.text('Upload the Image')

            model = pickle.load(open('img_model.p', 'rb'))

            uploaded_file = st.file_uploader('Choose an image', type='jpg')

            test_url = st.text_input('Enter the test image url')

            # results of uploads

            if uploaded_file is not None or test_url != '':
                if test_url != "":
                    img = imread(test_url)
                else:
                    img = Image.open(uploaded_file)
                st.image(img, 'Uploaded Image')

                if st.button('PREDICT CLASS'):
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
