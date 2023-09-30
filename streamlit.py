# -*- coding: utf-8 -*-

import copy
import streamlit as st
from streamlit_option_menu import option_menu
import PIL
from PIL import Image
from colorthief import ColorThief
import webcolors
import os
import io
from ultralytics import YOLO
from tensorflow.keras.applications.resnet_rs import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import textwrap
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as tf_hub
from streamlit_extras.let_it_rain import rain
from tensorflow.keras.applications.vgg16 import preprocess_input as pinp
from io import BytesIO
import streamlit.components.v1 as components
from st_custom_components import st_audiorec
from st_custom_components1 import st_audiorec1
import speech_recognition as SR
import deepl 
import openai
from apikey import apikey
import replicate
from playsound import playsound
import time
import urllib.request
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import sqlite3
from selenium.webdriver.common.keys import Keys
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Set the Streamlit page configuration
# Set the title, icon, and layout of the Streamlit app page
st.set_page_config(page_title="KTA by ColdShower team", page_icon="random", layout="wide")

# Define a caching decorator to avoid loading the image repeatedly and to speed up subsequent visits
@st.cache_data
def load_image():
    # Load an image from the given path
    image = Image.open("C:/streamlit_files/title.jpg")
    return image

# Display the image on the Streamlit page with full column width
st.image(load_image(), caption="", use_column_width=True)

# Create a sidebar in Streamlit 
with st.sidebar:
    # Option menu in the sidebar with specific icons for each menu item
    selected = option_menu("Main Menu", ['Home', 'Know Thy Art','Neural Style Transfer','Artwork MBTI','Speech to Art to Speech', 'Art Chatbot'], 
        icons=['shop', 'palette','camera fill','puzzle','easel','lightbulb'], menu_icon="cast", default_index=0)

# Define a caching decorator to avoid loading and resizing images repeatedly
@st.cache_data
def load_and_resize_images():
    images = []
    # Load and resize five images from the specified path
    for i in range(1, 6):
        img=Image.open(f"C:/streamlit_files/home_{i}.jpg")
        images.append(img)
    return images

# When the selected option from the sidebar is 'Home'
if selected == 'Home':
    # Display a header text on the Streamlit page
    st.header("Welcome to our Homepage!")
    
    # Load and display the resized images on the Streamlit page
    images = load_and_resize_images()
    for img in images:
        st.image(img, use_column_width=True)

# When the selected option from the sidebar is 'Know Thy Art'
elif selected == 'Know Thy Art':
    
    # Define a caching decorator to avoid loading the YOLO model repeatedly and to speed up subsequent visits
    @st.cache_resource
    def yolo():
        # Load the YOLO model from the given path
        model = YOLO(r"C:\streamlit_files\best_m.pt")
        return model
    
    # Load the YOLO model using the defined function
    model = yolo()
    
    # A form element in the Streamlit with a unique key "form"
    with st.form(key="form"):
        
        # Add a file uploader widget to the form, allowing users to upload 'png', 'jpg', or 'jpeg' files
        # The uploaded file is stored in the variable 'source_img'
        source_img = st.file_uploader(label='Choose an image...', type=['png','jpg', 'jpeg'])
        
        # Add a submit button to the form with the label "Analyze"
        # When clicked, any processing related to this form would be executed
        submit_button = st.form_submit_button(label="Analyze")
        
        # When the submit button was clicked
        if submit_button:
            
            # If an image has been uploaded
            if source_img:
                # Open the uploaded image using the Image module
                uploaded_image = Image.open(source_img)
                
                # If the uploaded_image is successfully loaded
                if uploaded_image:
                    # Use the previously loaded YOLO model to make predictions on the uploaded image
                    result = model.predict(uploaded_image)
                    # Extract and prepare the plot of the result, reversing the color channels from BGR to RGB
                    result_plot = result[0].plot()[:, :, ::-1]               
                    
                    # Display a spinner with the message "Running...." while the following code block is being executed
                    with st.spinner("Running...."):
                        try:
                            result_2 = result[0] # Extract the prediction result
                            box = result_2.boxes[0] # Get the bounding box of the detected object (for one object only)         
                            cords = box.xyxy[0].tolist() # Extract the coordinates of the bounding box
                            cords = [round(x) for x in cords] # Round the coordinates to get integer values
                            area = tuple(cords) # Convert the coordinates to a tuple which represents the area for cropping
                            
                            # Define a caching decorator to avoid reloading and cropping the image repeatedly
                            @st.cache_data
                            def load_and_crop_image(source_img, area):
                                lc_img = uploaded_image.crop(area) # Crop the image using the specified area
                                lc_img=copy.deepcopy(lc_img) # Deep copy the cropped image to ensure original data isn't modified
                                return lc_img
                            
                            # Load and crop the image using the defined function
                            cropped_img = load_and_crop_image(source_img, area)
                            
                            # Split the Streamlit layout into three columns
                            col1, col2,col3 = st.columns(3)
                            
                            # In the first column
                            with col1:
                                # Display the original uploaded image
                                st.image(image=uploaded_image,
                                         caption='Uploaded Image',
                                         use_column_width=True) # Adjust the image to fit the column width    
                            
                            # In the second column
                            with col2:
                                # Display the result image with detected objects
                                st.image(result_plot, 
                                         caption="Detection Image", 
                                         use_column_width=True)
                            
                            # In the third column
                            with col3:
                                # Display the cropped image containing only the detected object
                                st.image(cropped_img, 
                                         caption="Cropped Image", 
                                         use_column_width=True)
                                
                        # In case there's an exception/error in the above code block
                        except:
                            # Display only the original uploaded image with an alternative caption
                            st.image(image=uploaded_image,
                                     caption='No paintings detected in the uploaded image', 
                                     use_column_width=True)  
                            
                            # Define a caching decorator to ensure that the uncropped image is processed only once and reused on subsequent calls without having to deep copy the image again
                            @st.cache_data
                            def uncropped_img():
                                # Deep copy the uploaded image to ensure original data isn't modified
                                uc_img=copy.deepcopy(uploaded_image)
                                return uc_img
                            
                            # Fetch the uncropped image using the defined function
                            cropped_img=uncropped_img()
                        
                        # Define a caching decorator to ensure the model is loaded only once and reused on subsequent calls
                        @st.cache_resource
                        def rnrs50():
                            # Load a pre-trained model from the specified path
                            model=load_model(r"C:\streamlit_files\model_resnetrs50_lion_dense10240_noda.h5")
                            return model
                        
                        m = rnrs50() # Load the model using the defined function
                        x = img_to_array(cropped_img) # Convert the cropped image into an array format
                        x = tf.image.resize(x, [224, 224]) # Resize the image to match the model's input size
                        x = np.array([x]) # Expand the dimensions of the image to match the model's input shape
                        x = preprocess_input(x) # Preprocess the image using the appropriate preprocessing function for the model (from tensorflow.keras.applications.resnet_rs import preprocess_input)                
                        predict = m.predict(x) # Make a prediction using the loaded model                      
                        
                        # Define a dictionary to map class indices to their respective art style names in English
                        class_indices = {0: 'Abstract Expressionism', 1: 'Baroque', 2: 'Cubism', 3: 'Impressionism',4 : 'Primitivism',5:'Rococo',6:'Surrealism'}  # Replace with the correct class indices and labels.
                        
                        # Define a dictionary to map class indices to their respective art style names in Korean
                        korean_class_indices={0:'추상표현주의 (Abstract Expressionism)',
                                              1:'바로크 (Baroque)',
                                              2:'입체주의 (Cubism)',
                                              3:'인상주의 (Impressionism)',
                                              4:'원시주의 (Primitivism)',
                                              5:'로코코 (Rococo)',
                                              6:'초현실주의 (Surrealism)'}                    
                        
                        # Get the indices of the top 3 predicted classes in descending order of confidence
                        top_3_indices = np.argsort(predict[0])[-3:][::-1]
                        
                        # Map the top 3 indices to their respective class labels
                        top_3_labels = [class_indices[index] for index in top_3_indices]
                        
                        # Fetch the predicted probabilities for the top 3 predictions
                        top_3_probabilities = [predict[0][index] * 100 for index in top_3_indices]
                        
                        # Get the index of the top prediction
                        top_prediction_index = np.argsort(predict[0])[-1]
                        
                        # Add a divider in the Streamlit app for better visual separation
                        st.divider()
                        
                        # Split the Streamlit layout into two columns
                        col1,col2 = st.columns(2)
                        
                        # In the first column
                        with col1:
                            
                            # Use HTML to display a stylized header
                            st.markdown("<h2 style='text-align: center; color: black;'>Top 3 Predicted Classes</h2>", unsafe_allow_html=True)
                            
                            fig, ax = plt.subplots() # Create a matplotlib figure and axis for a pie chart
                            
                            # Plot a pie chart with the top 3 predicted probabilities
                            wedges, texts= ax.pie(
                                top_3_probabilities, 
                                labels=['', '', ''], # Empty labels and use a legend instead
                                startangle=90, 
                                pctdistance=0.8, 
                                labeldistance=0.7,
                                colors=['#161953','#B3CEE5','#FAC898']
                                )
                            
                            # Create a white circle to convert the pie chart into a doughnut chart
                            circle = plt.Circle((0, 0), 0.6, color='white')
                            
                            # Add the white circle to the axis
                            ax.add_artist(circle)
                            
                            # Prepare a list of labels for the pie chart, displaying the class names and their probabilities
                            top_3_info=[]
                            
                            for index in top_3_indices:
                                class_label = class_indices[index]
                                probability = predict[0][index] * 100
                                top_3_info.append(f'{class_label} ({probability:.2f}%)')
                                
                            # Add a legend to the pie chart
                            ax.legend(wedges, top_3_info, loc='lower center', fontsize=12, bbox_to_anchor=(0, -0.2, 1, 1))
                            
                            # Display the pie chart in the Streamlit
                            st.pyplot(fig)  
                        
                        # In the second column
                        with col2:
                            # Add vertical spacing using empty titles
                            st.title('')
                            st.title('')
                            
                            # Use HTML to display a stylized header
                            st.markdown("<h3 style='text-align: center; color: black;'>❣️ 해당 그림의 사조는<br></h3>", unsafe_allow_html=True)
                            
                            # Display the top predicted art style in Korean using the korean_class_indices dictionary
                            st.markdown(f"<h2 style='text-align: center; color: #161953;'>{korean_class_indices[top_prediction_index]}<br></h2>", unsafe_allow_html=True)
                            
                            # Continue the stylized message
                            st.markdown("<h3 style='text-align: center; color: black;'>와 가장 비슷합니다.<br></h3>", unsafe_allow_html=True)
                            
                            # Add more vertical spacing using an empty title
                            st.title('')
                            
                            # Retrieve the style name in Korean corresponding to the predicted class index
                            sajo = korean_class_indices[top_prediction_index]
                            
                            try:
                                # Set up credentials for accessing the Naver Text-to-Speech API
                                client_id = "" # Your Naver CLOVA API key here
                                client_secret = "" # Your Naver CLOVA API key here
                                
                                # Convert the text message into URL encoded format
                                encText = urllib.parse.quote("해당 그림의 사조는" + sajo + "와 가장 비슷합니다.")
                                
                                # Create the data payload for the API request, including the text message and TTS settings
                                data = "speaker=nara&volume=0&speed=0&pitch=0&format=mp3&text=" + encText
                                
                                # API endpoint for the Naver Text-to-Speech service
                                url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
                                
                                # Create a request to the Naver API
                                request = urllib.request.Request(url)
                                request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
                                request.add_header("X-NCP-APIGW-API-KEY",client_secret)
                                
                                # Send the request and get the response
                                response = urllib.request.urlopen(request, data=data.encode('utf-8'))
                                rescode = response.getcode()
                                
                                # If the API request was successful (status code 200)
                                if(rescode==200):
                                    print("TTS mp3 저장")
                                    
                                    # Read the response body, which contains the generated TTS audio in mp3 format
                                    response_body = response.read()
                                    
                                    # Save the mp3 file locally
                                    with open('1111.mp3', 'wb') as f:
                                        f.write(response_body)
                                        
                                    # Play the saved mp3 audio file
                                    playsound('C:/project2/1111.mp3')
                                    
                                else:
                                    print("Error Code:" + rescode)
                                    
                            finally:
                                
                                # Remove the mp3 file after playing it
                                os.remove('C:/project2/1111.mp3')
                            
                            # Define a caching decorator to ensure that the data is loaded only once and reused on subsequent calls
                            @st.cache_data
                            def styles_v4():
                                # Load a CSV file into a Pandas DataFrame
                                styles_df = pd.read_csv("C:/streamlit_files/styles_v9.csv")
                                return styles_df
                            
                            # Load the data using the defined function
                            df = styles_v4()
                            
                            # Filter rows from the DataFrame where the style matches the top predicted style
                            matching_rows = df[df['style'] == class_indices[top_prediction_index]]
                            
                            # Extract the descriptions related to the matched style
                            matching_description = matching_rows['app'].values
                            
                            # Extract the experiences (or some related data) associated with the matched style.
                            # Assuming 'exp' column contains list-like structures, fetch the first list
                            matching_exps = list(matching_rows['exp'].values)[0]
                            
                            # Create a 1:6 ratio split columns layout
                            col1, col2 = st.columns([1, 6])
                            
                            # Check if there are any matching descriptions
                            if len(matching_description) > 0:
                                # Display each matching descriptions in the larger column (col2)
                                for app in matching_description:
                                    col2.markdown(app, unsafe_allow_html=True)
                                
                                try:
                                    
                                    # Convert the text message into URL encoded format
                                    encText = urllib.parse.quote(matching_exps)
                                    
                                    # Create the data payload for the API request, including the text message and TTS settings
                                    data = "speaker=nara&volume=0&speed=0&pitch=0&format=mp3&text=" + encText
                                    
                                    # API endpoint for the Naver Text-to-Speech service
                                    url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
                                    
                                    # Create a request to the Naver API
                                    request = urllib.request.Request(url)
                                    request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
                                    request.add_header("X-NCP-APIGW-API-KEY",client_secret)
                                    
                                    # Send the request and get the response
                                    response = urllib.request.urlopen(request, data=data.encode('utf-8'))
                                    rescode = response.getcode()
                                    
                                    # If the API request was successful (status code 200)
                                    if(rescode==200):
                                        print("TTS mp3 저장")
                                        
                                        # Read the response body, which contains the generated TTS audio in mp3 format
                                        response_body = response.read()
                                        
                                        # Save the mp3 file locally
                                        with open('1112.mp3', 'wb') as f:
                                            f.write(response_body)
                                            
                                        # Play the saved mp3 audio file
                                        playsound('C:/project2/1112.mp3')
                                        
                                    else:
                                        print("Error Code:" + rescode)
                                        
                                finally:
                                    
                                    # Remove the mp3 file after playing it
                                    os.remove('C:/project2/1112.mp3')
                                 
                            else:
                                # Display a subheader if no descriptions are found for the predicted art style
                                st.subheader("No related descriptions found for the predicted art style.")
                                
                        # Try to execute the following block of code
                        try:
                            
                            # Check if the bytes representation of cropped_img is not equal to a new 5x5 white image
                            # This step ensures that the cropped_img is not an empty/white image
                            if not cropped_img.tobytes() == Image.new('RGB', (5, 5), color='white').tobytes():
                                
                                st.divider() # Add a divider to the Streamlit for better visual separation
                                st.subheader('') # Add some vertical space using an empty subheader
                                
                                # Use HTML to display a stylized header for color analysis
                                st.markdown("<h2 style='text-align: center;'>Color Analysis</h2>", unsafe_allow_html=True)
                                
                                st.subheader('') # Add more vertical space using an empty subheader
                                
                                cropped = cropped_img.convert('RGB') # Convert the cropped image to RGB format                            
                                image_bytes = io.BytesIO() # Create a BytesIO object to store the cropped image in memory
                                cropped.save(image_bytes, format='JPEG') # Save the cropped image to the BytesIO object in JPEG format
                                image_bytes.seek(0) # Reset the position of the BytesIO object
                                
                                # Initialize ColorThief with the in-memory image to extract dominant colors and a color palette
                                color_thief = ColorThief(image_bytes)
                                dominant_color = color_thief.get_color() # Extract the dominant color
                                color_palette = color_thief.get_palette(color_count=6) # Extract a color palette of 6 colors
                                
                                # Lists to store the extracted dominant color and color palette
                                dominant_color_list = []
                                color_palette_list = []
                                
                                # Append the colors to the respective lists
                                dominant_color_list.append(dominant_color)
                                color_palette_list.append(color_palette)
                                
                                # Define a function to convert an RGB tuple to its hex representation
                                def rgb_to_hex(rgb_tuple):
                                    r, g, b = rgb_tuple
                                    return "#{:02x}{:02x}{:02x}".format(r, g, b)            
                                
                                # Convert the dominant color to its hex representation
                                code = rgb_to_hex(dominant_color)
                                
                                # Split the Streamlit layout into two columns
                                col1,col2 = st.columns(2)
                                
                                # In the first column
                                with col1:
                                    # Display the cropped image
                                    st.image(cropped_img,use_column_width=True)                        
                                st.subheader('') # Add vertical spacing using an empty subheader
                                
                                # In the second column
                                with col2:
                                    st.title('') # Add vertical spacing using an empty title
                                    
                                    col1,col2 = st.columns(2) # Split the column into two more columns for better alignment
                                    
                                    # In the first sub-column
                                    with col1:
                                        # Use HTML to display a label for the dominant color
                                        st.markdown("<p style='padding:1.5em;font-size:20px;'>Dominant Color : </p>", unsafe_allow_html=True)
                                    
                                    # In the second sub-column
                                    with col2:
                                        st.title('') # Add vertical spacing using an empty title
                                        
                                        # Display the dominant color using a color picker widget, which will show the color but won't allow changes
                                        st.color_picker(label='dominant_color',value=code,label_visibility='collapsed')
                                    
                                    st.title('') # Add vertical spacing using an empty title
                                    
                                    # Use HTML to display a label for the color palette
                                    st.markdown("<p style='padding:1.5em;font-size:20px;'>Color Palette : </p>", unsafe_allow_html=True)
                                    
                                    st.title('') # Add vertical spacing using an empty title
                                    
                                    # Create seven columns to display the color palette
                                    columns = st.columns(7)
                                    
                                    # Iterate through the color palette and display each color using a color picker widget in the corresponding column
                                    for i, color in enumerate(color_palette):
                                        hex_color = rgb_to_hex(color)
                                        columns[i+1].color_picker(label=f"Color {i+1}", value=hex_color,label_visibility='collapsed')
                                
                                st.divider() # Add a divider to the Streamlit for better visual separation
                                
                                st.subheader('') # Add some vertical space using an empty subheader
                                
                                # Use HTML to display a stylized header for artworks with similar colors
                                st.markdown("<h2 style='text-align: center; color: black;'>Artworks with similiar colors</h2>", unsafe_allow_html=True)
                               
                                st.subheader('') # Add more vertical space using an empty subheader
                                
                                # Define a function to calculate the Euclidean distance between two RGB color values
                                def rgb_distance(rgb1, rgb2): 
                                    r1, g1, b1 = rgb1
                                    r2, g2, b2 = rgb2
                                    return ((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2) ** 0.5
                                
                                # Define a function to find the closest matching color from a list of color names
                                def find_closest_color(rgb_color, color_list):
                                    closest_color = None
                                    min_distance = float('inf') # Initialize minimum distance to a high value
                                    closest_color_index = None
                                    
                                    # Iterate through the color list and find the closest matching color
                                    for i, color_name in enumerate(color_list):
                                        distance = rgb_distance(rgb_color, webcolors.name_to_rgb(color_name))
                                        if distance < min_distance:
                                            min_distance = distance
                                            closest_color = color_name
                                            closest_color_index = i
                                        
                                    return closest_color, closest_color_index
                                
                                # Extract the dominant RGB color from the list
                                rgb_color = dominant_color_list[0]
                                
                                # Define a list of color names for comparison
                                color_names = ['orangered', 'bisque', 'sandybrown', 'linen', 'antiquewhite', 'lavender', 'darkslateblue', 
                                                       'lightsteelblue', 'steelblue', 'midnightblue', 'cadetblue', 'wheat', 'goldenrod', 'palegoldenrod', 
                                                       'beige', 'khaki', 'rosybrown', 'indianred', 'maroon', 'darkolivegreen', 'darkkhaki', 'darkseagreen', 
                                                       'olivedrab', 'tan', 'sienna', 'peru', 'saddlebrown', 'burlywood', 'darkslategray', 'thistle', 'dimgray', 
                                                       'silver', 'gray', 'darkgray', 'lightgray', 'gainsboro', 'lightslategray', 'slategray', 'whitesmoke', 'palevioletred', 'black']
                                
                                # Find the closest color name to the dominant color using the defined function above
                                closest_color, closest_color_index = find_closest_color(rgb_color, color_names)
                                
                                # Define a caching decorator to ensure that the data is loaded only once and reused on subsequent calls
                                @st.cache_data
                                def final_v5():
                                    final_v5 = pd.read_csv(r"C:\streamlit_files\12_final_v5(0806).csv") # Load a CSV file into a Pandas DataFrame
                                    return final_v5
                                
                                simcol_df = final_v5() # Load the data using the defined function
                                selected_rows = simcol_df[simcol_df['rep_clr'] == closest_color] # Filter rows where the 'rep_clr' column matches the closest color
                                group = selected_rows.iloc[0]['group'] # Extract the 'group' value from the first row of the filtered DataFrame
                                selected_rows = simcol_df[simcol_df['web_cg_dt'] == group] # Filter rows based on the extracted 'group' value
                                random_sample = selected_rows.sample(n=9) # Randomly sample 9 rows from the filtered DataFrame
                                file_names = random_sample['file_name'].tolist() # Extract the 'file_name' values from the sampled rows into a list
                                
                                # Define paths to image folders of 12 art styles
                                folder_paths = [r"C:\streamlit_files\abstract_expressionism_img",
                                                        r"C:\streamlit_files\nap_img",
                                                        r"C:\streamlit_files\symbolism_img",
                                                        r"C:\streamlit_files\rc_img",
                                                        r"C:\streamlit_files\cu_img",
                                                        r"C:\streamlit_files\bq_img",
                                                        r"C:\streamlit_files\northern_renaissance_img",
                                                        r"C:\streamlit_files\impressionism_img",
                                                        r"C:\streamlit_files\romanticism_img",
                                                        r"C:\streamlit_files\sr_img",
                                                        r"C:\streamlit_files\expressionism_img",
                                                        r"C:\streamlit_files\realism_img"]
                                
                                # Define filename prefixes associated with 12 different art styles
                                files = ['abstract_expressionism_', 'nap_', 'symbolism_', 'rc_', 'cu_', 'bq_', 'orthern_renaissance',
                                                      'impressionism_', 'romanticism_', 'sr_', 'expressionism_', 'realism_']
                                
                                # Define a function to construct the full file path for a given art style and file number
                                def get_style_filename(prefix, number):
                                    idx = files.index(prefix) # Find the index of the provided prefix in the 'files' list
                                    folder_path = folder_paths[idx] # Extract the corresponding folder path
                                    filename = f'{prefix}{number}.jpg' # Construct the filename using the provided prefix and number
                                    file_path = os.path.join(folder_path, filename) # Construct the full file path by joining the folder path and filename
                                    return file_path
                                
                                numbers = file_names # Store the filenames in the 'numbers' variable
                                
                                # Create a new figure of specified size (10,10) for plotting
                                plt.figure(figsize=(10, 10))
                                
                                # Iterate over the filenames in 'numbers'
                                for i, num in enumerate(numbers):
                                    
                                    # Check for each prefix to determine the style and retrieve the correct image file
                                    for prefix in files:
                                        if num.startswith(prefix): # If the file name starts with the current prefix
                                            number = num[len(prefix):] # Extract the file number by removing the prefix
                                            file_path = get_style_filename(prefix, number) # Get the complete path using the function defined earlier
                                            image = imread(file_path) # Read the image from the file path
                                        
                                            plt.subplot(3, 3, i + 1) # Plot the image in a 3x3 grid
                                            plt.imshow(image) # Display the image
                                            plt.axis('off') # Hide axes ticks and labels
                                                    
                                            # Retrieve relevant information from the dataframe based on the filename
                                            row = simcol_df[simcol_df['file_name'] == num]
                                            if not row.empty: # Ensure that there's relevant data
                                                title = row['Title'].values[0] # Extract the title of the artwork
                                                painter = row['Painter'].values[0] # Extract the painter's name
                                                
                                                # Annotate the image with the title, wrapping the text for better readability
                                                plt.annotate(textwrap.fill(f"{title}", width=35), (0,0), (0, -10), xycoords='axes fraction', textcoords='offset points', va='top')
                                                
                                                # Calculate the y-offset for the painter's name based on the title's length
                                                n1 = (len(title)) // 35 
                                                if (len(title)) % 35 == 0:
                                                    n1 -= 1
                                                y1 = -23 - 13*n1
                                                
                                                # Annotate the image with the painter's name, wrapping the text for better readability
                                                plt.annotate(textwrap.fill(f"by {painter}", width=35), (0,0), (0, y1), xycoords='axes fraction', textcoords='offset points', va='top')
                                
                                plt.tight_layout(h_pad=5) # Adjust the layout for better display
                                st.pyplot(plt.gcf()) # Display the plotted figure in Streamlit
                                st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress any warnings related to the use of pyplot in Streamlit
                                
                                st.divider() # Add a divider to Streamlit for visual separation
                                
                                st.subheader('') # Add some vertical space using an empty subheader            
                                
                                # Use HTML to display a stylized header indicating the artworks with similar styles
                                st.markdown("<h2 style='text-align: center; color: black;'>Artworks with similiar styles</h2>", unsafe_allow_html=True)
                                
                                # Define a caching decorator to ensure that the VGG model is loaded only once and reused on subsequent calls
                                @st.cache_resource
                                def vgg_model():
                                    model=load_model("C:/streamlit_files/vgg16.h5") # Load a pre-trained VGG16 model from the specified path
                                    return model
                                
                                m = vgg_model() # Load the VGG16 model using the defined function
                                x = img_to_array(cropped_img) # Convert the cropped image to an array format
                                
                                x = tf.image.resize(x, [224, 224]) # Resize the image to the size expected by the VGG16 model (224x224)
                                x = np.array([x]) # Expand the dimensions of the array to match the input shape expected by VGG16, i.e., (batch_size, height, width, channels)
                                predict = m.predict(pinp(x)) # Preprocess the resized image with pinp and then extract feature vectors using the VGG16 model
                                # from tensorflow.keras.applications.vgg16 import preprocess_input as pinp
                                
                                # Define a caching decorator to ensure that the total dataset is loaded only once and reused on subsequent calls
                                @st.cache_data
                                def total_db():
                                    # Load a serialized dataframe (pickle format) from the specified path
                                    file = open("C:/streamlit_files/total.txt","rb")
                                    total_df = pickle.load(file)
                                    file.close()
                                    return total_df 
                                
                                total=total_db() # Load the dataset using the defined function
                                index_predict = total['predict'] # Extract predicted feature vectors from the dataset; total['predict']: Values processed through VGG 16 feature extraction                            
                                similarities = [] # List to store similarity scores between the cropped image and the artworks in the dataset                                        
                                
                                # Calculate cosine similarities between the cropped image and all artworks in the dataset
                                for i in index_predict:
                                    similarities.append(cosine_similarity(predict, i))                                            
                                x = np.array(similarities).reshape(-1,) # Convert the list of similarity scores to a numpy array
                                # reshape the array to a one-dimensional form (a single vector) aka flatten                                           
                                
                                # Get the top 9 rows with the highest similarity scores and reset their indices
                                top_9 = total.iloc[np.argsort(x)[::-1][:9]].reset_index(drop=True)         
            
                                # Append a prefix to the 'url' column, indicating the location where the image files are stored                                   
                                top_9['url'] = top_9['url'].apply(lambda x: 'C:/streamlit_files/paintings/' + x)                                    
                                plt.figure(figsize=(10, 10)) # Initialize a new figure for plotting
                                i = 1 # Initial value for subplot indexing    
                                
                                # Iterate over the 'url' column of the top_9 DataFrame using enumerate, tracking index and url
                                for idx, url in enumerate(top_9['url']):
                                    image = imread(url) # # Read an image from the provided URL using imread
                                    plt.subplot(3, 3, i) # Create a subplot grid of 3x3 and select the i-th subplot (i starts from 1)
                                    plt.imshow(image) # Display the loaded image on the current subplot
                                    plt.axis('off') # Hide axes ticks and labels
                                    i += 1 # Increment the subplot counter i
                                    
                                    # Extract title and painter for the current artwork
                                    title = top_9['title'][idx]
                                    painter = top_9['painter'][idx]

                                    # Annotate the image with the artwork's title
                                    plt.annotate(textwrap.fill(f"{title}", width=35), (0,0), (0, -10), xycoords='axes fraction', textcoords='offset points', va='top')
                                    # (0,0) : x and y coordinates of the point where the annotation arrow will point to
                                    # In this case, (0,0) indicates that the arrow will point to the bottom-left corner of the plot
                                    
                                    # (0,-10) :  offset from the specified point (0,0) where the annotation text will be placed
                                    # positioned slightly above the point specified in the previous step
                                    
                                    # xycoords='axes fraction': specifies the coordinate system for the xy point, which is (0,0) in this case. 
                                    # 'axes fraction' means that the coordinates are given as fractions of the axes' width and height. 
                                    # In this case, (0,0) corresponds to the bottom-left corner of the plot
                                    
                                    # textcoords='offset points': specifies the coordinate system for the text's offset from the xy point
                                    # 'offset points' means that the offset is given in points (a unit of measurement in typography), which is commonly used in plotting
                                    
                                    # va='top': stands for "vertical alignment" and specifies how the annotation text is aligned with respect to the xy point. 
                                    # 'top' means that the top of the annotation text will be aligned with the xy point.
                                    
                                    # Calculate the number of lines needed for the title annotation
                                    n1 = (len(title)) // 35 # quotient
                                    if (len(title)) % 35 == 0: # remainder
                                        n1 -= 1
                                    y1 = -23 - 13*n1
                                    
                                    # Annotate the image with the painter's name
                                    plt.annotate(textwrap.fill(f"by {painter}", width=35), (0,0), (0, y1), xycoords='axes fraction', textcoords='offset points', va='top')
                        
                                plt.tight_layout(h_pad = 5) # Adjust the layout for better spacing between the plots
                                st.pyplot(plt.gcf()) # Display the plotted figure in the Streamlit
                                st.set_option('deprecation.showPyplotGlobalUse', False) # Suppress any warnings related to the use of pyplot in Streamlit / Disable the warning for deprecated use of pyplot in the context of st.pyplot
                        
                        # Catch any exceptions that might occur during the process
                        except:
                            st.subheader('')            
                
                # If an image was processed but not uploaded
                else:
                    st.subheader('You didnt upload your image')
            
            # If the user didn't upload an image
            else:
                st.write("Please upload an image")

# When the selected option is 'Neural Style Transfer'
elif selected=='Neural Style Transfer':
    
    # Set the title for the page
    st.title('Neural Style Transfer')
    st.header('') # Add an empty header for spacing
    
    # Create two columns for uploading original and style images
    col1, col2 = st.columns(2)
    
    # In the first column, allow users to upload an original image
    with col1:
        original_image = st.file_uploader(label='Choose an original image', type=['jpg', 'jpeg'])
        
        if original_image : 
            # Display the uploaded original image
            st.image(image=original_image,
                     caption='Original Image',
                     use_column_width=True)
    
    # In the second column, allow users to upload a style image
    with col2: 
        style_image = st.file_uploader(label='Choose a style image', type=['jpg', 'jpeg'])
        
        if style_image :
            
            # Display the uploaded style image    
            st.image(image=style_image,
                         caption='Style Image',
                         use_column_width=True)    
    
    st.header('') # Add another empty header for spacing
    
    button=None # Initialize a button variable
    
    # Define a function to load and preprocess an image
    def load_image(image_file, image_size=(512, 256)):
        content = image_file.read() # read the content of the uploaded image file
        
        # Decode the image content using TensorFlow's decode_image function 
        # Convert the image to a tensor with an additional dimension using [tf.newaxis, ...]
        img = tf.io.decode_image(content, channels=3, dtype=tf.float32)[tf.newaxis, ...]
        
        # Resize the image while preserving its aspect ratio
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        return img
    
    # If both original and style images are uploaded
    if original_image and style_image :
        
        # Create columns for layout
        col1,col2,col3,col4,col5 = st.columns(5)
        with col3 : 
            
            # Create a stylize button
            button = st.button('Stylize Image')
            
            if button :
                
                # Display a spinner while processing
                with st.spinner('Running...') :
                    
                    # Load and preprocess the images using the load_image function defined above
                    original_image = load_image(original_image)
                    style_image = load_image(style_image)
                    
                    # Preprocess the style image
                    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='VALID')
                   
                   # Load the arbitrary image stylization model 
                    @st.cache_resource
                    def ais():
                        ais_model=tf_hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
                        return ais_model
                    stylize_model = ais()
            
                    # Stylize the image using the loaded model
                    results = stylize_model(tf.constant(original_image), tf.constant(style_image))
                    stylized_photo = results[0].numpy().squeeze()  # Convert to NumPy array and squeeze dimensions
                    stylized_photo_pil = PIL.Image.fromarray(np.uint8(stylized_photo * 255))  # Convert to PIL image and rescale to [0, 255]
    
    st.header('') # Add an empty header for spacing
    
    col1,col2,col3 = st.columns(3)
    
    # If the stylize button was pressed
    if button :
        with col2 :
            # Display the stylized image
            st.image(image=stylized_photo_pil,
                             caption='Stylized Image',
                             use_column_width=True)
            
            # Add a rain animation using a custom rain function
            rain(
                emoji="🎈",
                font_size=30,
                falling_speed=10,
                animation_length="infinite"
                )

# When the selected option is 'Artwork MBTI'
elif selected == 'Artwork MBTI':

    # Function to resize the image to a given width and height
    def resize_image(image, width, height):
        return image.resize((width, height), Image.Resampling.LANCZOS)

    # Function to run the sequential matchup game to determine user's MBTI based on art style preference
    def sequential_matchup_game(images, image_folder, mbti_data):
        st.subheader("더 마음에 드는 사진을 골라주세요 :smile:")
        # Instructions for the user
        st.write("본 미니게임은 11라운드로 진행되는 토너먼트식 게임입니다.")

        # Pair up image indices with their respective images
        image_list = list(zip(range(len(images)), images))

        # Initialize the match counter
        match_count = 0

        # Set the dimensions for displaying images
        width, height = 750, 572

        # Continue matchups until there's only one image left
        while len(image_list) > 1:
            match_count += 1
            st.write(f"{match_count}번째 라운드 :point_down: ")
            
            # Display two competing images side by side
            col1, col2 = st.columns(2)
            image_1 = image_list[0]
            image_2 = image_list[1]

            # Display the first image
            with col1:
                st.image(resize_image(image_1[1], width, height), use_column_width=True, caption='첫번째 이미지')

            # Display the second image
            with col2:
                st.image(resize_image(image_2[1], width, height), use_column_width=True, caption='두번째 이미지')

            # Let the user choose their preferred image
            choice = st.radio(f"어느 쪽이 더 좋나요? {match_count} 번째 선택", ('선택안함', '첫번째 이미지', '두번째 이미지'))

            # Process the user's choice
            if choice == '선택안함':
                # If the user doesn't make a choice, prompt them to select an option
                st.write("선택을 진행해주세요. 당신의 MBTI 유형을 맞혀보겠습니다. :bulb:")
                break
            
            elif choice == '첫번째 이미지':
                # If the user chooses the first image, move it to the end of the list and remove the first two images
                image_list.append(image_1)
                image_list.pop(0)
                image_list.pop(0)
            
            elif choice == '두번째 이미지':
                # If the user chooses the second image, move it to the end of the list and remove the first two images
                image_list.append(image_2)
                image_list.pop(0)
                image_list.pop(0)

            # Inform the user of the next step or the end of the game
            if match_count != 11:
                st.info('선택을 마쳤습니다. 스크롤을 내려 다음 라운드를 진행해주세요.', icon="ℹ️")
            else:
                st.info('모든 라운드가 끝났습니다. 스크롤을 내려 결과를 확인해주세요.', icon="ℹ️")

        # Once the matchups are done, display the winning image
        if len(image_list) == 1:
            winner_image = image_list[0]
            st.subheader("경기 종료!")
            st.write("최종 선택을 받은 작품은 :")
            st.image(resize_image(winner_image[1], width, height), use_column_width=True)

            # Fetch the MBTI data for the winning image
            mt = mbti_data.iloc[winner_image[0]]
            mbti_exp_info = mt['exp']
            mbti_short = mt['mbti']
            mbti_style = mt['style']
            st.subheader(mbti_style + " 작품이 제일 마음에 드는 당신의 MBTI 유형은....")
            st.subheader(mbti_short + ' 입니까:question:')
            st.write(mbti_exp_info)
    
    # The main function that initiates the entire MBTI art style test
    def main():
        st.title("Mini Game - 미술사조 mbti test :heart:") # Set the page title
        image_folder = "C:/streamlit_files/mbti/"  # Define the directory where the images are stored
        
        # List comprehension to generate the image filenames
        # It assumes there are 12 images named from img_1.jpg to img_12.jpg
        image_names = [f"img_{i}.jpg" for i in range(1, 13)]  
    
        images = [Image.open(image_folder + name) for name in image_names] # Load all images from the folder into a list
    
        mbti_data = pd.read_csv(r"C:\streamlit_files\style_mbti_v2.csv") # Load the MBTI data from a CSV file
        
        # Call the sequential_matchup_game function with the loaded images and MBTI data
        sequential_matchup_game(images, image_folder, mbti_data) 
    
    # This condition ensures that the main() function is only called if the script is being run directly and not imported as a module in another script
    if __name__ == "__main__":
        main()

# When the selected option is 'Speech to Art to Speech'
elif selected == 'Speech to Art to Speech':
    
    # The below session states are created so that the outputs above doesn't rerun entirely everytime the buttons are pressed 
    # Initialize the session state variable 'load_state' with a default value of False if it's not already set
    if 'load_state' not in st.session_state:
        st.session_state.load_state = False
        
    # Initialize the session state variable 'generate_state' with a default value of False if it's not already set  
    if 'generate_state' not in st.session_state:
        st.session_state.generate_state = False
    
    # Initialize the session state variable 'load_state_2' with a default value of False if it's not already set
    if 'load_state_2' not in st.session_state:
        st.session_state.load_state_2 = False
    
    # Initialize the session state variable 'generate_state_2' with a default value of False if it's not already set
    if 'generate_state_2' not in st.session_state:
        st.session_state.generate_state_2 = False
    
    # Define tabs in the UI with labels "Impressionism" and "Surrealism"
    tab1, tab2 = st.tabs(["Impressionism", "Surrealism"])
    
    # Set an environment variable for the REPLICATE_API_TOKEN
    os.environ["REPLICATE_API_TOKEN"] = "" # Your REPLICATE_API_TOKEN here
    
    # Cache the result of the function to improve performance and to store the outputs above, so the results do not change everytime the buttons are pressed
    # The function 'generate_image' communicates with a replicate model and generates an image based on the given description
    @st.cache_resource
    def generate_image(img_description):
        output = replicate.run(
            "ryan-koo92/sdxl-ip:530e6d94142dc71729cfb592499c81ba511b3c289c08cece5cb686a764f91862", # SDXL that was trained using impressionist painting images 
            input={"prompt": f"{img_description}"})
        return output
    
    # Similar to the previous function but targets the Surrealism style
    @st.cache_resource
    def generate_image_2(img_description):
        output = replicate.run(
            "ryan-koo92/sdxl-sr:121c2f6a3bbb85b9f34430565fc3b0398a3b4b469e2179fcd6210638089feab0", # SDXL that was trained using surrealist painting images 
            input={"prompt": f"{img_description}"})
        return output
    
    # The function 'generate_text' communicates with another replicate model to generate a textual description of the provided image
    @st.cache_resource
    def generate_text(img_description):
        output = replicate.run(
            "joehoover/mplug-owl:51a43c9d00dfd92276b2511b509fcb3ad82e221f6a9e5806c54e69803e291d6b",
            input={"prompt" : 'This image is an art painting, and please describe this art painting in the order below. First, describe its genre. Genre refers to the subject matter and objects depicted. Second, describe its media. Media refers to materials the artwork is made from, and to techniques used by the artist to create that artwork.  Third, describe its style. Style refers to art movements such as impressionism and surrealism. And finally, describe this art painting as visually accurate as possible.',
                "img":image})
        text=[]
        
        for item in output: # Iterate through the output and store it in a list
            text.append(item)
        text=''.join(text) # Join the list items into a single string and return
        return text
    
    # Function to translate the input text into Korean using the Deepl API
    @st.cache_resource
    def translate_ko(text):
        translator = deepl.Translator('2fc1546f-1428-6006-bf76-feaf45564fb0:fx') # Initialize the translator object with a given API key  
        result = translator.translate_text(text, target_lang='KO') # Translate the input text to Korean
        return result.text # Return the translated text

    # Function to translate the input text into English using the Deepl API
    @st.cache_resource
    def translate_en(text):
        translator = deepl.Translator('2fc1546f-1428-6006-bf76-feaf45564fb0:fx') # Initialize the translator object with a given API key
        result = translator.translate_text(text, target_lang='EN-US') # Translate the input text to English
        return result.text # Return the translated text
    
    # Function to convert the input text into speech using the Naver API and plays the resulting audio
    def naver_clover_tts(text) :
        try:
            # Set the client ID and secret for the Naver API
            client_id = "" # Your Naver CLOVA API key here
            client_secret = "" # Your Naver CLOVA API key here
            
            # Convert the text to a URL-encoded format
            encText = urllib.parse.quote(text)
            
            # Define the audio settings for the speech generation
            data = "speaker=nara&volume=0&speed=0&pitch=0&format=mp3&text=" + encText;
            url = "https://naveropenapi.apigw.ntruss.com/tts-premium/v1/tts"
            
            # Set up the HTTP request
            request = urllib.request.Request(url)
            request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
            request.add_header("X-NCP-APIGW-API-KEY",client_secret)
            
            # Send the request and get the response
            response = urllib.request.urlopen(request, data=data.encode('utf-8'))
            rescode = response.getcode()
            
            # If the API response is successful, save the audio and play it
            if(rescode==200):
                response_body = response.read()
                with open('C:/project2/1113.mp3', 'wb') as f:
                    f.write(response_body)
                playsound('C:/project2/1113.mp3')
            else:
                print("Error Code:" + rescode)
        finally:
            os.remove('C:/project2/1113.mp3') # Clean up by deleting the generated audio file
    
    # Inside the 'Impressionism' tab
    with tab1 :
    
        st.title("Speech to Art to Speech") # Set the main title for the section
        st.subheader(':blue[Impressionism] :male-artist:') # Set a subheader
    
        st.markdown("""---""") # A divider 
        
        # Capture the audio input from the user
        wav_audio_data = st_audiorec()
        
        # If the user provides audio data or the session state load_state is True
        if wav_audio_data or st.session_state.load_state:
            
            st.session_state.load_state = True # Set the session_state.load_state as True
            
            try : 
                
                # Store the captured audio temporarily
                temp_audio_file = "temp_audio.wav"
                with open(temp_audio_file, "wb") as f:
                    f.write(wav_audio_data)
                
                speech_sr = SR.Recognizer() # Setup a speech recognition object
                
                # Extract audio data from the file
                with SR.AudioFile(temp_audio_file) as source:
                    audio = speech_sr.record(source)
                    
                # Convert the audio to text using Google's speech recognition
                text = speech_sr.recognize_google(audio_data=audio, language='ko-KR')
                
                st.markdown("""---""") # A divider
                
                # Display the recognized text
                st.markdown("<h3 style='text-align: left; color: black;'> 입력된 음성 : <br></h3>", unsafe_allow_html=True)
                st.write(text)
                
                translated_text = translate_en(text) # Translate the recognized text to English
                
                # Display the translated text
                st.markdown("<h3 style='text-align: left; color: black;'> 한영 번역: <br></h3>", unsafe_allow_html=True)
                st.write(translated_text)
                
                st.markdown("""---""") # A divider
                
                os.remove(temp_audio_file) # Remove the temporary audio file
                
                # Allow users to modify the translated text before generating an artwork
                img_description = st.text_input(label='Image Description', value=translated_text)
                
                generate = st.button('Generate Impressionist Painting')
                
                # If the 'Generate' button is clicked or the session state generate_state is True
                if generate or st.session_state.generate_state:
                    
                    st.session_state.generate_state = True # Set the session_state.generate_state as True
                    
                    st.markdown("""---""") # A divider
                    
                    # Generate an image based on the provided description
                    generated_img = generate_image(img_description)
                    
                    # Display the generated image
                    st.markdown("<h3 style='text-align: left; color: black;'> 그림 작품 : <br></h3>", unsafe_allow_html=True)
                    st.image(generated_img)
                    
                    image= generated_img[0] # Extract the first item from the generated_img list ????
                    
                    text = generate_text(image) # Describe the generated image
                    
                    st.markdown("""---""") # A divider
                    
                    # Display the description of the image
                    st.markdown("<h3 style='text-align: left; color: black;'> 그림에 대한 설명 : <br></h3>", unsafe_allow_html=True)
                    st.write(text)
                    
                    # Translate the image description back to Korean
                    translated_text_2 = translate_ko(text)
                    
                    # Display the Korean translation
                    st.markdown("<h3 style='text-align: left; color: black;'> 영한 번역 : <br></h3>", unsafe_allow_html=True)
                    st.write(translated_text_2)
                    
                    st.write('') # To provide Spacing between the texts and the button
                    
                    # Button to playback the description as audio
                    button = st.button(label='음성지원 :mega:') 
                    
                    st.markdown("""---""") # A divider 
                    
                    # If the playback button is clicked, convert the text to speech and play the audio
                    if button :
                        naver_clover_tts(translated_text_2)
                        
                else:
                    pass
            
            # If an error occurs, inform the user
            except :
                st.write('다시 시도해 주시길 바랍니다.')
                
        else:
            pass
   
    # Inside the 'Surrelaism' tab; Same thing with the Impressionism tab
    with tab2 :
        
        st.title("Speech to Art to Speech")
        st.subheader(':red[Surrealism] :art:')

        st.markdown("""---""")
        
        wav_audio_data = st_audiorec1()
        
        if wav_audio_data or st.session_state.load_state_2:
            st.session_state.load_state_2 = True
            try : 
                temp_audio_file = "temp_audio.wav"
                with open(temp_audio_file, "wb") as f:
                    f.write(wav_audio_data)
                
                speech_sr = SR.Recognizer()
                
                with SR.AudioFile(temp_audio_file) as source:
                    audio = speech_sr.record(source)
                    
                text = speech_sr.recognize_google(audio_data=audio, language='ko-KR')
                
                st.markdown("""---""")
                
                st.markdown("<h3 style='text-align: left; color: black;'> 입력된 음성 : <br></h3>", unsafe_allow_html=True)
                st.write(text)
                
                translated_text = translate_en(text)
                
                st.markdown("<h3 style='text-align: left; color: black;'> 한영 번역: <br></h3>", unsafe_allow_html=True)
                st.write(translated_text)
                
                st.markdown("""---""")
                
                os.remove(temp_audio_file)
                
                img_description = st.text_input(label='Image Description', value=translated_text)
                
                generate_2 = st.button('Generate Surrealist Painting')
                            
                if generate_2 or st.session_state.generate_state_2:
                    st.session_state.generate_state_2 = True
                    st.markdown("""---""")
                    generated_img = generate_image_2(img_description)
                    st.markdown("<h3 style='text-align: left; color: black;'> 그림 작품 : <br></h3>", unsafe_allow_html=True)
                    st.image(generated_img)
                    
                    image= generated_img[0]
                    
                    text = generate_text(image)
                    
                    st.markdown("""---""")
                    
                    st.markdown("<h3 style='text-align: left; color: black;'> 그림에 대한 설명 : <br></h3>", unsafe_allow_html=True)
                    st.write(text)
                    
                    translated_text_2 = translate_ko(text)
                    
                    st.markdown("<h3 style='text-align: left; color: black;'> 영한 번역 : <br></h3>", unsafe_allow_html=True)
                    st.write(translated_text_2)
                    
                    st.write('')
                    
                    button = st.button(label='음성지원 :loudspeaker:') 
                    
                    st.markdown("""---""")
                    
                    if button :
                        naver_clover_tts(translated_text_2)
                        
                else:
                    pass
            
            except :
                st.write('다시 시도해 주시길 바랍니다..')
                
        else:
            pass

# When the selected option is 'Art Chatbot'
elif selected == 'Art Chatbot' :
    
    # Begin a form for the chatbot interface using Streamlit's form functionality
    with st.form('chatbot') :
        OP=Options() # Initialize Options for Chrome WebDriver, which will control a browser session
        OP.add_argument('--headless=new') # Set Chrome to run in a "headless" mode, which means it runs in the background without displaying the browser GUI
        
        # Set browser preferences to disable certain functionalities and prompts, for smoother browsing
        OP.add_experimental_option(
                                    "prefs",
                                    {
                                        "credentials_enable_service": False, # Disable the credential manager
                                        "profile.password_manager_enabled": False, # Disable the password manager
                                        "profile.default_content_setting_values.notifications": 2 # Disable notifications           
                                    },
                                )
        
        # More Chrome Options to refine the browsing experience
        OP.add_argument('--disable-notifications') # Disable all notifications
        OP.add_argument("--disable-infobars") # Disable infobars on top of the browser
        OP.add_argument("--disable-extensions") # Disable extensions
        OP.add_argument("--start-maximized") # Start the browser maximized
        OP.add_argument("--window-size=1920,1080") # Set default window size
        OP.add_argument('--ignore-certificate-errors') # Bypass any SSL certificate errors
        OP.add_argument('--allow-running-insecure-content') # Allow running content even if insecure
        OP.add_argument("--disable-web-security") # Disable web security settings
        OP.add_argument("--disable-site-isolation-trials") # Disable site isolation trials
        OP.add_argument("--user-data-dir=C:\\Users\\home\\Downloads") # Set directory for user data
        OP.add_argument("--disable-features=NetworkService,NetworkServiceInProcess") # Disable certain network-related features
        OP.add_argument("--test-type") # Indicate the browser is being run for tests
        OP.add_argument('--no-sandbox') # Disable sandboxing features
        OP.add_argument('--disable-gpu') # Disable GPU acceleration
        OP.add_argument('--profile-directory=Default')  # Set default profile directory
        OP.add_argument('--user-data-dir=C:/Temp/ChromeProfile') # Set temp directory for user data
        
        
        # The `retry` decorator is used to retry the function if it fails
        # This is particularly useful for API calls that might occasionally timeout or fail
        # Here, the function will be retried with a random exponential backoff waiting time between retries, ranging from 1 to 60 seconds
        # The function will stop retrying after 6 attempts
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def completion_with_backoff(**kwargs): # This function sends a request to the OpenAI API to get a completion (prediction) based on the given input
            return openai.Completion.create(**kwargs)
        
        # Set the API key for OpenAI. This key is used to authenticate with the OpenAI API
        openai.api_key = "" # Your OpenAI API key here
        
        # Load an art-related CSV file from the given path
        uploaded_file = 'C:/project2/v2_new_art_df.csv'
        art_df = pd.read_csv(uploaded_file, encoding='utf-8')
        
        df_ref = art_df.Reference # Extract the 'Reference' column and store it separately
        df = art_df.drop(columns=['Reference'])  # Drop the 'Reference' column from the dataframe
        
        # Extract the structure and information of the dataframe `df` into a string format.
        # This is useful for understanding the dataframe's columns, data types, and other attributes
        buffer = io.StringIO() # Create an in-memory text buffer
        df.info(buf = buffer) # Write the dataframe's information to the buffer
        info_str = buffer.getvalue() # Convert the buffer content to a string and store it in `info_str`
        
        # The function 'return_answer' is designed to process and respond to user input related to art
        def return_answer(user_input):
            
            # If the user provides an input
            if user_input:
                
                try:
                    my_bar=st.progress(0, text='질문 내용 분석중...') # Initialize a progress bar on the front-end with the message '질문 내용 분석중...' (Analyzing the question...)
                    
                    # Define a prompt to check if the user's question is about art
                    pre_prompt_1=f'''If the question below is about art, return 0, otherwise return 1.
        
                                    "{user_input}" 
                                    '''
                    
                    # Send the prompt to OpenAI's model for evaluation
                    response = completion_with_backoff(
                                                      model="text-davinci-003",
                                                      prompt=pre_prompt_1,
                                                      temperature=1,
                                                      max_tokens=10,
                                                      top_p=1,
                                                      frequency_penalty=0,
                                                      presence_penalty=0
                                                    )
                    
                    # Extract the response from the model
                    qna_c=int(response.choices[0].text.strip())
                    
                    # If the model thinks the question is about art
                    if qna_c == 0 : 
                        # Update the progress bar with a new message
                        my_bar.progress(10, text='미술 쪽 질문이 맞는 것 같습니다.') 
                        
                        # After confirming that the user's input is related to art, further categorization is performed
                        # The second prompt is designed to determine if the user's question pertains to specific art topics:
                        # Art movements, characteristics of art, famous artists, notable artworks, or the historical context of art
                        pre_prompt_2=f'''If there is a question related to any of the following: "Art movements, characteristics of art, artists in art, representative art works, and the historical context of art," please output 0; otherwise, output 1.
        
                                            "{user_input}" 
                        
                                            '''
                        # The prompt is then sent to OpenAI's Davinci model to get a more specific categorization                   
                        response_1 = completion_with_backoff(
                                                              model="text-davinci-003",
                                                              prompt=pre_prompt_2,
                                                              temperature=1,
                                                              max_tokens=10,
                                                              top_p=1,
                                                              frequency_penalty=0,
                                                              presence_penalty=0
                                                            )
                        
                        # The model's response is then extracted and converted to an integer for easier processing
                        qna_c_1 = int(response_1.choices[0].text.strip())

                        # If the question is identified to be related to art specifics                        
                        if qna_c_1 == 0:
                            
                            # Update the progress bar message
                            my_bar.progress(25, text='KTA팀의 Database를 사용하겠습니다.')
                            
                            # Given a user input, the objective is to generate an SQL query that fetches the relevant content from a DataFrame 'df'
                            
                            # The prompt details the user input, and provides guidelines to the model about the DataFrame's structure and content
                            # It further instructs the model to generate an SQL query based on this information
                            prompt_1 = f'''Look at this carefully, while thinking step by step.
                            
                                        "{user_input}" 
                                        
                                        We'll call this a 'Input' from now on.
                                        I need a sqlite3 query to retrieve the contents related to this 'Input' from a pandas.DataFrame named 'df'. Below are the df.info(), df.head() df.iloc[0].T of df that you should refer to when writing your query. Note that df.head() is created exactly as it is in df.
                                        
                                        {info_str}
                                        
                                        {df.head()}
                                        
                                        {df.iloc[0].T}
                                        
                                        I'll teach you some precautions when writing queries. You must follow these precautions when answering. Any answer that does not follow these precautions will greatly interfere with my work.
                                        
                                        1. write the query by interpreting or translating the language appropriately to make it easier to apply the above question to df.
                                        If you need to rephrase the question in your query, be sure to refer to the texts described in df.head(), df.iloc[0].T, and df.info().
                                        
                                        2. Always write query statements using only the information in df.
                                        
                                        3. answer the query in the form of a "1-line sql query" starting with 'SELECT' and ending with ';'. Do not attach any text other than the requested answer. Do not include any '\n' or line breaks. Your answer should look like a one-line SQL query.
                                        '''    
                            
                            # The prompt is passed to OpenAI's Davinci model to get an SQL query based on the provided instructions and DataFrame structure
                            response_1 = completion_with_backoff(
                                                                  model="text-davinci-003",
                                                                  prompt=prompt_1,
                                                                  temperature=1,
                                                                  max_tokens=256,
                                                                  top_p=1,
                                                                  frequency_penalty=0,
                                                                  presence_penalty=0
                                                                )    
                            
                            # Extract the SQL query from the model's response using regex
                            criteria=re.search(r'SELECT.*?;',response_1.choices[0].text.strip())
                            if criteria:
                               response_2=' '.join(criteria.group().split())
                            ref_genre=[]
                            
                            # Check if any genre in the dataframe matches the response
                            for i in list(df.사조):
                                if i in response_2:
                                    ref_genre.append(i)
                            
                            # Update the progress bar message
                            my_bar.progress(50, text='데이터 조사 준비 완료')
                            
                            result = ''
                            
                            # Use SQLite to query the data
                            if response_2: 
                                conn = sqlite3.connect(':memory:') # Establish an in-memory SQLite connection
                                df.to_sql('df', conn, index=False) # Convert the pandas DataFrame 'df' into a SQL table named 'df' within the SQLite memory database
                                query = response_2 # Extracting the SQL query from the response
                                cursor = conn.cursor() # Initialize the database cursor
                                cursor.execute(query) # Execute the SQL query on the database
                                result_rows = cursor.fetchall() # Fetch all rows resulting from the query
                                columns = [desc[0] for desc in cursor.description] # Extract column names from the cursor's description
                                result_df = pd.DataFrame(result_rows, columns=columns) # Convert the result rows and columns into a pandas DataFrame
                                
                                # Iterate over the rows of the result DataFrame to compile the results into the string
                                for i, row in result_df.iterrows():
                                    result += f"index_number: {i} \n"
                                    for column in result_df.columns:
                                        result += f"{column}: {row[column]}"
                                        
                                        # Add a newline character after each column value, except for the last column
                                        if column != result_df.columns[-1]:
                                            result += " \n"
                                    result += " \n\n" # Add two newline characters after each row
                                    
                                # If the result string remains empty after processing, raise a ValueError    
                                if result == '':
                                    raise ValueError()
                            
                            # Update the progress bar message
                            my_bar.progress(75, text='데이터 조사 완료.')
                            
                            # Define another prompt to format the response in Korean based on the results extracted from the DataFrame
                            prompt_2 = f'''The following content is written as a single line of text for a dataframe:
                                
                                            {result}
            
                                            Using the information above, respond to the text below in Korean language, written in fluent and correct grammar.
                                            
                                            "{user_input}"'''
                            
                            # Update progress bar to inform the user that the system is processing their input
                            my_bar.progress(90, text='답변을 생각하는 중..')
                            
                            # Request a response from the OpenAI model based on the provided prompt
                            response_3 = completion_with_backoff(
                                                                  model="text-davinci-003",
                                                                  prompt=prompt_2,
                                                                  temperature=1,
                                                                  max_tokens=900,
                                                                  top_p=1,
                                                                  frequency_penalty=0,
                                                                  presence_penalty=0
                                                                )
                            
                            # Extract the generated text from the model's response
                            final_response = response_3.choices[0].text.strip()
                            
                            # If the query matched any genres from the DataFrame, add the related references to the response
                            if ref_genre:
                                final_response=final_response+'\n\n\n[Reference]\n'+'\n'.join(art_df.loc[art_df['사조'].isin(ref_genre),'Reference'].tolist())
                            
                            # If there are no genre-specific references, cite the general database
                            else:
                                final_response=final_response+'\n\n\n[Reference]\n'+'KTA팀 DB'
                            
                            # Update progress bar to inform the user that the processing is complete
                            my_bar.progress(100, text='답변 완료.')
                        
                        # If the user's input is related to art but not about "Art movements, characteristics of art, artists in art, representative art works, and the historical context of art"
                        if qna_c_1 == 1:
                            
                            # Update the progress bar to indicate that the system is switching to web search mode
                            my_bar.progress(25, text='Web Search 모드 실행.') 
                            
                            # A prompt is defined to get the appropriate Korean search term based on the user's input
                            search_prompt = f'''Below is a question. Replace it with an Korean search term that the questioner would use to get the answer they want when searching on Google, and most important expression in your Korean search term should be enclosed in ".
                            
                            {user_input} 
                            '''
                            
                            # OpenAI's model is requested to provide a suitable search term based on the user's input
                            search_response = completion_with_backoff(
                                                                      model="text-davinci-003",
                                                                      prompt=search_prompt,
                                                                      temperature=1,
                                                                      max_tokens=256,
                                                                      top_p=1,
                                                                      frequency_penalty=0,
                                                                      presence_penalty=0
                                                                    )    
                            
                            # Extract the refined search term from the model's response
                            search_response_1=search_response.choices[0].text.strip()
                            
                            # Update the progress bar to indicate the refined search term
                            my_bar.progress(30, text=f'"{search_response_1}" 검색중..')
                            
                            # Initialize the Chrome browser to search Google with the refined term
                            driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=OP)
                            driver.get(f'https://www.google.com/search?q={search_response_1}')
                            
                            # Wait until the search results are visible
                            try:
                                WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="result-stats"]')))
                            
                            finally:
                                # Attempt to click on the top results to gather more detailed information from the search results
                                for i in range(1, 4):
                                    try:
                                        element = driver.find_element(By.XPATH, f'//*[@id="rso"]/div[{i}]/div/div/div[1]/div/div/span/a')
                                        element.send_keys(Keys.ENTER)
                                        break
                                    except:
                                        pass
                           
                            # Wait for the full content of the webpage to load by waiting for the 'body' tag to be visible
                            WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))
                            
                            # Update the progress bar to indicate that data analysis from the web page has started
                            my_bar.progress(50, text='조사 자료 분석중..')
                            
                            # Fetch the source HTML of the webpage
                            html=driver.page_source
                            
                            # Use BeautifulSoup to parse the HTML
                            soup=BeautifulSoup(html,'html.parser')
                            del html # Since we have already loaded the HTML content into 'soup', the 'html' variable is no longer needed and can be deleted to free up memory

                            temp_soup=soup.select_one('body').text # Extract the text content of the 'body' tag from the parsed HTML
                            my_bar.progress(85, text='최종 자료를 기반한 답변 작성중..') # Update the progress bar to indicate that a response is being formulated based on the extracted data
                            
                            # Use OpenAI's model to generate a response. The model is guided with a set of messages, indicating its role and the user's intent
                            # The user's original question and the extracted content from the webpage are provided as resources to the model
                            response_test = openai.ChatCompletion.create(
                                                                          model="gpt-3.5-turbo-16k",
                                                                          messages=[
                                                                            {
                                                                              "role": "system",
                                                                              "content": "You are a helpful assistant."
                                                                            },
                                                                            {
                                                                              "role": "user",
                                                                              "content": f"""Find the answers in the resources I provide and answer my questions in Korean with correct grammar. Below are the questions and resources.
                                                                              Question : {user_input}
                                                                              
                                                                              Resources : 
                                                                              {temp_soup}"""
                                                                            }
                                                                          ],
                                                                          temperature=1,
                                                                          max_tokens=512,
                                                                          top_p=1,
                                                                          frequency_penalty=0,
                                                                          presence_penalty=0
                                                                        )
                                
                            # Extract the formulated response from the model's output    
                            search_resp_fin = response_test['choices'][0]['message']['content']
                            
                            # If there's a valid response from the web page scraping and OpenAI model querying:
                            if search_resp_fin:
                                # Compile the final response with the generated answer and add references (page title and URL)
                                final_response=search_resp_fin+'\n\n[Reference]\n\n'+f"{driver.title}\n\n"+f"{driver.current_url}"
                                driver.quit() # Close the browser instance
                                my_bar.progress(100, text='답변 완료.') # Update the progress bar to indicate completion
                            
                            # If no satisfactory response could be generated
                            else :
                                final_response='좋은 답변으로 삼을만한 Reference를 찾지 못했습니다. 죄송합니다.' # Provide a default apology response
                                driver.quit() # Close the browser instance
                                my_bar.progress(100, text='답변 완료.') # Update the progress bar to indicate completion
                    
                    # If the initial categorization suggests the user's question isn't about art:
                    if qna_c == 1:
                        my_bar.progress(100, text='분석 완료') # Update the progress bar to indicate the analysis is done
                        # Provide a default response indicating the mismatch in the expected question type
                        final_response='''미술과 관련된 궁금증이나 질문을 주신 것이 확실한가요?\n전 <b>요청사항의 형식</b>, <b>미술과 관련이 없는 내용</b>, 또는 <b>궁금증이나 질문이 아닌 것</b>들을 답변할 수 없습니다.'''
                
                # Handling exceptions
                except Exception:
                    
                    # If any unexpected error occurs, attempt a web search to provide a more reliable answer
                    my_bar.progress(50, text='좀더 확실한 답변을 위해 Web Search 모드가 실행됩니다.')
                    
                    # Form a prompt to refine the search term for Google search
                    search_prompt = f'''Below is a question. Replace it with an Korean search term that the questioner would use to get the answer they want when searching on Google, and most important expression in your Korean search term should be enclosed in ".
                    
                    {user_input} 
                    '''
                    
                    # Send the prompt to OpenAI's model for refinement
                    search_response = completion_with_backoff(
                                                              model="text-davinci-003",
                                                              prompt=search_prompt,
                                                              temperature=1,
                                                              max_tokens=256,
                                                              top_p=1,
                                                              frequency_penalty=0,
                                                              presence_penalty=0
                                                            )    
                    
                    # Extract the refined search term from the model's response
                    search_response_1=search_response.choices[0].text.strip()
                    
                    # Update the progress bar with searching the 'search_response_1'
                    my_bar.progress(60, text=f'"{search_response_1}" 검색중..')
                    
                    # Initialize a Chrome browser instance and search for the refined term on Google
                    driver=webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=OP)
                    driver.get(f'https://www.google.com/search?q={search_response_1}')
                    
                    # Using a 'try-finally' block to navigate and search Google results with the refined search term
                    try:
                        # Waiting for the search results page to load by looking for an element identified by its XPATH
                        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="result-stats"]')))
                    
                    finally:
                        # Loop to try and open the first three search results
                        for i in range(1, 4):
                            try:
                                # Locating the search result element by its XPATH and simulating the 'ENTER' key to visit the page
                                element = driver.find_element(By.XPATH, f'//*[@id="rso"]/div[{i}]/div/div/div[1]/div/div/span/a')
                                element.send_keys(Keys.ENTER)
                                break
                            except:
                                pass
                    
                    # Waiting for the selected web page to load by checking for the visibility of the 'body' element
                    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))
                    
                    # Updating the progress bar as analyzing the searched result 
                    my_bar.progress(70, text='조사한 자료 분석중..')
                    html = driver.page_source # Extracting the entire page source
                    soup = BeautifulSoup(html,'html.parser') # Parsing the HTML using BeautifulSoup to extract relevant content
                    
                    del html # Deleting the 'html' variable to free memory
                    
                    temp_soup=soup.select_one('body').text # Extracting the textual content from the 'body' tag
                    
                    my_bar.progress(95, text='최종 자료를 기반한 답변 작성중..') # Updating the progress bar with a custom message
                    
                    # Sending the scraped content to OpenAI's model for evaluation and response generation
                    response_test = openai.ChatCompletion.create(
                                                                  model="gpt-3.5-turbo-16k",
                                                                  messages=[
                                                                    {
                                                                      "role": "system",
                                                                      "content": "You are a helpful assistant."
                                                                    },
                                                                    {
                                                                      "role": "user",
                                                                      "content": f"""Find the answers in the resources I provide and answer my questions in Korean with correct grammar. Below are the questions and resources.
                                                                      Question : {user_input}
                                                                      
                                                                      Resources : 
                                                                      {temp_soup}"""
                                                                    }
                                                                  ],
                                                                  temperature=1,
                                                                  max_tokens=512,
                                                                  top_p=1,
                                                                  frequency_penalty=0,
                                                                  presence_penalty=0
                                                                )
                        
                    # Extracting the model's response from the result    
                    search_resp_fin=response_test['choices'][0]['message']['content']
                    
                    # Compiling the final response and closing the webdriver
                    if search_resp_fin:
                        final_response=search_resp_fin+'\n\n[Reference]\n\n'+f"{driver.title}\n\n"+f"{driver.current_url}"
                        driver.quit()
                        my_bar.progress(100, text='답변 완료.')
                    else :
                        final_response='좋은 답변으로 삼을만한 Reference를 찾지 못했습니다. 죄송합니다'
                        driver.quit()
                        my_bar.progress(100, text='답변 완료.')
                        
            # Displaying the final response on the web application
            return st.markdown(f'<p>{final_response}</p>',unsafe_allow_html=True)
        
        # Define a three-column layout with the middle column being the widest
        col1,col2,col3=st.columns([0.5,9,0.5])
        
        # Set the title in the middle column
        with col2:
            st.title("더 궁금하신 것이 있으신가요?")
        
        # Define a two-column layout
        col1,col2 = st.columns([8.75,1.25])
        
        with col1:
            
            # If the 'key' does not exist in session state, initialize it with 'value'
            if 'key' not in st.session_state:
                st.session_state['key'] = 'value'
            
            # Get input from the user regarding their question related to art
            your_question_input = st.text_input("미술과 관련된 모든 궁금증과 질문들을  작성하신 후 엔터를 눌러주세요!")
        
        with col2:
            st.subheader('') # Provide spacing
            submitted = st.form_submit_button("Submit") # Define a submit button in the second column
        
        if submitted:
            try:
                st.session_state['key'] = 'value_1' # session state of 'key' becomes 'value_1'
                st.divider() # A divider
                return_answer(your_question_input) # Process and return the answer for the given question using the defined function above
            except:
                st.empty()  # Clear the output in case of any exception
    
    try:
        if st.session_state['key'] == 'value_1': # Check if the session state indicates that a user has submitted a question
            
            # Get user feedback about the application using radio buttons
            user_feedback = st.radio('맘에 드시는 기능인가요?',['맘에 드셨다면 네, 싫으셨다면 아니오를 골라주세요.','네','아니오'])
            
            # If the user liked the service
            if user_feedback=='네': 
                st.write('사용해주셔서 감사합니다. 다른 질문을 또 해주세요!')
                time.sleep(3) # Pause for 3 seconds to allow the user to read the message
                
                # Clear the session state
                for key in st.session_state.keys():
                    del st.session_state[key]
                
                # Reset the session state key if it doesn't exist
                if 'key' not in st.session_state:
                    st.session_state['key'] = 'value'
            
            # If the user didn't like the service
            elif user_feedback == "아니오":
                st.divider() # A divider
                st.empty() # Clear any existing content 
                st.title('저희 팀에게 피드백을 주세요!') # Provide a title prompting user feedback.
                
                # Provide additional instructions to the user
                st.markdown("""
                **사용하시면서 불편하셨던 점에 대한 의견을 전부 적어서 저희들에게 알려주세요. 더 나은 서비스로 보답드릴 것을 약속드립니다**
                """)
                
                # Define a contact form for the user to submit detailed feedback
                contact_form="""<form action="https://formsubmit.co/chohk4198@gmail.com" method="POST">
                     <label for="message">Feedback</label><br>
                     <textarea id="message" name="message" rows="10" cols="100" placeholder="여기에 메시지를 입력하세요."  required></textarea><br><br>
                     <label for="email">email</label><br>
                     <input type="email" name="email" size='81' required>
                     <button type="submit">Send</button>
                </form>"""
                
                # Display a contact form for detailed feedback
                st.markdown(contact_form,unsafe_allow_html=True)  
            
            # If the user hasn't chosen a feedback option
            elif user_feedback =='맘에 드셨다면 네, 싫으셨다면 아니오를 골라주세요.': 
                st.session_state['key'] == 'value_1'  # Maintain the current session state
    
    except:
        st.empty() # Clear the output in case of any exception