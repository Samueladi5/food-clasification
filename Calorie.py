import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from fuzzywuzzy import process
from PIL import Image
import io

# Load model CNN (pastikan path-nya benar)
@st.cache_resource
def load_model():
    model_path = r'C:\Streamlit\AppCalorie\Clasification_new_food.h5'  
    with st.spinner("Loading model..."):
        model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
    return model

# Load dataset ABBREV (pastikan path-nya benar)
@st.cache_data
def load_abbrev():
    return pd.read_csv('C:\Streamlit\AppCalorie\ABBREV.csv')

# Daftar label makanan
food_labels = [
   'ramen', 'falafel', 'french_toast', 'ice_cream', 'bibimbap', 'omelet',
    'cannoli', 'sushi', 'fried_chicken', 'fried_rice', 'rice', 'apple_pie',
    'tiramisu', 'edamame'
]

def find_best_match(food_name, choices):
    return process.extractOne(food_name, choices)[0]

def predict_food_and_calories(img_array, model, abbrev_df):
    # Prediksi dengan model CNN
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_food = food_labels[predicted_class]
    
    # Cari makanan yang cocok di ABBREV menggunakan fuzzy matching
    best_match = find_best_match(predicted_food, abbrev_df['Shrt_Desc'].tolist())
    matching_food = abbrev_df[abbrev_df['Shrt_Desc'] == best_match]
    
    if not matching_food.empty:
        calories = matching_food['Energ_Kcal'].values[0]
        return predicted_food, calories, best_match
    else:
        return predicted_food, "Kalori tidak ditemukan", None

def main():
    st.title('Klasifikasi Makanan dan Prediksi Kalori')
    
    model = load_model()
    abbrev_df = load_abbrev()
    
    uploaded_file = st.file_uploader("Pilih gambar makanan...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
        
        # Preprocessing gambar
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.
        
        # Prediksi
        predicted_food, calories, matched_food = predict_food_and_calories(img_array, model, abbrev_df)
        
        st.write(f"Makanan terdeteksi: {predicted_food}")
        if matched_food:
            st.write(f"Makanan yang cocok di database: {matched_food}")
            st.write(f"Perkiraan kalori: {calories}")
        else:
            st.write("Kalori tidak ditemukan dalam database.")

if __name__ == '__main__':
    main()