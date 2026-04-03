import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from gtts import gTTS
import base64
import os

# Səhifə Ayarları
st.set_page_config(page_title="Görən Göz AI", layout="wide")
st.title("👁️ Görən Göz - Sürətli Versiya")

# Modeli Yüklə (YOLOv8 Nano - Ən yüngül və sürətli model)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

def speak(text):
    try:
        tts = gTTS(text=text, lang='az')
        tts.save("temp.mp3")
        with open("temp.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
            st.markdown(md, unsafe_allow_html=True)
    except:
        pass

# Kamera Girişi
img_file = st.camera_input("Kameranı Aktivləşdirin")

if img_file:
    img = Image.open(img_file)
    # Analiz
    results = model(img)
    
    names = results[0].names
    detected_indices = results[0].boxes.cls.cpu().numpy()
    
    if len(detected_indices) > 0:
        translations = {"person": "İnsan", "bus": "Avtobus", "car": "Maşın", "cell phone": "Telefon"}
        found = []
        for idx in detected_indices:
            name = names[int(idx)]
            found.append(translations.get(name, name))
        
        result_text = "Görürəm: " + ", ".join(list(set(found)))
        st.success(result_text)
        speak(result_text)
    else:
        speak("Yol təmizdir.")
