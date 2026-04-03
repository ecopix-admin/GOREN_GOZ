import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from gtts import gTTS
import base64

# Səhifə başlığı
st.title("👁️ Görən Göz AI")

# Modeli yüklə (Ən yüngül versiya)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Səs funksiyası
def speak(text):
    tts = gTTS(text=text, lang='az')
    tts.save("voice.mp3")
    with open("voice.mp3", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">', unsafe_allow_html=True)

# Kamera
img_file = st.camera_input("Kameranı açın")

if img_file:
    img = Image.open(img_file)
    results = model(img)
    
    # Nəticəni tap
    names = results[0].names
    detected = [names[int(box.cls[0])] for box in results[0].boxes]
    
    if detected:
        # Tərcümə lüğəti
        dic = {"person": "İnsan", "car": "Maşın", "bus": "Avtobus", "cell phone": "Telefon"}
        objects = [dic.get(o, o) for o in set(detected)]
        cavab = "Görürəm: " + ", ".join(objects)
        st.success(cavab)
        speak(cavab)
    else:
        st.info("Heç nə tapılmadı.")
        speak("Yol təmizdir")
