import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from gtts import gTTS
import base64

# Proqramın adı
st.title("👁️ Görən Göz AI")

# Modeli ən sadə yolla yüklə
@st.cache_resource
def get_model():
    return YOLO("yolov8n.pt")

model = get_model()

# Səs funksiyası
def play_voice(text):
    tts = gTTS(text=text, lang='az')
    tts.save("v.mp3")
    with open("v.mp3", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">', unsafe_allow_html=True)

# Şəkil çəkmə hissəsi
img_data = st.camera_input("Kameranı açın")

if img_data:
    img = Image.open(img_data)
    results = model(img)
    
    # Nəticəni analiz et
    if len(results[0].boxes) > 0:
        names = results[0].names
        detected = [names[int(box.cls[0])] for box in results[0].boxes]
        
        # Tərcümə
        tr = {"person": "İnsan", "car": "Maşın", "bus": "Avtobus", "cell phone": "Telefon"}
        final_list = [tr.get(i, i) for i in set(detected)]
        
        msg = "Görürəm: " + ", ".join(final_list)
        st.write(msg)
        play_voice(msg)
    else:
        play_voice("Yol təmizdir")
