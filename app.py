import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from gtts import gTTS
import base64

# Səhifəni başlat
st.set_page_config(page_title="Görən Göz AI", layout="wide")
st.title("👁️ Görən Göz - Stabil Versiya")

# Modeli yüklə (Nano versiya ən sürətlisidir)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Səs funksiyası
def speak(text):
    try:
        tts = gTTS(text=text, lang='az')
        tts.save("temp.mp3")
        with open("temp.mp3", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Səs xətası: {e}")

# Kamera interfeysi
img_file = st.camera_input("Şəkil çəkin və ya kameranı açın")

if img_file:
    img = Image.open(img_file)
    img_array = np.array(img)
    
    # Süni İntellekt Analizi
    results = model(img_array)
    
    # Nəticələri emal et
    found_objects = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = results[0].names[class_id]
            
            # Azərbaycan dilinə tərcümə
            translations = {"person": "İnsan", "car": "Maşın", "bus": "Avtobus", "cell phone": "Telefon", "bottle": "Butulka"}
            found_objects.append(translations.get(label, label))
        
        detected_text = "Görürəm: " + ", ".join(list(set(found_objects)))
        st.success(detected_text)
        speak(detected_text)
    else:
        st.info("Heç nə tapılmadı.")
        speak("Yol təmizdir")
