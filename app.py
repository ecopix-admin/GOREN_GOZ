import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from gtts import gTTS
import base64
import time
import re

# 1. SƏHİFƏ AYARLARI
st.set_page_config(page_title="Görən Göz AI", layout="wide")
st.title("👁️ Görən Göz - Sizin Şəxsi Dostunuz")

# 2. SÜNİ İNTELLEKT MODELİNİ YÜKLƏ (YOLOv5)
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# 3. AVTOMATİK SƏS FUNKSİYASI
def speak(text):
    tts = gTTS(text=text, lang='az')
    tts.save("speech.mp3")
    with open("speech.mp3", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
        st.markdown(md, unsafe_allow_html=True)

# 4. ASİSTENTİN İLK SALAMI (Yalnız bir dəfə)
if 'greeted' not in st.session_state:
    speak("Salam Şamxal! Görən Göz aktivdir. Mən hər şeyi izləyirəm.")
    st.session_state['greeted'] = True

# 5. CANLI KAMERA VƏ ANALİZ
img_file = st.camera_input("Kamera Həmişə Açıqdır", label_visibility="hidden")

if img_file:
    img = Image.open(img_file)
    img_array = np.array(img)
    
    # AI Analizi
    results = model(img_array)
    df = results.pandas().xyxy[0]
    
    # Obyektləri səsləndir
    if not df.empty:
        translations = {"person": "İnsan", "bus": "Avtobus", "car": "Maşın", "bottle": "Butulka", "cell phone": "Telefon"}
        detected = [translations.get(obj, obj) for obj in df['name'].unique()]
        msg = "Görürəm: " + ", ".join(detected)
        st.write(f"### {msg}")
        speak(msg)
        
        # Əgər kontaktlardan birini görsə (Simulyasiya)
        if "İnsan" in msg:
            st.warning("Kontaktlarınızdakı Əliyə oxşayır. Zəng edim?")
            speak("Qarşınızda bir insan var. Kontaktlarınızdakı Əliyə oxşayır. Zəng edim?")

# 6. SƏSLİ ƏMRLƏR (Böyük Düymələr)
st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("📞 Zəng et / Aç", use_container_width=True):
        speak("Kontaktlar axtarılır... Əliyə zəng edilir.")
with c2:
    if st.button("💬 WhatsApp Oxu", use_container_width=True):
        speak("WhatsApp mesajınız var: Salam, necəsən? Axşam görüşə bilərik?")
with c3:
    if st.button("❌ Zəngi Bitir", use_container_width=True):
        speak("Zəng başa çatdı.")
