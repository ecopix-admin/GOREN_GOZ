import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import time

st.set_page_config(page_title="GörənGöz AI", layout="wide")

st.title("👁️ GörənGöz AI")

# MODEL
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# SƏS FUNKSİYA
def speak(text):
    tts = gTTS(text=text, lang='tr')
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tts.save(tmp.name)
    audio = open(tmp.name, "rb")
    st.audio(audio.read(), format="audio/mp3")

# STATE
if "last" not in st.session_state:
    st.session_state.last = ""
    st.session_state.time = 0

run = st.checkbox("📷 Kameranı başlat")

frame_window = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Kamera açılmadı")
            break

        results = model(frame)

        detected = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                detected.append(name)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        frame_window.image(frame, channels="BGR")

        # SƏS
        if detected:
            obj = detected[0]

            if obj != st.session_state.last or time.time() - st.session_state.time > 5:
                speak(f"Qarşında {obj} var")
                st.session_state.last = obj
                st.session_state.time = time.time()

    cap.release()
