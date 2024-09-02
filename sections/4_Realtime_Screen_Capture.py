import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
import time

import cv2
import numpy as np
import streamlit as st
import mss

# Deep learning framework
from ultralytics import YOLO

st.set_page_config(
    page_title="Detecção por Captura de Tela",
    page_icon="🖥️",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Seleção do modelo
model_choice = st.sidebar.selectbox("Escolha o modelo YOLO", ["YOLOv8x", "YOLOv9e"], index=0)
MODEL_LOCAL_PATH = ROOT / f"./models/{model_choice.lower()}/best.pt"

# Cache da sessão específica para carregar o modelo
@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

# Carregar o modelo selecionado
if "model_choice" not in st.session_state or st.session_state.model_choice != model_choice:
    st.session_state.model_choice = model_choice
    st.session_state.net = load_model(MODEL_LOCAL_PATH)

net = st.session_state.net

CLASSES = [
    "Rachadura Longitudinal",
    "Rachadura Transversal",
    "Rachadura em Aligator",
    "Buracos"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Detecção de Danos em Rodovias - Captura de Tela")

st.write("Detecte danos em rodovias em tempo real usando captura de tela do seu PC. A captura de tela será realizada em intervalos regulares e processada pelo modelo YOLO.")

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# Placeholder para a imagem
image_placeholder = st.empty()

# Variável de controle para parar a captura
if "stop_capture" not in st.session_state:
    st.session_state.stop_capture = False

def process_frame(image):
    h_ori = image.shape[0]
    w_ori = image.shape[1]
    image_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)

    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]
        result_queue.put(detections)

    annotated_frame = results[0].plot()
    _image = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    return _image, detections

def capture_and_infer():
    interval = 2.0

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while not st.session_state.stop_capture:
            start_time = time.time()
            region = {
                "top": monitor["top"] + 100,
                "left": monitor["left"] + 100,
                "width": 800,
                "height": 600,
            }
            screenshot = sct.grab(region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            _image, detections = process_frame(frame)
            image_placeholder.image(_image, channels="BGR")
            
            elapsed_time = time.time() - start_time
            time.sleep(max(0, interval - elapsed_time))

def start_capture():
    st.session_state.stop_capture = False
    capture_and_infer()

def stop_capture():
    st.session_state.stop_capture = True

score_threshold = st.slider("Limite de Confiança", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.write("Diminua o limite se nenhum dano for detectado e aumente o limite se houver previsão falsa.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Iniciar Captura de Tela"):
        start_capture()
with col2:
    if st.button("Parar Captura de Tela"):
        stop_capture()
