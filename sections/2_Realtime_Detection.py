import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Deep learning framework
from ultralytics import YOLO

st.set_page_config(
    page_title="Detecção em Tempo Real",
    page_icon="📷",
    layout="centered",  
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)



# Seleção do modelo
model_choice = st.sidebar.selectbox("Escolha o modelo YOLO", ["YOLOv8x", "YOLOv9e"], index=0)
MODEL_LOCAL_PATH = ROOT / f"./models/{model_choice.lower()}/best.pt"

# Servidor STUN fixo
STUN_SERVER = [{"urls": ["stun:stun.l.google.com:19302"]}]

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
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Detecção de Danos em Rodovias em Tempo Real")

st.write("Detecte danos em rodovias em tempo real usando webcam USB. Isso pode ser útil para monitoramento no local com pessoal em campo. Selecione o dispositivo de entrada de vídeo e inicie a inferência.")

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h_ori = image.shape[0]
    w_ori = image.shape[1]
    image_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)

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

    return av.VideoFrame.from_ndarray(_image, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "min": 800},
        },
        "audio": False
    },
    async_processing=True,
)

score_threshold = st.slider("Limite de Confiança", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.write("Diminua o limite se nenhum dano for detectado e aumente o limite se houver previsão falsa.")


st.divider()

if st.checkbox("Mostrar Tabela de Previsões", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
