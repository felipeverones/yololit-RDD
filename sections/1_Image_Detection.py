import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Detecção de Imagens",
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

# Cache da sessão específica para carregar o modelo
@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

# Carregar o modelo selecionado
if "model_choice" not in st.session_state or st.session_state.model_choice != model_choice:
    st.session_state.model_choice = model_choice
    st.session_state.net = load_model(MODEL_LOCAL_PATH)

net = st.session_state.net

# Carregar as classes do modelo (se disponível)
CLASSES = net.names if hasattr(net, 'names') else [
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

st.title("Detecção de Danos em Rodovias por Imagem")
st.write("Detecte danos em rodovias usando uma entrada de imagem. Faça o upload da imagem e comece a detectar. Esta seção pode ser útil para examinar dados de base.")

image_file = st.file_uploader("Carregar Imagem", type=['png', 'jpg'])

score_threshold = st.slider("Limite de Confiança", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Diminua o limite se nenhum dano for detectado e aumente o limite se houver previsão falsa.")


if image_file is not None:
    # Load the image
    image = Image.open(image_file)
    
    col1, col2 = st.columns(2)

    # converte para array np 
    _image = np.array(image)
    #pega as dimensões da imagem
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]

    # redimensiona para 640x640, padrão yolo
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA) 
    # faz a inferência
    results = net.predict(image_resized, conf=score_threshold)
    
    # Save the results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls.item()),  # Extrair elementos únicos antes da conversão
                label=CLASSES[int(_box.cls.item())],  
                score=float(_box.conf.item()),  
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    # Original Image
    with col1:
        st.write("#### Imagem")
        st.image(_image)
    
    # Predicted Image
    with col2:
        st.write("#### Previsões")
        st.image(_image_pred)

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        st.download_button(
            label="Baixar Imagem de Previsão",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )
