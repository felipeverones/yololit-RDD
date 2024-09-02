import os
import logging
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO

st.set_page_config(
    page_title="Detecção de Vídeo",
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

# Create temporary folder if doesn't exist
if not os.path.exists('./temp'):
    os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Grava o conteúdo do BytesIO fornecido em um arquivo.
    Cria o arquivo ou sobrescreve o arquivo se ele ainda
    não existir.
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    # Gravar o arquivo no disco
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)

    # Check the video
    if not videoCapture.isOpened():
        st.error('Erro ao abrir o arquivo de vídeo')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS)
        _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        _duration = _frame_count / _fps
        _duration_minutes = int(_duration / 60)
        _duration_seconds = int(_duration % 60)
        _duration_strings = f"{_duration_minutes}:{_duration_seconds}"

        st.write("Duração do Vídeo :", _duration_strings)
        st.write("Largura, Altura e FPS :", _width, _height, _fps)

        inferenceBarText = "Realizando inferência no vídeo, por favor, aguarde."
        inferenceBar = st.progress(0, text=inferenceBarText)

        imageLocation = st.empty()

        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        # Read until video is completed
        _frame_counter = 0
        while videoCapture.isOpened():
            ret, frame = videoCapture.read()
            if ret:
                # Convert color-chanel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform inference
                _image = np.array(frame)
                image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
                results = net.predict(image_resized, conf=score_threshold)

                # Save the results
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

                annotated_frame = results[0].plot()
                _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation=cv2.INTER_AREA)
                
                # Write the image to file
                _out_frame = cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR)
                cv2writer.write(_out_frame)
                
                # Display the image
                imageLocation.image(_image_pred)

                _frame_counter += 1
                inferenceBar.progress(_frame_counter / _frame_count, text=inferenceBarText)
            else:
                inferenceBar.empty()
                break

        # When everything done, release the video capture object
        videoCapture.release()
        cv2writer.release()

    # Download button for the video
    st.success("Vídeo Processado!")

    col1, col2 = st.columns(2)
    with col1:
        with open(temp_file_infer, "rb") as f:
            st.download_button(
                label="Baixar Vídeo de Previsão",
                data=f,
                file_name="RDD_Prediction.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
    with col2:
        if st.button('Restart Apps', use_container_width=True, type="primary"):
            st.experimental_rerun()

st.title("Detecção de Danos em Rodovias por Vídeo")
st.write("Detecte danos em rodovias usando entrada de vídeo. Faça o upload do vídeo e comece a detectar. Esta seção pode ser útil para examinar e processar vídeos gravados.")


video_file = st.file_uploader("Carregar Vídeo", type=".mp4", disabled=st.session_state.runningInference)
st.caption("Há um limite de 1GB para o tamanho do vídeo com extensão .mp4. Redimensione ou corte seu vídeo se for maior que 1GB.")

score_threshold = st.slider("Limite de Confiança", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=st.session_state.runningInference)
st.write("Diminua o limite se nenhum dano for detectado e aumente o limite se houver previsão falsa. Você pode alterar o limite antes de executar a inferência.")


if video_file is not None:
    if st.button('Processar Vídeo', use_container_width=True, disabled=st.session_state.runningInference, type="secondary", key="processing_button"):
        _warning = "Processando Vídeo " + video_file.name
        st.warning(_warning)
        processVideo(video_file, score_threshold)
