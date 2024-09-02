# Aplicativo de Detecção de Danos em Rodovias

Este aplicativo utiliza modelos de deep learning YOLOv8x e YOLOv9e para detectar danos em pavimentos asfálticos. Ele foi desenvolvido como parte de um Trabalho de Conclusão de Curso e oferece várias funcionalidades, incluindo detecção em tempo real, detecção em imagens, vídeos e captura de tela.

<p align="center">
  <a href="https://youtu.be/DTf12ZWyYDk">
    <img src="https://img.youtube.com/vi/DTf12ZWyYDk/0.jpg" alt="Vídeo de Demonstração" width="600">
  </a>
</p>

<p align="center">
*Clique na imagem acima para assistir ao vídeo de demonstração do aplicativo*
</p>


## Pré-requisitos

- Python 3.11 ou superior
- pip (gerenciador de pacotes do Python)
- Git (opcional, para clonar o repositório)

## Instalação

1. Clone o repositório (ou baixe o código-fonte):
   ```
   git clone https://github.com/felipeverones/yololit-RDD.git
   cd yololit-RDD
   ```

2. (Recomendado) Crie e ative um ambiente virtual:
   ```
   python -m venv venv
   source venv/bin/activate  # No Windows use: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

## Configuração

1. Certifique-se de que os modelos YOLOv8x e YOLOv9e estão na pasta `models` com a seguinte estrutura:
   ```
   models/
   ├── yolov8x/
   │   └── best.pt
   └── yolov9e/
       └── best.pt
   ```

2. Verifique se o arquivo `.streamlit/config.toml` está presente e configurado corretamente.

## Executando o aplicativo

Para iniciar o aplicativo, execute o seguinte comando no terminal:

```
streamlit run app.py
```


O aplicativo será iniciado e você poderá acessá-lo através do seu navegador no endereço indicado no terminal (geralmente `http://localhost:8501`).

## Uso

O aplicativo oferece quatro funcionalidades principais:

1. **Detecção de Imagens**: Permite o upload e análise de imagens.
2. **Detecção em Tempo Real**: Utiliza a webcam para detecção em tempo real.
3. **Detecção de Vídeo**: Processa vídeos uploadados.
4. **Captura de Tela em Tempo Real**: Realiza detecção em capturas de tela do seu computador.

Selecione a funcionalidade desejada na barra lateral e siga as instruções na tela.

## Problemas Conhecidos

- Algumas funcionalidades podem apresentar certas falhas. Este aplicativo está em constante desenvolvimento.
- O upload de vídeos está limitado a 1GB. Para vídeos maiores, considere redimensioná-los ou cortá-los antes do upload.




## Estrutura do Código

- `app.py`: Ponto de entrada do aplicativo.
- `Home.py`: Página inicial do aplicativo.
- `sections/`: Diretório contendo os módulos para cada seção do aplicativo.
- `models/`: Diretório contendo os modelos YOLO.


## Contato

Para mais informações, entre em contato através do email: felipe.verones@gmail.com
