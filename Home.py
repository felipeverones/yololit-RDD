import streamlit as st


st.set_page_config(
    page_title="Detecção de Danos em Rodovias",
    page_icon="🛣️",
)

st.markdown(
    """
    <style>
    /* Seletor para a seção da barra lateral */
    section[data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Aplicativo de Detecção de Danos em Rodovias")
st.divider()

st.markdown(
    """
    Aplicativo de Detecção de Danos em Pavimento Asfáltico, operado pelos modelos de deep learning YOLOv8x e YOLOv9e, treinados no Conjunto de Dados RDDFilteredSamplev5, baseado no RDD2022, do Desafio de Detecção de Danos Rodoviários baseado em
    sensoriamento coletivo (do inglês Crowdsensing-based Road Damage Detection Challenge -
    CRDDC’2022).
    
    Este aplicativo foi projetado para demonstrar a capacidade de inferência dos modelos treinados para melhorar a segurança nas estradas e a manutenção da infraestrutura, identificando e categorizando rapidamente várias formas de danos nas rodovias, como buracos e rachaduras.

    Existem quatro tipos de danos que estes modelos podem detectar:
    - Rachadura Longitudinal (Logitudinal Crack)
    - Rachadura Transversal (Transverse Crack)
    - Rachadura Jacaré (Aligator Crack)
    - Buracos e Poças (Potholes)


    Selecione os aplicativos na barra lateral para experimentar e testar com qualquer tipo de entrada **(webcam em tempo real, vídeo, imagens e captura de tela em tempo real)**, dependendo do seu caso de uso.

    #### Documentação e Links
    - Página do Projeto no [Github](https://github.com/felipeverones/yololit_RDD)
    - Contato: felipe.verones@gmail.com

    #### Licença e Citações
    - Conjunto de Dados de Danos nas Estradas do Desafio de Detecção de Danos nas Estradas baseado em Crowdsensing (CRDDC2022)
    - Todos os direitos reservados sob a licença YOLOv8 fornecida por [Ultralytics](https://github.com/ultralytics/ultralytics) e ao [Streamlit](https://streamlit.io/)
    - Agradecimentos aos autores do [YOLOv9](https://arxiv.org/pdf/2402.13616) pelas suas contribuições significativas no domínio da deteção de objetos em tempo real:
    
    """
)


st.divider()

st.markdown(
    """
    Esse projeto foi desenvolvido como parte do Trabalho de Conclusão de Curso intitulado [Detecção de Danos em Rodovias por meio de Aprendizado Profundo](https://repositorio.ufsc.br/handle/123456789/255766).
    """
)
