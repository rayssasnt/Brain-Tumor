import torch 
from torchvision import transforms
import streamlit as st
from PIL import Image
import torch.nn as nn

## carregando o modelo
def carregar_modelo ():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU() , nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU() , nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU() , nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 *16 *16, 256) , nn.ReLU() , nn.Dropout(0.5),
        nn.Linear(256,4))
 
    model.load_state_dict(torch.load('modelo_brain-tumor.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

#convertendo as imgs 
def converter_img(user_img):
    img = Image.open(user_img).convert('L').convert('RGB') 

    tf =transforms.Compose([
        transforms.Resize((128,128)),# Redimensiona a imagem para  128x128 pixels
        transforms.ToTensor(), #Converte a imagem para um Tensor do PyTorch
        transforms.Normalize([0.5 ,0.5 ,0.5],[0.5 ,0.5 ,0.5]) #cores (cinza)
    ])
    return tf(img).unsqueeze(0)


############ INTERFACE ############

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered", 
) #page config

### links contato 
def contato():
    linkedin = """
    <a href="https://www.linkedin.com/in/rayssasnt/" target="_blank" style="text-decoration:none;">
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:#0A66C2;color:white;padding:6px 12px;
                    border-radius:12px;font-weight:600;">
            <span class="material-symbols-outlined">Rayssa Santos</span> LinkedIn
        </div>
    </a>
    """

    email= """
    <a href="mailto:rayssasantos3025@gmail.com" style="text-decoration:none;">
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:#34A853;color:white;padding:6px 12px;
                    border-radius:12px;font-weight:600;">
            <span class="material-symbols-outlined">Contato</span> E-mail
        </div>
    </a>
    """

    links = st.markdown(linkedin + "&nbsp;&nbsp;" + email , unsafe_allow_html=True)
    return links

## Config menu
from streamlit_option_menu import option_menu

selected = option_menu(None, ["Home", "Imagem para teste"], 
    icons=['house', 'cloud-upload'], 
    menu_icon="cast", default_index=0, orientation="horizontal")


if selected == "Home":

    # title
    st.title("🧠⚕️ Classificador de Tumores Cerebrais",)

    #Aviso
    st.warning("""⚠️ **Esse projeto é apenas um exercício de Deep Learning!! Não deve ser usado para diagnósticos reais**
            
    """)

    st.info("""
    Esse modelo foi treinado com 3 tipos de tumores : **Glioma , Meningioma , Pituitary**.
            
    Para testar você pode enviar uma imagem de ressonância magnética para que o modelo a classifique.
    """)

    st.divider()

    modelo = carregar_modelo()

    NAME_CLASS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    uploaded_file = st.file_uploader("**Envie uma imagem para análise**", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        
        if st.button("🔎  Classificar Imagem"):

            st.image(uploaded_file, caption='Imagem Enviada.',use_container_width=False, width=400)

            with st.spinner("Analisando...."):
            
                try:
                    image_tensor = converter_img(uploaded_file)

                    with torch.no_grad():
                        output = modelo(image_tensor)

                        probabilidade = torch.nn.functional.softmax(output , dim =1)

                        maior_probabilidade , top_id = torch.max(probabilidade ,1)

                        indice_predito = top_id.item() 
                        acuracia = maior_probabilidade.item() * 100
                        result = NAME_CLASS[indice_predito]

                        st.success(f"Diagnóstico Predito : {result}")
                        st.metric(label="**Nível de Acurácia**", value=f"{acuracia:.2f}%")

                except Exception as e:
                    st.error(f"Erro : {e}")
                    
                st.markdown("<br><br>", unsafe_allow_html=True)
                contato()
else:
   
    st.header("Não tem uma imagem? Teste com um dos nossos exemplos:")

    st.markdown("Imagens de exemplo:")

    col1, col2 = st.columns(2 ,gap="medium")
    col3 , col4 = st.columns(2,gap="medium")

    img1 = "imgs/img1.jpg"
    img2 = "imgs/imgM.jpg"
    img3 = "imgs/img3.jpg"
    img4 = "imgs/img4.jpg"

    with col1:
        st.image(img1, caption="Exemplo Glioma")
        
        with open("imgs/img1.jpg", "rb") as file:
            st.download_button(
                label="Download Glioma",
                data=file,
                file_name="Tumor Glioma.png",
                mime="image/png",
                
                )
    with col2:
        st.image(img2, caption="Exemplo Meningioma")
        with open("imgs/imgM.jpg", "rb") as dowloand:
            st.download_button(
                label="Download Meningioma",
                data = dowloand,
                file_name="Tumor Meningioma.png",
                mime="image/png",
                )

    with col3:
        st.image(img3, caption="Exemplo Pituitary")
        with open("imgs/img3.jpg", "rb") as download:
            st.download_button(
                label="Download Pituitary",
                data=download,
                file_name="Tumor Pituitary.png",
                mime="image/png",
              )

    with col4:
        st.image(img4, caption="Exemplo SEM Tumor")
        with open("imgs/img4.jpg", "rb") as file:

            st.download_button(
                label="Download image",
                data=file,
                file_name="SEM Tumor.png",
                mime="image/png",
                )
            
    st.markdown("<br><br>", unsafe_allow_html=True)

    #### links para contato
    contato()

   
       
       


      
    

   


    
            

    
    

    
        

