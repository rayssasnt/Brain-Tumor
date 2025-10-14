# Classificador de Tumor Cerebral com PyTorch e Streamlit


Este projeto implementa uma Rede Neural Convolucional (CNN) usando o framework PyTorch para classificar imagens de ressonância magnética em quatro categorias distintas: **glioma, meningioma, pituitário** ou se **não há presença de tumor**.

O modelo foi treinado e avaliado em um ambiente Google Colab, utilizando aceleração por GPU para otimizar o tempo de treinamento.

## 📋 Visão Geral

- **Framework:** PyTorch
- **Dataset:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Arquitetura:** Rede Neural Convolucional (CNN)
- **Resultado:** Acurácia de **96.03%** no conjunto de teste.
-   **Interface Web Interativa:** Simples e intuitiva para o usuário final.
-   **Upload de Imagens:** Permite que o usuário envie suas próprias imagens de ressonância magnética (formatos `jpg`, `jpeg`, `png`).
-   **Classificação em Tempo Real:** O modelo processa a imagem e retorna o diagnóstico predito e o nível de confiança da predição.
-   **Imagens para Teste:** Oferece um conjunto de imagens de exemplo para que o usuário possa testar a aplicação sem precisar de uma imagem própria.


## 📸 Demonstração

[Insira aqui um GIF ou uma captura de tela da sua aplicação em funcionamento para torná-la mais atrativa!]

*(Exemplo de como a aplicação se parece)*
![Imagem da interface do classificador de tumor cerebral]

## 🛠️ Tecnologias Utilizadas

-   **Streamlit:** Para a criação da interface web.
-   **PyTorch:** Para carregar e executar o modelo de Deep Learning.
-   **Pillow (PIL):** Para pré-processamento e manipulação das imagens.
