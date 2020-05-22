# Pneumonia Prediction API

## Como executar e servir a API
1. Clonar o repositório
2. Executar o notebook `flask_prediction_api.ipynb`, que irá baixar os pesos treinados desse repositório e servir a API pelo ngrok

## Como treinar seu modelo
1. Clonar o repositório
2. Criar uma chave da API do Kaggle
3. Substituir as chaves e os caminhos onde for apropriado no notebook `pneumonia_prediction_depthwise.ipynb`
4. Executar o notebook `pneumonia_prediction_depthwise.ipynb`, que irá baixar o conjunto de dado e o VGG-16 do Kaggle e treinar o modelo, o serializando juntamente com seus pesos
5. Comentar célula com o download do modelo desse repositório no notebook `flask_prediction_api.ipynb`
6. Executar o notebook `flask_prediction_api.ipynb` com o servidor para verificar se os artefatos estão sendo gerados e carregados corretamente

## Como treinar o modelo executar o servidor separadamente de Jupyter Notebook

1. Clonar o repositório
2. Baixar todos os artefatos do notebook na linha de comando
   1. Datasets do Kaggle
   2. Dependências do Python
3. Converter o notebook do servidor para um arquivo Python
4. Para utilizar sua própria infraestrutura com e.g. Nginx
   1. É necessário alterar as dependências do servidor, removendo o módulo flask-ngrok da importação e das instalações
   2. Alterar a execução do servidor, substituindo sua execução utilizando o flask-ngrok para o app.run() padrão do Flask
   3. Configurar um servidor para abrir a porta do servidor para a internet
