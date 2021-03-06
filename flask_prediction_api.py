#!/usr/bin/env python3

# Módulos para o modelo e predição
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.transform import resize
from keras.models import model_from_json
import cv2

# Módulos para api com flask
import pydload
import uuid
import json
import time
import requests
import os

from flask import Flask, flash, get_flashed_messages, request, redirect, url_for, render_template_string, Response, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS, cross_origin

PROJECT_PATH = os.curdir # os.path.join(os.curdir,'drive','My Drive','TCC') para Google Colab


# Configurando o servidor
app = Flask(__name__)
app.secret_key = b'\x19Q\xb0h0,N\xb9\x0c\xad\\\\\xde\x9d\xaf\xc5' # python -c 'import os; print(os.urandom(16))'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = os.curdir
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#app.debug = True

#app.config['CORS_HEADER'] = 'Content-Type'
cors = CORS(app, resources={r'/*': {"origins": '*'}})


# Funções para carregar o modelo com pesos e fazer a predição
def load_model_and_weights(path=PROJECT_PATH, model_json_name='/model.json', weight_name='xray_class'):
    json_path = "{}{}".format(path, model_json_name)
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    weight_path=os.path.join("{}".format(path),"{}_weights.best.hdf5".format(weight_name))
    
    weight_exists = os.path.exists(weight_path)
    if not weight_exists:
        print('Beginning download of model with requests')

        url = 'https://github.com/asperino45/pneumonia-prediction-api/releases/download/v1.0/xray_class_weights.best.hdf5'
        r = requests.get(url)

        with open(weight_path, 'wb') as f:
            f.write(r.content)

    # Load the model weights
    model.load_weights(weight_path)
    model._make_predict_function()
    #model.summary()
    return model

def make_prediction(img):
    img = cv2.resize(img, (224,224))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.

    pred = model.predict(np.array([img]), batch_size=32, verbose=True)
    return {
        'classification': 'Pneumonia' if pred[0][1] >= 0.85 else 'Normal',
        'prediction': 'Pneumonia: {:.09f}, Normal: {:.09f}'.format(pred[0][1], pred[0][0])
        }


# Tratamento de erros
class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['status_code'] = self.status_code
        rv['mensagem'] = self.message
        return rv

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

# Caminhos da Api

@cross_origin()
@app.route('/predict', methods=['GET', 'POST'])
def classifier_from_url():
    if request.method == 'GET':
        url = request.args.get('url')
    elif request.method == 'POST':
        url = request.json.get('url')

    if url is None:
        raise InvalidUsage('Não existe campo url')

    try:
        path = str(uuid.uuid4())
        dload_status = pydload.dload(url, path, timeout=2 ,max_time=3)

        if not dload_status:
            os.remove(path)
            raise InvalidUsage('Arquivo é grande demais', status_code=413)

        img = cv2.imread(str(path))
        os.remove(path)
        res = make_prediction(img)
        return jsonify(res)
    except Exception as ex:
        print(ex)
        raise InvalidUsage(str(ex), status_code=500)

# Função para filtrar extensão do arquivo
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@cross_origin()
@app.route('/predict/upload', methods=['POST'])
def classifier_from_upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            raise InvalidUsage('Não existe parte "file" na requisição', status_code=400)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            raise InvalidUsage('Nenhum arquivo selecionado ou sem nome', status_code=400)
        if file and not allowed_file(file.filename):
            raise InvalidUsage('Extensão não permitida', status_code=400)

        try:
            # https://stackoverflow.com/questions/17170752/python-opencv-load-image-from-byte-string
            nparr = np.fromstring(file.read(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
            res = make_prediction(img)
            return jsonify(res)
        except Exception as ex:
            print(ex)
            raise InvalidUsage(str(ex), status_code=500)


@app.route('/', methods=['GET'])
def home():
    if request.method == 'GET':
        return render_template_string(
            '''
            <!doctype html>
<title>API de Predição com Flask</title>
<h2>Selecione um arquivo para subir e classificar</h2>
<form method="post" action="{{ url_for('classifier_from_upload') }}" enctype="multipart/form-data">
    <dl>
		<p>
			<input type="file" name="file">
		</p>
    </dl>
    <p>
		  <input type="submit" value="Submit">
  	</p>
</form>
</br>

<h2>Entre com uma URL para classificar</h2>
<form method="get" action="{{ url_for('classifier_from_url') }}">
    <dl>
    <p>
      <input type="url" name="url" value="https://prod-images-static.radiopaedia.org/images/1436778/7c47ea1fb9a8510e765ef510d36012_jumbo.jpeg">
    </p>
    </dl>
    <p>
      <input type="submit" value="Submit">
    </p>
</form>
            '''
        )


if __name__=='__main__':
    # https://stackoverflow.com/questions/59741453/is-there-a-general-way-to-run-web-applications-on-google-colab
    # from google.colab.output import eval_js
    # print(eval_js("google.colab.kernel.proxyPort(6788)"))

    model = load_model_and_weights()
    old_run = app.run
    is_server_running = True
    try:
        run_with_ngrok(app)  # Start ngrok when app is run
        app.run()
    except Exception as ex:
        is_server_running = False
        print(str(ex))
    
    if not is_server_running:
        try:
            old_run(debug=False, threaded=False, use_reloader=False)
        except Exception as ex:
            print(str(ex))





