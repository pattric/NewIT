# coding: utf-8

import flask
import json
import numpy as np
from keras.models import Model
#import EAT_ML_RNN_LSTM1_layer as RNNLSTM
import EAT_keras_LSTM1_layer as RNNLSTM

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_keras_model():
    # load the pre-trained Keras model (here we are using a model pre-trained on ImageNet and provided by Keras, 
    # but you can substitute in your own networks just as easily)
    global model_instance
    model_instance = RNNLSTM.RNN_LSTM_Model()

@app.route('/')
def index():
   return 'EAT AI API'

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False, "Category": ""}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        #
        request_data = flask.request.get_json()        
        if request_data == None:
            request_data = json.loads(flask.request.data)
        #
        desc = request_data['description']
        category = model_instance.pred_category(desc)
        # indicate that the request was a success
        data["success"] = True
        data["Category"] = category

        # return the data dictionary as a JSON response
        return flask.jsonify(data)

# if this is the main thread of execution first load the model and then start the server
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server... " \
          "please wait until server has fully started"))
    load_keras_model()
    #
    app.run(port=8080, host="127.0.0.1")
