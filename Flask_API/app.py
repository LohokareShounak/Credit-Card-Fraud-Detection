from flask import Flask, request
import json, pickle
import numpy as np
app = Flask(__name__)

def load_model():
    file_name = "model_XGBC.p"
    with open(file_name, "rb") as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model

@app.route("/predict", methods = ['GET'])

def predict():
   # parse input features from request
    request_json = request.get_json()
    #print(request_json)
    data = request_json['input']
    #print(data)
    print(np.shape(np.array(data).reshape(1,-1)))
    print(np.shape(np.array(data)))
    # load model
    model = load_model()
    prediction = model.predict(np.array(data)).tolist()
    print(prediction)
    response = json.dumps({'response': json.dumps(prediction)})
    return response, 200
if __name__ == '__main__':
    app.run(debug=True)