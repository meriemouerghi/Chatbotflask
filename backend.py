import joblib
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)
# Load the RNN modelpip show keras
genai.configure(api_key='AIzaSyAsdJzG-XslY5w7i7aHWijyd3iLa4ElgPg')
model = load_model('model_name.h5')
# Load the scaler
#scaler = joblib.load('path_to_your_scaler.pkl')
@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')
@app.route('/model', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    # Preprocess the input data
    input_array = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
   
    input_reshaped = input_array.reshape((1, 1, input_array.shape[1]))
    # Make prediction
    prediction = model.predict(input_reshaped)
    predicted_class = np.argmax(prediction, axis=1)
    return render_template('index.html', prediction=f'Predicted class: {predicted_class[0]}')

@app.route('/ask', methods=['POST'])
def ask_question():
    print('hellooooo', request)
    data = request.get_json()
    question = data['question']
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(question)
    if response is not None:
        print(response.text)
        return (jsonify({"response": response.text})), 200
    else:
        return jsonify({"error": "No response from model"}), 404
    # Utilisez la fonction `chat` pour interroger le modèle avec la question
 

if __name__ == '__main__':
    app.run(port=3000, debug=True)