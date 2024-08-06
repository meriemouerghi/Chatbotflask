from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configurez votre clé API

genai.configure(api_key='AIzaSyAsdJzG-XslY5w7i7aHWijyd3iLa4ElgPg')

@app.route('/')
def home():
    return "Bienvenue sur le Chatbot Biologie. Utilisez /ask pour poser des questions."

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
    app.run(debug=True)



