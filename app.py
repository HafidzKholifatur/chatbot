from flask import Flask, request, jsonify
import random
import torch
import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

# Device configuration
device = torch.device('cpu')

# Load intents file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Function to get response from the model
def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "Aing gak ngerti ngab..."

# Route for regular chat messages
@app.route('/pesan', methods=['POST'])
def regular():
    incoming_msg = request.json['pesan']
    response = get_response(incoming_msg)
    return jsonify({"response": response})

# Route for Twilio webhook
@app.route('/twilio', methods=['POST'])
def twilio():
    incoming_msg = request.values.get('Body', '')
    response = get_response(incoming_msg)
    resp = MessagingResponse()
    resp.message(response)
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
