from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModel.from_pretrained("facebook/opt-350m")
generator = pipeline('text-generation', model="facebook/opt-350m")
messages = []

@app.route("/delete-all", methods=["GET"])
def delete_all_messages():
    global messages
    messages = []
    return jsonify({"messages": messages})

@app.route("/", methods=["GET", "POST"])
def handle_message():

    if request.method == "POST":
        # Get the user message from the form
        user_message = request.form.get("chat-input")
        # extract the message from the request
        response = generator(user_message)[0]["generated_text"]
        # Add the messages to the list
        messages.append({"sender": "user", "text": user_message})
        messages.append({"sender": "bot", "text": response})
        # render a template that displays the response
        return render_template("index.html", messages=messages)
    return render_template("index.html", messages=messages)

if __name__ == "__main__":
    app.run()
