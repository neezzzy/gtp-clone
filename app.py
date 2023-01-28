from flask import Flask, render_template
import torch
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350")
model = AutoModel.from_pretrained("facebook/opt-350")


@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run()
