from flask import Flask
from flask import jsonify
import json

app = Flask(__name__)

@app.route("/")
def home_view():
        return jsonify("Hello world")
