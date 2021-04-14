from flask import Flask, jsonify, request
import json

test = ['U fInNa CrAsH', 'U good']
app = Flask(__name__)


@app.route("/", methods=['POST'])

def home_view():
        data = request.json
        return jsonify(data)
