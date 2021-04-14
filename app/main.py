from flask import Flask, jsonify, request
import json

test = ['U fInNa CrAsH', 'U good']
app = Flask(__name__)


@app.route("/", methods=['POST'])
def home_view():
        #JSON that is sent
        data = request.json
        if (data["ICY"] == True):
                return jsonify(test[0])
        else:
                return jsonify(test[1])
