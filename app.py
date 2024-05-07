import time
from flask import Flask, request, jsonify

from src.model import Model
from src.nearest_neighbor import NearestNeighbor

app = Flask(__name__)
model = Model()
nn = NearestNeighbor(vectors_dir='./data/', dim=384, metric='l2')


@app.route("/")
def hello_world():
    return "Hello! I am model service"


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time_ns()
    if request.is_json:
        data = request.get_json()
        input_data = data['list_text']
        query_vectors = model.predict(input_data)
        result = nn.find(query_vectors)
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        return jsonify({"message": "predict successfully", "result": result}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
