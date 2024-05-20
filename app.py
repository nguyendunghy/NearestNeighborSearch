import time
from argparse import ArgumentParser

from flask import Flask, request, jsonify

from src.model import Model
from src.nearest_neighbor import NearestNeighbor


def parse():
    parser = ArgumentParser()
    parser.add_argument('--npy-dir', default='./data/', help='Path to the directory containing .npy files.')
    parser.add_argument('--metric', default='l2', help='The metric used for data analysis.')
    parser.add_argument('--model_name_or_path', type=str, default='sentence-transformers/all-MiniLM-L12-v2')
    parser.add_argument('--server', type=str, default='gpu', choices=['gpu', 'cpu'], required=True,
                        help='Choose a backend that construct index.')
    return parser.parse_args()


args = parse()

app = Flask(__name__)
model = Model(model_name_or_path=args.model_name_or_path)
nn = NearestNeighbor(vectors_dir=args.npy_dir, dim=384, metric=args.metric, build_with_gpu=args.server == 'gpu')


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
    app.run(host='0.0.0.0', debug=False, port=8080)
