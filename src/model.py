from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class Model:
    def __init__(self, model_name_or_path: str) -> None:
        self._model = SentenceTransformer(
            model_name_or_path,
            trust_remote_code=True,
            device='cpu'
        )

    def predict(self, sentences: List[str]):
        query_vectors = self._model.encode(sentences)
        return query_vectors


if __name__ == '__main__':
    model = Model()
    sentences = ['I love cats', 'I love dogs', 'I hate dogs']
    vectors = model.predict(sentences)
    np.save('../data/1.npy', vectors)
