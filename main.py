import time

import numpy as np

from vector_index import FaissIndex
from model import Model


def norm(data):
    return data / np.linalg.norm(data)


class TestFaissIndex:

    def __init__(self, dim=384):
        self._dim = dim
        self._dataset = np.load('data/1.npy')

        self._model = Model()

    def test(self, metric):
        dataset = self._dataset
        if metric == 'cosine':
            dataset = norm(self._dataset)

        index = FaissIndex(dataset, dim=self._dim, metric=metric)  # takes a lot of time

        while True:
            t1 = time.time()
            sentence = input("Enter a sentence: ")
            query_vectors = self._model([sentence])
            if metric == 'cosine':
                query_vectors = norm(query_vectors)
            distances = index.query(query_vectors)
            t2 = time.time()
            print(distances, f"{t2 - t1:.2f} ms")


if __name__ == '__main__':
    test_faiss_index = TestFaissIndex(dim=384)
    test_faiss_index.test(metric='l2')
    # test_faiss_index.test(metric='cosine')
