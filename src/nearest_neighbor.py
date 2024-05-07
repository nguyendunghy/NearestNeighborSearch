from pathlib import Path

import numpy as np

from src.vector_index import FaissIndex


def norm(data):
    return data / np.linalg.norm(data)


class NearestNeighbor:

    def __init__(self, vectors_dir, metric, dim=384):
        self._dim = dim
        self._metric = metric

        dataset = []
        for npy_filename in Path(vectors_dir).glob('*.npy'):
            data = np.load(npy_filename)
            dataset.append(data)
        dataset = np.vstack(dataset)
        if metric == 'cosine':
            dataset = norm(dataset)
        self._index = FaissIndex(dataset, dim=self._dim, metric=metric)  # takes a lot of time

    def find(self, query_vectors):
        if self._metric == 'cosine':
            query_vectors = norm(query_vectors)
        distances = self._index.query(query_vectors)
        distances = np.squeeze(distances, axis=1)
        distances = distances.tolist()
        return distances


if __name__ == '__main__':
    test_faiss_index = TestFaissIndex(dim=384)
    test_faiss_index.test(metric='l2')
    # test_faiss_index.test(metric='cosine')
