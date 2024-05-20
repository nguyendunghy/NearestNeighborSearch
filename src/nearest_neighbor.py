import numpy as np

from src.vector_index import FaissIndex


class NearestNeighbor:

    def __init__(self, vectors_dir, metric, build_with_gpu=True, dim=384):
        self._dim = dim
        self._metric = metric

        self._index = FaissIndex(vectors_dir, dim=self._dim, metric=metric, build_with_gpu=build_with_gpu)  # takes a lot of time

    def find(self, query_vectors):
        distances = self._index.query(query_vectors)
        distances = np.squeeze(distances, axis=1)
        distances = distances.tolist()
        return distances


if __name__ == '__main__':
    test_faiss_index = NearestNeighbor(vectors_dir='data', metric='l2', dim=384)
    import time

    t1 = time.time()
    distances = test_faiss_index.find(np.random.rand(300, 384))
    t2 = time.time()
    print(distances)
    print(t2 - t1, 'sec')
