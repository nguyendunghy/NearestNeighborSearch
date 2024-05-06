import faiss


class FaissIndex:
    def __init__(self, vectors, dim, metric):
        self._dim = dim

        quantizer = self.get_quantizer(metric)(dim)
        self._flat_index = self.build_index(quantizer, vectors, dim)

    def get_quantizer(self, metric):
        return faiss.IndexFlatL2 if metric == 'l2' else faiss.IndexFlatIP

    def build_index(self, quantizer, vectors, dim):
        # index = quantizer
        index = faiss.IndexIVFFlat(quantizer, dim, 128)
        index.nprobe = 8
        assert not index.is_trained
        index.train(vectors)
        assert index.is_trained
        index.add(vectors)
        return index

    def query(self, query, k=1):
        distances, indexes = self._flat_index.search(query, k=k)
        return distances
