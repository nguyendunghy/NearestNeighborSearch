from pathlib import Path

import faiss
import tqdm
import numpy as np


def norm(data):
    return data / np.linalg.norm(data)


class FaissIndex:
    def __init__(self, vectors_dir, dim, metric):
        self._dim = dim
        self._metric = metric

        self._index = faiss.IndexShards(dim)

        if not Path('indexes').exists():
            for npy_filename in tqdm.tqdm(list(Path(vectors_dir).glob('*.npy'))):
                vectors = np.load(npy_filename, mmap_mode='c')
                if metric == 'cosine':
                    vectors = norm(vectors)
                quantizer = self.get_quantizer(metric)(dim)
                index = self.build_index(quantizer, vectors, dim)

                index_path = Path('indexes') / f"{npy_filename.stem}.index"
                index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(index, str(index_path))

                self._index.add_shard(index)
        self._index = None

        for index_filename in tqdm.tqdm(list(Path('indexes').glob('*.index'))):
            index = faiss.read_index(str(index_filename), faiss.IO_FLAG_MMAP)
            self._index.add_shard(index)

    def get_quantizer(self, metric):
        return faiss.IndexFlatL2 if metric == 'l2' else faiss.IndexFlatIP

    def build_index(self, quantizer, vectors, dim):
        # Number of centroid IDs
        num_cent_ids = 8
        nlist = 200

        # Number of bits in each centroid
        cent_bits = 8
        # index = faiss.IndexIVFPQ(quantizer, dim, nlist, num_cent_ids, cent_bits)

        index = faiss.IndexIVFFlat(quantizer, dim, 128)
        index.nprobe = 8
        assert not index.is_trained
        index.train(vectors)
        assert index.is_trained
        index.add(vectors)
        return index

    def query(self, query_vectors, k=1):
        if self._metric == 'cosine':
            query_vectors = norm(query_vectors)
        all_distances = []
        # for index in self._indexes:
        distances, indexes = self._index.search(query_vectors, k=k)
        # all_distances.append(distances)
        # all_distances = np.array(all_distances)
        return distances
