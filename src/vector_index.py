from pathlib import Path

import faiss
import tqdm
import numpy as np

indexes_dir = Path('indexes')


def norm(data):
    return data / np.linalg.norm(data)


def load_data(npy_filename, metric):
    """
    load npy file and return vectors. Preprocess vector by type of metric
    """
    vectors = np.load(npy_filename, mmap_mode='c')
    if metric == 'cosine':
        vectors = norm(vectors)
    return vectors


def build_ivf(vectors_dir, dim, metric):
    """
    load npy files and build index
    """
    npy_filenames = list(sorted(list(Path(vectors_dir).glob('*.npy'))))

    vectors = load_data(npy_filenames[0], metric)
    metric = faiss.METRIC_L2 if metric == 'l2' else faiss.METRIC_INNER_PRODUCT

    # index is needed in a lot of count of clusters if dataset is big.
    index = faiss.index_factory(dim, "IVF65536,Flat", metric)  # 65536

    index_ivf = faiss.extract_index_ivf(index)

    train_vectors = vectors
    assert not index.is_trained
    index.train(train_vectors)

    indexes_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, "trained_block.index")

    for npy_filename in tqdm.tqdm(sorted(npy_filenames)):
        vectors = load_data(npy_filename, metric)

        index = faiss.read_index("trained_block.index")
        assert index.is_trained
        index.add(vectors)

        index_path = indexes_dir / f"{npy_filename.stem}.index"
        faiss.write_index(index, str(index_path))


class FaissIndex:
    def __init__(self, vectors_dir, dim, metric):
        self._dim = dim
        self._metric = metric

        faiss.omp_set_num_threads(16)

        # 1. build index for every npy blocks
        ivf_filepaths = list(indexes_dir.glob('*.index'))
        if not Path('trained_block.index').exists():
            build_ivf(vectors_dir, dim, metric)

        # 2. merge all ivf indexes into one
        if not Path("populated.index").exists():
            ivfs = []
            for ivf_filepath in ivf_filepaths:
                index = faiss.read_index(str(ivf_filepath), faiss.IO_FLAG_MMAP)
                ivfs.append(index.invlists)
                index.own_invlists = False

            index = faiss.read_index("trained_block.index")
            invlists = faiss.OnDiskInvertedLists(index.nlist, index.code_size, "merged_index.ivfdata")
            ivf_vector = faiss.InvertedListsPtrVector()

            for ivf in ivfs:
                ivf_vector.push_back(ivf)

            ntotal = invlists.merge_from(ivf_vector.data(), ivf_vector.size())
            index.ntotal = ntotal  # заменяем листы индекса на объединенные
            index.replace_invlists(invlists)
            faiss.write_index(index, "populated.index")

        # 3. read ready index to memory.
        self._index = faiss.read_index('populated.index', faiss.IO_FLAG_ONDISK_SAME_DIR)

        # set how many probes to use. it increases quality
        self._index.nprobe = 256

    def query(self, query_vectors, k=1):
        if self._metric == 'cosine':
            query_vectors = norm(query_vectors)
        distances, indexes = self._index.search(query_vectors, k=k)
        return distances


if __name__ == '__main__':
    index = FaissIndex(vectors_dir='data', metric='l2', dim=384)
