import concurrent.futures
from pathlib import Path

import faiss
import tqdm
import numpy as np


def load_data(npy_filename):
    """
    load npy file and return vectors. Preprocess vector by type of metric
    """
    vectors = np.load(npy_filename, mmap_mode='c')
    return vectors


def build_trained_vector_index(npy_filename, dim, build_with_gpu, indexes_dir):
    vectors = load_data(npy_filename)

    # index is needed in a lot of count of clusters if dataset is big.
    index = faiss.index_factory(dim, "IVF1048576,PQ64", faiss.METRIC_L2)  # 65536   262144   1048576

    index_ivf = faiss.extract_index_ivf(index)

    if build_with_gpu:
        index_flat = faiss.IndexFlatL2(dim)

        res = faiss.StandardGpuResources()
        clustering_index = faiss.index_cpu_to_gpu(res, 0, index_flat)  # 0 – № GPU
        index_ivf.clustering_index = clustering_index

    print(f'Train index ... with gpu={build_with_gpu}')
    train_vectors = vectors
    assert not index.is_trained
    index.train(train_vectors)

    indexes_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, "trained_block.index")


def process_block(npy_filename, index_path):
    vectors = load_data(npy_filename)

    index = faiss.read_index("trained_block.index")

    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    index = faiss.index_cpu_to_gpu(res, 0, index, co)

    assert index.is_trained
    index.add(vectors)

    index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, str(index_path))


def build_ivf(vectors_dir, dim, indexes_dir, build_with_gpu):
    """
    load npy files and build index
    """
    npy_filenames = list(sorted(list(Path(vectors_dir).glob('*.npy'))))

    if not Path("trained_block.index").exists():
        build_trained_vector_index(npy_filenames[0], dim, build_with_gpu, indexes_dir)

    for npy_filename in tqdm.tqdm(sorted(npy_filenames)):
        index_path = indexes_dir / f"{npy_filename.stem}.index"
        process_block(npy_filename, index_path)
