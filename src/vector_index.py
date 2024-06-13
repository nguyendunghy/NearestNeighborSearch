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
    index = faiss.index_factory(dim, "IVF262144,PQ64", faiss.METRIC_L2)  # 65536   262144

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
    assert index.is_trained
    index.add(vectors)

    faiss.write_index(index, str(index_path))


def build_ivf(vectors_dir, dim, indexes_dir, build_with_gpu):
    """
    load npy files and build index
    """
    npy_filenames = list(sorted(list(Path(vectors_dir).glob('*.npy'))))

    if not Path("trained_block.index").exists():
        build_trained_vector_index(npy_filenames[0], dim, build_with_gpu, indexes_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for npy_filename in tqdm.tqdm(sorted(npy_filenames)):
            index_path = indexes_dir / f"{npy_filename.stem}.index"
            future = executor.submit(process_block, npy_filename, index_path)
            futures.append(future)

        # Wait for all futures to complete
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                future.result()  # this will raise an exception if the function raised
            except Exception as e:
                print(f"Error processing file {npy_filename}: {e}")
