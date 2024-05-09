"""
Script generates stub random vectors and save it to npy files
"""
import tqdm
import numpy as np

if __name__ == '__main__':
    size = (100_000, 384)
    for i in tqdm.tqdm(range(10)):
        vectors = np.random.random(size=size).astype(np.float32)
        np.save(f'./data/{i}.npy', vectors)
