from pathlib import Path

import tqdm
import numpy as np

DIM = 384
TRAIN_SET = 25_000_000
BLOCK_SIZE = 500_000


def main(root='data'):
    Path(root).mkdir(parents=True, exist_ok=True)

    for i in tqdm.tqdm(range(TRAIN_SET // BLOCK_SIZE)):
        data = np.random.rand(BLOCK_SIZE, DIM).astype(np.float16)
        np.save(f'{root}/{i}.npy', data)


if __name__ == '__main__':
    main()
