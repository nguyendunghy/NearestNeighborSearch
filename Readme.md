# Instruction

## №1 Server requirements

- RAM 64 GB.
- GPU with 8GB or more. If you don't have this, set flag server=cpu
- HDD 2TB or more for store data.
- CPU:
-     AMD EPYC 7763 64-Core Processor for 13 second per request.
-     Intel® Xeon® Gold 6438M for 5-7 second per request.

## №2 Requirements

### Install Anaconda

 ```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh 
 ```

create env

```shell
source ~/.bashrc # initialize conda
conda create --name env1 python=3.10 # create new env
conda activate env1 # activate existing env
```

install requirements

```shell
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install flask
conda install -c conda-forge sentence-transformers
```

Run server

```shell
conda run --no-capture-output python app.py --npy-dir ../data --server gpu
```

## №3 Running

To run the application, execute the following command:

`conda run python app.py --npy-dir="./data/npy/" --metric="l2"`

Command Line Arguments

`--npy-dir`: Path to the directory containing .npy files. The application will analyze data from this directory.

`--metric`: The metric used for data analysis. Allowed values: "l2", "cosine"

`--model_name_or_path` : The name of the model to use encoding text to vector

`--server`: Choose a backend that construct index. Gpu is fast in some times. Choices ['cpu','gpu'].

# №4 Errors

if you got an error like this:

```
Error in void faiss::OnDiskInvertedLists::do_mmap() at /project/faiss/faiss/invlists/OnDiskInvertedLists.cpp:287: Error: 'ptro != MAP_FAILED' failed: could not mmap merged_index.ivfdata: Invalid argument```
```

just to try run script one more time.
