# Instruction

## №1 Server requirements
 - RAM 64 GB. 
 - GPU with 8GB or more. If you don't have this, set flag server=cpu
 - HDD 2TB or more for store data.

## №2 Requirements

### 2.1 Install python3.10 and pip

```shell
sudo apt-get update -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update -y
sudo apt install python3.10 python3.10-venv python3.10-dev -y
```

```shell 
curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
python3 get-pip.py
```

### 2.2 Install requrements.txt

`python3 -m pip install -r requirements.txt`

## №3 Running

To run the application, execute the following command:

`python3 app.py --npy-dir="./data/npy/" --metric="l2"`

Command Line Arguments

`--npy-dir`: Path to the directory containing .npy files. The application will analyze data from this directory.

`--metric`: The metric used for data analysis. Allowed values: "l2", "cosine"

`--model_name_or_path` : The name of the model to use encoding text to vector

`--server`: Choose a backend that construct index. Gpu is fast in some times. Choices ['cpu','gpu']. 

# №4 Errors
if you got an error like this: 
```
Error in void faiss::OnDiskInvertedLists::do_mmap() at /project/faiss/faiss/invlists/OnDiskInvertedLists.cpp:287: Error: 'ptro != MAP_FAILED' failed: could not mmap merged_index.ivfdata: Invalid argument```

just to try run script one more time.