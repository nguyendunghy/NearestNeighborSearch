# Instruction

## №1 Increase swap space up to 256 GB

https://askubuntu.com/questions/178712/how-to-increase-swap-space

## №2 Requirements

### 2.1 Install pip

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

##