# Bachelor thesis 2021

## Run training pipeline

If on Ubuntu just run the included script:

```[bash]
./install_ubuntu.sh
```

If one a different OS, make sure you have CUDA and CUDNN installed. To install the Python dependencies run:

```[bash]
pip3 install numpy
pip3 install Cython
pip3 install -r requirements.txt
```

To exectue the training pipeline with hyperparameters defined in the file `test.json` (while in root of the project):

```[bash]
python3 train.py --args test.json
```
