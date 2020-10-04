# broadcast model
This project is used to fix position for broadcast source.A radio wave propagation model class and a broadcast pisition location model class are built in this project.

## Requirement

- numpy
- matplotlib

## Installation

```
pip install numpy matplotlib
```

## A  quick demo

```
#assume that you are under the root directory of this project
python -m broadcast_model.py
```

## Use

You can use you own monitor data or use ```main_test.py```to generate data,note that you own data should be a  (N,4) numpy array with format:[latitude,longtitude,frequency,power]*N,N is the number of you data.