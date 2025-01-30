# MORL-FB

## Continuous Environment

First of all, go to continuous directory

```
cd continuous
```

### Requirements
* Python version : tested in Python 3.9.16
* Operation Systems : Ubuntu 20.04
* pytorch version : 2.0.1

Install other required packages:

```
pip install -r requirements.txt
```

### Usage
* How to Run ? 

```python
python main.py --env_name "mo-halfcheetah-v4" --seed 10 --cuda_device 0 --project_name "MORL-FB"
```