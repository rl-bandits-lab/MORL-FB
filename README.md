# MORL-FB

## Discrete Environment


### Code Structure

```
discrete/
    ├── agent.py --- the training agent of discrete MORL-FB
    ├── base.py --- the structure of trajectory replay buffer
    ├── main.py --- main execution file for MORL-FB algorithms
    ├── module.py --- the model of MORL-FB
    ├── multi_step.py --- the structure of trajectory replay buffer
    ├── requirements.txt --- the require package for MORL-FB
    └── utils.py --- utility functions
```



First of all, go to discrete directory

```
cd discrete
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
python main.py --env_name "deep-sea-treasure-v0" --seed 10 --cuda_device 0 --project_name "MORL-FB"
```

## Continuous Environment

### Code Structure
```
continuous/
    ├── base.py --- the structure of trajectory replay buffer
    ├── main.py --- main execution file for MORL-FB algorithms
    ├── mo_agent.py --- the training agent of discrete MORL-FB
    ├── module.py --- the model of MORL-FB
    ├── multi_step.py --- the structure of trajectory replay buffer
    ├── requirements.txt --- the require package for MORL-FB
    └── utils.py --- utility functions
```


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

## Evaluation Metrics Calculation

### Requirements
* Python version : tested in Python 3.9.16
* Operation Systems : Ubuntu 20.04
* pytorch version : 2.0.1

Install other required packages:

```
pip install -r requirements.txt
```
### Usage
* How to Run hv.py? 

```python
python hv.py --pref pref/mo-halfcheetah.npy --ref 0 -8000 --data rewards/FB/mo-halfcheetah.npy
```

Config:
* pref : preference set use on calculating hypervolumn(HV) and utility(UT).
* ref : reference point on calculating hypervolumn.
* data : testing rewards for calculating hypervolumn(HV) and utility(UT).

* How to Run ed.py? 

```python
python hv.py --pref pref/mo-halfcheetah.npy --base rewards/FB/mo-halfcheetah.npy --others rewards/Q-Pensieve/mo-halfcheetah.npy
```
Config:
* pref : preference set use on calculating Episodic Dominance(ED).
* base : rewards used as base on calculating Episodic Dominance(ED).
* others : rewards for calculating Episodic Dominance(ED).
