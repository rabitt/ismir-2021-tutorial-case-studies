# ISMIR 2021 Tutorial: Programming MIR Baselines from Scratch - Pitch Tracking
Rachel Bittner

## To run this code:

### Setup

Install the requirements
```
pip install -r requirements.txt
```

Download the "medleydb_pitch" and "vocadito" datasets via [mirdata](https://mirdata.readthedocs.io/en/stable/)

```python
import mirdata

vocadito = mirdata.initialize("vocadito")
vocadito.download()
# >>>>>>>

medleydb_pitch = mirdata.initialize("medleydb_pitch")
medleydb_pitch.download()
# >> follow the printed instructions to request the data on Zenodo
```

### Generate the training data:

```
$ python generate_dataset.py <data_path>
```
for example
```
$ python generate_dataset.py data
```

This will take about 30 minutes to run. This could be sped up with parallelization!


### Run training:

```
$ python train.py <data_path> <model_path> <tensorboard_path>
```
for example
```
$ python train.py data pitch_salience.pt tensorboard
```

### Run evaluation:

```
$ python evaluate.py <model_path> <sonification_path>
```
for example
```
$ python evaluate.py pitch_salience.pt sonifications
```