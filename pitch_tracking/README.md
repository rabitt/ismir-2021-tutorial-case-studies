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

## Related Reading

### Datasets Used

* Bittner, Rachel M., Justin Salamon, Mike Tierney, Matthias Mauch, Chris Cannam, and Juan Pablo Bello. ["Medleydb: A multitrack dataset for annotation-intensive mir research."](https://archives.ismir.net/ismir2014/paper/000322.pdf) International Society for Music Information Retrieval (ISMIR) Conference. 2014.

* Bittner, Rachel M., Katherine Pasalo, Juan José Bosch, Gabriel Meseguer-Brocal, and David Rubinstein. ["vocadito: A dataset of solo vocals with $ f_0 $, note, and lyric annotations."](https://arxiv.org/pdf/2110.05580) arXiv preprint arXiv:2110.05580 (2021).


### Neural networks for pitch tracking / F0 estimation

* Kim, Jong Wook, Justin Salamon, Peter Li, and Juan Pablo Bello. ["Crepe: A convolutional representation for pitch estimation."](https://arxiv.org/pdf/1802.06182) IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2018.

* Bittner, Rachel M., Brian McFee, Justin Salamon, Peter Li, and Juan Pablo Bello. ["Deep Salience Representations for F0 Estimation in Polyphonic Music."](https://archives.ismir.net/ismir2017/paper/000085.pdf) International Society for Music Information Retrieval (ISMIR) Conference. 2017.


### Pitch Salience

* Müller, Meinard. [Fundamentals of music processing: Audio, analysis, algorithms, applications.](https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP) Springer, 2015. pp. 445-450.

* Salamon, Justin, Emilia Gómez, Daniel PW Ellis, and Gaël Richard. ["Melody extraction from polyphonic music signals: Approaches, applications, and challenges."](https://repositori.upf.edu/bitstream/handle/10230/42183/Gomez_iee_melo.pdf?sequence=1&isAllowed=y) IEEE Signal Processing Magazine 31.2 (2014): 118-134.


### The Harmonic CQT

* Bittner, Rachel M., Brian McFee, Justin Salamon, Peter Li, and Juan Pablo Bello. ["Deep Salience Representations for F0 Estimation in Polyphonic Music."](https://archives.ismir.net/ismir2017/paper/000085.pdf) International Society for Music Information Retrieval (ISMIR) Conference. 2017.

* Bittner, Rachel M., Brian McFee, and Juan Pablo Bello. ["Multitask learning for fundamental frequency estimation in music."](https://arxiv.org/pdf/1809.00381) arXiv preprint arXiv:1809.00381 (2018).


### Evaluation Metrics

* Salamon, Justin, Emilia Gómez, Daniel PW Ellis, and Gaël Richard. ["Melody extraction from polyphonic music signals: Approaches, applications, and challenges."](https://repositori.upf.edu/bitstream/handle/10230/42183/Gomez_iee_melo.pdf?sequence=1&isAllowed=y) IEEE Signal Processing Magazine 31.2 (2014): 118-134.

* Bittner, Rachel M., and Juan J. Bosch. ["Generalized Metrics for Single-f0 Estimation Evaluation."](https://archives.ismir.net/ismir2019/paper/000090.pdf) International Society for Music Information Retrieval (ISMIR) Conference. 2019.