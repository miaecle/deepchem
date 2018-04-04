"""
Script that trains ANI models on qm7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
from deepchem.models.tensorgraph.optimizers import ExponentialDecay


# Load Tox21 dataset
tasks, datasets, transformers = dc.molnet.load_qm7_from_mat(
    featurizer='BPSymmetryFunction', split='random', reload=False)
train_dataset, valid_dataset, test_dataset = datasets

# Batch size of models
max_atoms = 23
batch_size = 128
layer_structures = [128, 128, 64]
atom_number_cases = [1, 6, 7, 8, 16]

# Fit models
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]

valids = []
tests = []
for ct in range(6):
  model_dir = '/home/zqwu/deepchem/examples/qm7/ANI1_model'
  rate = ExponentialDecay(0.01 * 10**(-ct), 0.9, 1000)
  model = dc.models.ANIRegression(
      len(tasks),
      max_atoms,
      layer_structures=layer_structures,
      atom_number_cases=atom_number_cases,
      batch_size=batch_size,
      learning_rate=rate,
      use_queue=False,
      mode="regression",
      model_dir=model_dir)
  if ct > 0:
    model.restore()
  # Fit trained model
  model.fit(train_dataset, nb_epoch=1000, checkpoint_interval=1000)
  train_scores = model.evaluate(train_dataset, metric, transformers)
  valid_scores = model.evaluate(valid_dataset, metric, transformers)
  valids.append(valid_scores['mean_absolute_error'])
  test_scores = model.evaluate(test_dataset, metric, transformers)
  tests.append(test_scores['mean_absolute_error'])

print("Evaluating model")
for i in range(100):
  model.fit(train_dataset, nb_epoch=1, checkpoint_interval=100)
  valid_scores = model.evaluate(valid_dataset, metric, transformers)
  valids.append(valid_scores['mean_absolute_error'])
  test_scores = model.evaluate(test_dataset, metric, transformers)
  tests.append(test_scores['mean_absolute_error'])