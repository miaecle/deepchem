"""
Script that trains Tensorflow singletask models on QM7 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem as dc
import numpy as np
from deepchem.molnet import load_qm7_from_mat
from deepchem.models.tensorgraph.optimizers import ExponentialDecay

np.random.seed(123)
qm7_tasks, datasets, transformers = load_qm7_from_mat(split='stratified')
train_dataset, valid_dataset, test_dataset = datasets
fit_transformers = [dc.trans.CoulombFitTransformer(train_dataset)]
metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.pearson_r2_score, mode="regression")
]


rate = ExponentialDecay(0.001, 0.95, 1000)
model = dc.models.MultiTaskFitTransformRegressor(
    n_tasks=1,
    n_features=[23, 23],
    learning_rate=rate,
    momentum=.8,
    batch_size=25,
    weight_init_stddevs=[1 / np.sqrt(400), 1 / np.sqrt(100), 1 / np.sqrt(100)],
    bias_init_consts=[0., 0., 0.],
    layer_sizes=[400, 100, 100],
    dropouts=[0.01, 0.01, 0.01],
    fit_transformers=fit_transformers,
    n_evals=10,
    seed=123)
#model.restore()

# Fit trained model
model.fit(train_dataset, nb_epoch=3000)

train_scores = model.evaluate(train_dataset, metric, transformers)
print("Train scores [kcal/mol]")
print(train_scores)

valid_scores = model.evaluate(valid_dataset, metric, transformers)
print("Valid scores [kcal/mol]")
print(valid_scores)

test_scores = model.evaluate(test_dataset, metric, transformers)
print("Test scores [kcal/mol]")
print(test_scores)
