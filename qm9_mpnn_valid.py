#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:30:22 2017

@author: zqwu
"""
import deepchem
import numpy as np
import tensorflow as tf
import os
import pickle

seed=123
np.random.seed(seed)

data_dir = deepchem.utils.get_data_dir()

dataset_file = os.path.join(data_dir, "gdb9.sdf")
qm9_tasks = ["gap"]

featurizer = deepchem.feat.WeaveFeaturizer(graph_distance=False, explicit_H=True)
loader = deepchem.data.SDFLoader(
        tasks=qm9_tasks,
        smiles_field="smiles",
        mol_field="mol",
        featurizer=featurizer)
dataset = loader.featurize(dataset_file)

splitter = deepchem.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)

transformers = [deepchem.trans.NormalizationTransformer(transform_y=True, dataset=train_dataset)]
for transformer in transformers:
  train_dataset = transformer.transform(train_dataset)
  valid_dataset = transformer.transform(valid_dataset)
  test_dataset = transformer.transform(test_dataset)
  
batch_size = 80
nb_epoch = 10
learning_rate = 0.0005
T = 2 
M = 5
metric = [deepchem.metrics.Metric(deepchem.metrics.mean_absolute_error, mode="regression")]

tf.set_random_seed(seed)
model = deepchem.models.MPNNTensorGraph(
    1,
    n_atom_feat=70,
    n_pair_feat=8,
    n_hidden=20,
    T=T,
    M=M,
    batch_size=batch_size,
    learning_rate=learning_rate,
    use_queue=False,
    mode="regression")

model.build()
valid_scores = []
test_scores = []
for i in range(50):
  model.fit(train_dataset, nb_epoch=nb_epoch)
  #train_scores = model.evaluate(train_dataset, metric, transformers)
  valid_scores.append(model.evaluate(valid_dataset, metric, transformers))
  print(valid_scores)
  with open('/home/zqwu/deepchem/examples/qm9_mpnn_valid_results.pkl', 'w') as f:
    pickle.dump(valid_scores, f)
  test_scores.append(model.evaluate(test_dataset, metric, transformers))
  print(test_scores)
  with open('/home/zqwu/deepchem/examples/qm9_mpnn_test_results.pkl', 'w') as f:
    pickle.dump(test_scores, f)
  
