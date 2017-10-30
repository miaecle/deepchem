#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:11:10 2017

@author: zqwu
"""

import deepchem as dc
import tensorflow as tf
_, datasets, _ = dc.molnet.load_qm9_molgen(featurizer='Weave')

model_dir = '/home/zqwu/deepchem/examples/molgen/qm9_molgen/'
model = dc.models.MolGeneratorVAE(
    atom_case=['End', 'C', 'N', 'O'],
    bond_case=['N/A', 'Single', 'Double', 'Triple', 'Aromatic'],
    lambd = 1.,
    n_hidden=64,
    n_graph_feat=256,
    n_latent=128,
    output_max_out=12,
    use_queue=False, 
    learning_rate=1e-5,
    batch_size=64,
    model_dir=model_dir)
model.build()

with model._get_tf("Graph").as_default():
  saver = tf.train.Saver()
  saver.restore(model.session, model_dir + 'model-214999')
      
model.fit(datasets[0], nb_epoch=10000, checkpoint_interval=5000)