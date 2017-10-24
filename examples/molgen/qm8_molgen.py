#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:11:10 2017

@author: zqwu
"""

import deepchem as dc
_, datasets, _ = dc.molnet.load_qm8_molgen(featurizer='Weave', split='random')

model = dc.models.MolGeneratorTensorGraph(
    atom_case=['End', 'C', 'N', 'O'],
    bond_case=['N/A', 'Single', 'Double', 'Triple', 'Aromatic'],
    lambd = 1.,
    n_hidden=64,
    n_graph_feat=128,
    output_max_out=12,
    use_queue=False, 
    learning_rate=1e-5,
    batch_size=64)

model.fit(datasets[0], nb_epoch=10000, checkpoint_interval=500)