#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:06:13 2017

@author: zqwu
"""
import collections

import numpy as np
import six
import tensorflow as tf
from rdkit import Chem

from deepchem.data import NumpyDataset
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot, from_one_hot
from deepchem.models.tensorgraph.graph_layers import WeaveGather, SetGather, PairMap
from deepchem.models.tensorgraph.graph_layers import WeaveLayerFactory, MergeLoss
from deepchem.models.tensorgraph.layers import Dense, Concat, SoftMax, \
  SoftMaxCrossEntropy, GraphConv, BatchNorm, LSTM, SparseSoftMaxCrossEntropy, \
  WeightedError, Dropout, BatchNormalization, Stack, Flatten, Reshape, Constant
from deepchem.models.tensorgraph.layers import L2Loss, Label, Weights, Feature
from deepchem.models.tensorgraph.layers import TensorWrapper
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.trans import undo_transforms

class MolGeneratorTensorGraph(TensorGraph):

  def __init__(self,
               atom_case=['End', 'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'],
               bond_case=['N/A', 'Single', 'Double', 'Triple', 'Aromatic'],
               lambd = 1.,
               n_atom_feat=75,
               n_pair_feat=14,
               n_hidden=64,
               n_graph_feat=256,
               output_max_out=64,
               **kwargs):
    """
    Parameters
    ----------
    atom_case: list, optional
      List of possible labels for the nodes
    bond_case: list, optional
      List of possible labels for the edges
    lambd: float, optional
      Ratio of bond classification loss versus atom number classification loss
    n_atom_feat: int, optional
      Number of features per atom.
    n_pair_feat: int, optional
      Number of features per pair of atoms.
    n_hidden: int, optional
      Number of units(convolution depths) in corresponding hidden layer
    n_graph_feat: int, optional
      Number of output features for each molecule(graph)
    mode: str
      Either "classification" or "regression" for type of model.
    """
    self.atom_case = atom_case
    self.bond_case = bond_case
    self.lambd = lambd
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.n_hidden = n_hidden
    self.n_graph_feat = n_graph_feat
    self.output_max_out = output_max_out
    super(MolGeneratorTensorGraph, self).__init__(**kwargs)
    self.build_graph()

  def build_graph(self):
    self.atom_features = Feature(shape=(None, self.n_atom_feat))
    self.pair_features = Feature(shape=(None, self.n_pair_feat))
    self.pair_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_split = Feature(shape=(None,), dtype=tf.int32)
    self.atom_to_pair = Feature(shape=(None, 2), dtype=tf.int32)
    weave_layer1A, weave_layer1P = WeaveLayerFactory(
        n_atom_input_feat=self.n_atom_feat,
        n_pair_input_feat=self.n_pair_feat,
        n_atom_output_feat=self.n_hidden,
        n_pair_output_feat=self.n_hidden,
        in_layers=[
            self.atom_features, self.pair_features, self.pair_split,
            self.atom_to_pair
        ])
    weave_layer2A, weave_layer2P = WeaveLayerFactory(
        n_atom_input_feat=self.n_hidden,
        n_pair_input_feat=self.n_hidden,
        n_atom_output_feat=self.n_hidden,
        n_pair_output_feat=self.n_hidden,
        update_pair=False,
        in_layers=[
            weave_layer1A, weave_layer1P, self.pair_split, self.atom_to_pair
        ])
    dense1 = Dense(
        out_channels=self.n_graph_feat,
        activation_fn=tf.nn.tanh,
        in_layers=weave_layer2A)
    batch_norm1 = BatchNormalization(epsilon=1e-5, mode=1, in_layers=[dense1])
    weave_gather = WeaveGather(
        self.batch_size,
        n_input=self.n_graph_feat,
        gaussian_expand=True,
        in_layers=[batch_norm1, self.atom_split])
    
    
    zero_inputs = Constant(np.zeros((self.batch_size, self.output_max_out, self.n_graph_feat)))
    self.atom_vectors_out = LSTM(
        self.n_graph_feat,
        self.batch_size,
        in_layers=[zero_inputs, weave_gather])
    
    # Atom numbers outputs and loss
    atom_numbers = Dense(len(self.atom_case), in_layers=[self.atom_vectors_out])
    self.atom_numbers_label = Label(shape=(self.batch_size, self.output_max_out), 
                               dtype=tf.int32)
    
    atom_numbers_softmax = SoftMax(in_layers=[atom_numbers])
    self.add_output(atom_numbers_softmax)
    
    atom_numbers_cost = SparseSoftMaxCrossEntropy(in_layers=[self.atom_numbers_label, atom_numbers])
    self.weights = Weights(shape=(self.batch_size, self.output_max_out))
    atom_numbers_weights = Reshape([-1, 1], in_layers=[self.weights])
    self.atom_numbers_loss = WeightedError(in_layers=[atom_numbers_cost, atom_numbers_weights])
    
    # Connectivity outputs and loss
    pair_map = PairMap(in_layers=[self.atom_vectors_out])
    self.connectivity = Dense(len(self.bond_case), in_layers=[pair_map])
    self.bonds_label = Label(shape=(self.batch_size, self.output_max_out, self.output_max_out), 
                               dtype=tf.int32)
    
    bonds_softmax = SoftMax(in_layers=[self.connectivity])
    self.add_output(bonds_softmax)
    
    bonds_cost = SparseSoftMaxCrossEntropy(in_layers=[self.bonds_label, self.connectivity])
    bonds_weights = PairMap(in_layers=[self.weights])
    bonds_weights = Reshape([-1, 1], in_layers=[bonds_weights])
    self.bonds_loss = WeightedError(in_layers=[bonds_cost, bonds_weights])
    
    self.all_loss = MergeLoss(self.lambd, in_layers=[self.atom_numbers_loss,
                                                self.bonds_loss])
    self.set_loss(self.all_loss)


  def default_generator(self,
                        dataset,
                        epochs=1,
                        predict=False,
                        deterministic=True,
                        pad_batches=True):
    """ TensorGraph style implementation
    similar to deepchem.models.tf_new_models.graph_topology.AlternateWeaveTopology.batch_to_feed_dict
    """
    for epoch in range(epochs):
      if not predict:
        print('Starting epoch %i' % epoch)
      for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
          batch_size=self.batch_size,
          deterministic=deterministic,
          pad_batches=pad_batches):

        feed_dict = {}
        atom_feat = []
        pair_feat = []
        atom_split = []
        atom_to_pair = []
        pair_split = []
        atom_numbers_labels = []
        bonds_labels = []
        start = 0
        for im, mol in enumerate(X_b):
          n_atoms = mol.get_num_atoms()
          # number of atoms in each molecule
          atom_split.extend([im] * n_atoms)
          # index of pair features
          C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
          atom_to_pair.append(
              np.transpose(
                  np.array([C1.flatten() + start,
                            C0.flatten() + start])))
          # number of pairs for each atom
          pair_split.extend(C1.flatten() + start)
          start = start + n_atoms

          # atom features
          atom_feat.append(mol.get_atom_features())
          # pair features
          pair_feat.append(
              np.reshape(mol.get_pair_features(), (n_atoms * n_atoms,
                                                   self.n_pair_feat)))

          atom_numbers_label, bonds_label = self.generate_label(mol)
          atom_numbers_labels.append(atom_numbers_label)
          bonds_labels.append(bonds_label)
        
        # Padding
        n_valid_samples = len(atom_feat)
        for i in range(self.batch_size - n_valid_samples):
          im = i + n_valid_samples
          mol = X_b[i]
          n_atoms = mol.get_num_atoms()
          atom_split.extend([im] * n_atoms)
          C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
          atom_to_pair.append(
              np.transpose(
                  np.array([C1.flatten() + start,
                            C0.flatten() + start])))
          pair_split.extend(C1.flatten() + start)
          start = start + n_atoms
          atom_feat.append(mol.get_atom_features())
          pair_feat.append(
              np.reshape(mol.get_pair_features(), (n_atoms * n_atoms,
                                                   self.n_pair_feat)))

          atom_numbers_label, bonds_label = self.generate_label(mol)
          atom_numbers_labels.append(atom_numbers_label)
          bonds_labels.append(bonds_label)
        
        feed_dict[self.atom_features] = np.concatenate(atom_feat, axis=0)
        feed_dict[self.pair_features] = np.concatenate(pair_feat, axis=0)
        feed_dict[self.pair_split] = np.array(pair_split)
        feed_dict[self.atom_split] = np.array(atom_split)
        feed_dict[self.atom_to_pair] = np.concatenate(atom_to_pair, axis=0)
        feed_dict[self.bonds_label] = np.stack(bonds_labels, axis=0)
        feed_dict[self.atom_numbers_label] = np.stack(atom_numbers_labels, axis=0)
        feed_dict[self.weights] = np.concatenate([np.ones((n_valid_samples, self.output_max_out)),
                 np.zeros((self.batch_size - n_valid_samples, self.output_max_out))], axis=0)
        
        yield feed_dict

  def generate_label(self, mol):

    bonds = mol.get_pair_features()[:, :, :4]
    bonds = np.concatenate([np.zeros_like(bonds)[:,:, :1], bonds], axis=2)
    bonds = from_one_hot(bonds, axis=2)
    atom_numbers = from_one_hot(mol.get_atom_features()[:, :44])
    n_atoms = len(atom_numbers)
    #order = np.argsort(atom_numbers)
    order = np.arange(n_atoms)
    return self.tokenize(atom_numbers[order]), \
        np.pad(bonds[:, order][order, :], ((0, self.output_max_out - n_atoms),
                                           (0, self.output_max_out - n_atoms)),
               'constant')
    
    
  def tokenize(self, atom_numbers):
    symbol_dict = [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',  # H?
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'Unknown'
    ]
    atom_case_dict = dict(zip(self.atom_case, range(len(self.atom_case))))
    out = [atom_case_dict[symbol_dict[i]] for i in list(atom_numbers)]
    n_atoms = len(out)
    for i in range(self.output_max_out - n_atoms):
      out.append(atom_case_dict["End"])
    return out

  def predict(self, dataset, transformers=[], batch_size=None):
    # MPNN only accept padded input
    generator = self.default_generator(dataset, predict=True, pad_batches=False)
    return self.predict_on_generator(generator, transformers)

  def predict_on_generator(self, generator, transformers=[]):
    """
    Returns:
      y_pred: numpy ndarray of shape (n_samples, n_classes*n_tasks)
    """
    if not self.built:
      self.build()
    with self._get_tf("Graph").as_default():
      out_tensors = [x.out_tensor for x in self.outputs]
      results = [[], []]
      for feed_dict in generator:
        # Extract number of unique samples in the batch from w_b
        n_valid_samples = int(sum(feed_dict[self.weights][:, 0]))
        feed_dict = {
            self.layers[k.name].out_tensor: v
            for k, v in six.iteritems(feed_dict)
        }
        feed_dict[self._training_placeholder] = 0.0
        result = self.session.run(out_tensors, feed_dict=feed_dict)
        for i, out in enumerate(result):
          results[i].append(out[:n_valid_samples])
      return results

  def predict_mol(self, dataset, transformers=[], batch_size=None):
    pred = self.predict(dataset, transformers, batch_size)
    atoms_pred = from_one_hot(np.concatenate(pred[0], axis=0), axis=2)
    bonds_pred = from_one_hot(np.concatenate(pred[1], axis=0), axis=3)
    output_mols = []
    for i in range(atoms_pred.shape[0]):
      output_mols.append(self.rebuild_mol(atoms_pred[i], bonds_pred[i]))
    return output_mols
  
  def rebuild_mol(self, atoms, bonds):
    _bondtypes = {1: Chem.BondType.SINGLE,
                  2: Chem.BondType.DOUBLE,
                  3: Chem.BondType.TRIPLE,
                  4: Chem.BondType.AROMATIC}
    try:
      mol = Chem.Mol()
      edmol = Chem.EditableMol(mol)
      for i, atom_id in enumerate(atoms):
        atom = self.atom_case[atom_id]
        if atom == 'End':
          end = i
          break
        rdatom = Chem.Atom(atom)
        edmol.AddAtom(rdatom)
      for i in range(end):
        for j in range(i+1, end):
          if bonds[i, j] > 0:
            edmol.AddBond(i, j, _bondtypes[bonds[i,j]])
      mol = edmol.GetMol()
      Chem.SanitizeMol(mol)
      return Chem.MolToSmiles(mol)
    except:
      return None