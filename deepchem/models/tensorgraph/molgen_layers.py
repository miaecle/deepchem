#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:49:46 2017

@author: zqwu
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from deepchem.nn import activations
from deepchem.nn import initializations
from deepchem.nn import model_ops

from deepchem.models.tensorgraph.layers import Layer, LayerSplitter
from deepchem.models.tensorgraph.layers import convert_to_layers

class PairMap(Layer):
  """ Predicting connectivity(bonding) within the graph
  """

  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):

    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)

    atom_vectors = in_layers[0].out_tensor
    n_atoms = atom_vectors.get_shape().as_list()[1]
    if atom_vectors.get_shape().ndims < 3:
      atom_vectors = tf.expand_dims(atom_vectors, 2)
    
    out_tensor = tf.multiply(tf.tile(tf.expand_dims(atom_vectors, 2), (1, 1, n_atoms, 1)),
        tf.tile(tf.expand_dims(atom_vectors, 1), (1, n_atoms, 1, 1)))
    
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor


class MergeLoss(Layer):

  def __init__(self, lambd, **kwargs):
    self.lambd = lambd
    super(MergeLoss, self).__init__(**kwargs)
    
  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):

    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    
    out_tensor = in_layers[0].out_tensor + in_layers[1].out_tensor * self.lambd
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor
  

class GenerateLatent(Layer):
    
  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):

    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    
    mean = in_layers[0].out_tensor 
    log_var = in_layers[1].out_tensor
    
    eps = in_layers[2].out_tensor
    out_tensor = mean + tf.sqrt(tf.exp(log_var)) * eps
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor
  
class KLDivergence(Layer):
  
  def create_tensor(self, in_layers=None, set_tensors=True, **kwargs):

    if in_layers is None:
      in_layers = self.in_layers
    in_layers = convert_to_layers(in_layers)
    
    mean = in_layers[0].out_tensor 
    log_var = in_layers[1].out_tensor
    out_tensor = 0.5 * tf.reduce_sum(tf.exp(log_var) + tf.square(mean) - 1 - log_var)
    if set_tensors:
      self.out_tensor = out_tensor
    return out_tensor