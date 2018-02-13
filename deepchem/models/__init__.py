"""
Gathers all models in one place for convenient imports
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from deepchem.models.models import Model
from deepchem.models.sklearn_models import SklearnModel
from deepchem.models.xgboost_models import XGBoostModel
from deepchem.models.multitask import SingletaskToMultitask

from deepchem.models.tensorflow_models.fcnet import MultiTaskRegressor
from deepchem.models.tensorflow_models.fcnet import MultiTaskClassifier
from deepchem.models.tensorflow_models.fcnet import MultiTaskFitTransformRegressor
from deepchem.models.tensorflow_models.IRV import TensorflowMultiTaskIRVClassifier
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.models.graph_models import WeaveTensorGraph, DTNNTensorGraph, DAGTensorGraph, GraphConvTensorGraph, MPNNTensorGraph
from deepchem.models.tensorgraph.models.symmetry_function_regression import BPSymmetryFunctionRegression, ANIRegression

from deepchem.models.tensorgraph.models.seqtoseq import SeqToSeq
from deepchem.models.tensorgraph.models.gan import GAN, WGAN
from deepchem.models.tensorgraph.models.text_cnn import TextCNNTensorGraph
from deepchem.models.tensorgraph.models.mol_gen import MolGeneratorTensorGraph, MolGeneratorVAE
from deepchem.models.tensorgraph.sequential import Sequential
from deepchem.models.tensorgraph.models.sequence_dnn import SequenceDNN
