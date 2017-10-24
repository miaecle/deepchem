"""
qm8 dataset loader.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem


def load_qm8_molgen(featurizer='Weave', split='random', reload=True):
  data_dir = deepchem.utils.get_data_dir()
  if reload:
    save_dir = os.path.join(data_dir, "qm8_molgen/" + featurizer + "/" + split)

  dataset_file = os.path.join(data_dir, "qm8_molgen.csv")
  tasks = ["task"]

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return tasks, all_dataset, transformers

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  loader = deepchem.data.CSVLoader(
      tasks=tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file)
  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset)
  transformers = []
  if reload:
    deepchem.utils.save.save_dataset_to_disk(
        save_dir, train_dataset, valid_dataset, test_dataset, transformers)
  return tasks, (train_dataset, valid_dataset, test_dataset), transformers
