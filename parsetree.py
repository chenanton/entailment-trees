import json
import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA

from entailmenttree import EntailmentTree


def parse_trees(file_path):
  """
  Parses JSON encoded dateset stored at filepath 
  and returns a list of trees.
  """
  trees_json = []
  with open(file_path, 'r') as file:
    for line in file:
      json_data = json.loads(line)
      trees_json.append(json_data)

  trees = [EntailmentTree(tree_json) for tree_json in trees_json]
  return trees


def reduce_embeddings(trees, d_new):
  """
  Given a list of trees, 
  reduce the dimension of all embeddings
  from d to d' via PCA. 

  :param trees: the list of trees
  :param d_new: reduced dimensionality

  :return trees: the updated list of trees
  """

  # Get N x d tensor of all embeddings
  embeddings = []
  for tree in trees:
    for id, embedding in tree.id_to_embedding.items():
      embeddings.append(embedding.unsqueeze(0))

  embeddings = torch.cat(embeddings, dim=0)
  print(f'embeddings.shape = {embeddings.shape}')

  # Fit PCA
  N, d = embeddings.shape
  pca = PCA(n_components=d_new)
  pca.fit(embeddings.numpy())  # Fit PCA on the data

  # Update embeddings
  for tree in trees:
    for id, embedding in tree.id_to_embedding.items():
      e = embedding.unsqueeze(0).numpy()
      tree.id_to_embedding[id] = torch.tensor(pca.transform(e)).squeeze()

  return trees


if __name__ == "__main__":
  dataset = "dev"
  original_dataset_fp = f'data/task_1/{dataset}.jsonl'
  processed_dataset_fp = f'data/processed/{dataset}.pkl'

  trees = parse_trees(original_dataset_fp)
  trees = reduce_embeddings(trees, 32)

  with open(processed_dataset_fp, 'wb') as file:
    pickle.dump(trees, file)
