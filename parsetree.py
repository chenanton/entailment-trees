import json
import pickle

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


if __name__ == "__main__":
  dataset = "dev"
  original_dataset_fp = f'data/task_1/{dataset}.jsonl'
  processed_dataset_fp = f'data/processed/{dataset}.pkl'

  trees = parse_trees(original_dataset_fp)
  with open(processed_dataset_fp, 'wb') as file:
    pickle.dump(trees, file)
