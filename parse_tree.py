import json
import pickle

# from embed import sentence_to_vec
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

  trees = [EntailmentTree(tree_json) for tree_json in trees_json[:2]]

  # trees = []
  # for tree_json in trees_json:
  #   print(f'TREE: {tree_json["meta"]["lisp_proof"]}')
  #   tree = EntailmentTree(tree_json)
  #   print(tree)
  #   trees.append(tree)

  return trees


if __name__ == "__main__":
  dataset = "dev"
  original_dataset_fp = f'data/dataset/task_1/{dataset}.jsonl'
  processed_dataset_fp = f'data/processed/{dataset}.pkl'

  # trees = parse_trees(original_dataset_fp)

  # with open(processed_dataset_fp, 'wb') as file:
  #   pickle.dump(trees, file)

  with open(processed_dataset_fp, 'rb') as file:
    trees = pickle.load(file)

  for i in range(len(trees)):
    print(trees[i])

    print('GENERATED: ')
    generated = trees[i].generated_premises()
    print(generated)
    print([t.shape for t in trees[i].to_embedding([[id] for id in generated])])

    print('RETRIEVED: ')
    retrieved = trees[i].retrieved_premises()
    print(retrieved)
    print([t.shape for t in trees[i].to_embedding(retrieved)])

    print('AVAILABLE: ')
    available = trees[i].available_premises()
    print(available)
    print([t.shape for t in trees[i].to_embedding(available)])
