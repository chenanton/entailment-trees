import json
from parse_tree import parse_tree_string, print_tree


def read_jsonl_file(file_path):
  trees = []
  with open(file_path, 'r') as file:
    for line in file:
      json_data = json.loads(line)
      trees.append(json_data)
  return trees


# Example usage:
file_path = 'data/dataset/task_1/dev.jsonl'
trees = read_jsonl_file(file_path)
for i in range(20):
  lisp_proof = json.dumps(trees[i]["meta"]["lisp_proof"])[1:-1]
  print(f'\n\nLISP PROOF {i}')
  print(lisp_proof)
  tree = parse_tree_string(lisp_proof)
  print_tree(tree)
