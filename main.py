import json
from parse_tree import parse_trees, print_tree

file_path = 'data/dataset/task_1/dev.jsonl'
trees = parse_trees(file_path)

for i in range(20):
  print_tree(trees[i])
