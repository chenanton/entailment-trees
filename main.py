import json


class TreeNode:

  def __init__(self, value):
    self.value = value
    self.children = []


def parse_tree(s):

  def parse_helper(tokens):
    if not tokens:
      return None

    token = tokens.pop(0)
    if token == "(":
      node = TreeNode(None)
      while tokens[0] != ")":
        node.children.append(parse_helper(tokens))
      tokens.pop(0)  # pop ")"
      return node
    elif token == ")":
      return None
    else:
      return token

  tokens = s.replace("(", " ( ").replace(")", " ) ").split()
  return parse_helper(tokens)


def print_tree(node, indent=0):
  if node is None:
    return
  if isinstance(node, TreeNode):
    print(' ' * indent + "Node")
    for child in node.children:
      print_tree(child, indent + 4)
  else:
    print(' ' * indent + str(node))


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
  lisp_proof = json.dumps(trees[i]["meta"]["lisp_proof"])
  print(lisp_proof)
  tree = parse_tree(lisp_proof)
  # print_tree(tree)
