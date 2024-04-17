import json


class TreeNode:
  """
  Data structure representing entailment tree.
  """

  def __init__(self, id, val):
    self.id = id
    self.val = val
    self.children = []


def parse_trees(file_path):
  """
  Parses JSON encoded dateset stored at filepath and returns a list of trees.
  """
  trees_json = []
  with open(file_path, 'r') as file:
    for line in file:
      json_data = json.loads(line)
      trees_json.append(json_data)

  trees = [parse_tree(tree_json) for tree_json in trees_json]

  return trees


def parse_tree(tree_json):
  """
  Parses a JSON encoded object from the EntailmentBank dataset
  into a tree with id and value (sentence).

  The tree structure is encoded in tree_json["meta"]["lisp_proof"]

  E.g. "((((sent1 sent3) -> int1) sent2) -> int2)"

  is parsed into the following tree:

  int2
  |
  └── sent2
      |
      └── int1
          |
          ├── sent1
          └── sent3
  """

  def parse_helper(tokens, id_to_sentence, start, end):
    """
    Must have a single root node.

    String looks like "((((sent1 sent3) -> int1) sent2) -> int2)"
    """
    id = tokens[end - 1]
    root = TreeNode(id, id_to_sentence[id])
    root.children = parse_children(tokens, id_to_sentence, start + 1, end - 3)

    return root

  def parse_children(tokens, id_to_sentence, start, end):
    """
    Parses the children into an array.
    Implemented using a stack.

    String looks like "(((sent1 sent3) -> int1) sent2)"
    """
    children = []

    i = start + 1

    while i < end:
      # Case 1: base case (leaf node)
      if tokens[i] != "(":
        id = tokens[i]
        children.append(TreeNode(id, id_to_sentence[id]))
      # Case 2: recursive case
      else:
        j = find_matching_parenthesis(tokens, i)
        children.append(parse_helper(tokens, id_to_sentence, i, j))
        i = j

      i += 1

    return children

  def find_matching_parenthesis(tokens, left_index):
    left_count = 0
    for i in range(left_index, len(tokens)):
      if tokens[i] == '(':
        left_count += 1
      elif tokens[i] == ')':
        left_count -= 1
        if left_count == 0:
          return i
    return -1  # No matching right parenthesis found

  def parse_sentences(tree_json):
    """
    Extracts sentences from tree json from dataset.
    Each sentences corresponds to an id.

    Initial premises are stored in "triples", while
    Intermediate premises are stored in "intermediate_conclusions"
    """
    return {
        **tree_json["meta"]["triples"],
        **tree_json["meta"]["intermediate_conclusions"],
    }

  id_to_sentence = parse_sentences(tree_json)

  tree_string = tree_json["meta"]["lisp_proof"]
  tokens = tree_string.replace('(', ' ( ').replace(')', ' ) ').split()
  return parse_helper(tokens, id_to_sentence, 0, len(tokens) - 1)


def print_tree(node, depth=0):
  if node is not None:
    print("  " * depth + node.id + ": " + node.val)
    for child in node.children:
      print_tree(child, depth + 1)
