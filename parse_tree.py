class TreeNode:

  def __init__(self, id):
    self.id = id
    self.children = []


def parse_tree_string(tree_string):
  """
  Parses a lisp proof formatted string into a tree of type TreeNode.

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

  def parse_helper(tokens, start, end):
    res = []

    # Case 1
    if end - 3 >= 0 and tokens[end - 3] != '->':
      parse_children(tokens, start + 1, end - 1)
    else:
      node = TreeNode(tokens[end - 1])

    # initialize pointer at start

    # repeat:
    # 1. find letnod

  tokens = tree_string.replace('(', ' ( ').replace(')', ' ) ').split()
  print(tokens)
  return parse_helper(tokens, 0, len(tokens) - 1)


if __name__ == "__main__":
  tree_string = "((((sent1 sent3) -> int1) sent2) -> int2)"
  tree = parse_tree_string(tree_string)
  print_tree(tree)
