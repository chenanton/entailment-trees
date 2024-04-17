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
    """
    Must have a single root node.

    String looks like "((((sent1 sent3) -> int1) sent2) -> int2)"
    """
    root = TreeNode(tokens[end - 1])
    root.children = parse_children(tokens, start + 1, end - 3)

    return root

  def parse_children(tokens, start, end):
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
        children.append(TreeNode(tokens[i]))
      # Case 2: recursive case
      else:
        j = find_matching_parenthesis(tokens, i)
        children.append(parse_helper(tokens, i, j))
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

  tokens = tree_string.replace('(', ' ( ').replace(')', ' ) ').split()
  return parse_helper(tokens, 0, len(tokens) - 1)


def print_tree(node, depth=0):
  if node is not None:
    print("  " * depth + node.id)
    for child in node.children:
      print_tree(child, depth + 1)
