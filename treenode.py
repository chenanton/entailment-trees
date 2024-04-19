class TreeNode:
  """
  Data structure representing tree structure.
  """

  def __init__(self, id, children):
    """
    Construct tree from lisp proof string.
    """
    self.id = id
    self.children = children

  def __str__(self):
    """
    Print tree.
    """

    return self._str_recursive(self, depth=0)

  def _str_recursive(self, node, depth):
    result = "  " * depth + node.id + "\n"
    for child in node.children:
      result += self._str_recursive(child, depth + 1)
    return result
