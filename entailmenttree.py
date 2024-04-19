from treenode import TreeNode
from embed import sentence_to_vec


class EntailmentTree:
  """
  Represents an entailment tree.

  Sentences and tree structure are decoupled.
  """

  def __init__(self, tree_json):
    # Sentences
    self.id_to_sentence = self._parse_sentences(tree_json)
    self.id_to_embedding = self._parse_embedding(self.id_to_sentence)

    # Tree structure
    tree_string = tree_json["meta"]["lisp_proof"]
    tokens = tree_string.replace('(', ' ( ').replace(')', ' ) ').split()
    self.root = self._parse_root(tokens, 0, len(tokens) - 1)

  def __str__(self):
    return '\n'.join([f'{k}: {v}' for k, v in self.id_to_sentence.items()
                     ]) + '\n' + self.root.__str__()

  def generated_premises(self):
    """
    Returns a list of premises generated at each timestep
    based on the tree.

    :return: list of T ids, for T iterations.
    """

    def generated_helper(root):
      if len(root.children) == 0:
        return []

      # Recurse then concatenate
      ids = [generated_helper(child) for child in root.children]
      ids = [id for l in ids for id in l]
      ids.append(root.id)

      return ids

    return generated_helper(self.root)

  def available_premises(self):
    """
    Recursively fetches the IDs for the 
    available premises at each iteration.
    
    :return: return list of T lists
    """

    def get_initial_premises(root):
      """
      Returns initial premises, a.k.a. leaf nodes.
      """
      if len(root.children) == 0:
        return [root.id]

      # Recurse and flatten
      ids = [get_initial_premises(child) for child in root.children]
      return [id for l in ids for id in l]

    ids_g = self.generated_premises()
    ids_r = self.retrieved_premises()

    # Construct available premises per iteration
    ids = [get_initial_premises(self.root)]
    for t in range(len(ids_g)):
      ids.append(ids[-1].copy())

      # Remove retrieved premises
      for id in ids_r[t]:
        if id in ids[-1]:
          ids[-1].remove(id)

      # Add generated premise
      ids[-1].append(ids_g[t])

    return ids

  def retrieved_premises(self):
    """
    Recursively fetches the IDs for the 
    retrieved premises at each iteration.

    The retrieved premises are simply a list
    of children for each node.
    Retrieved premises are ordered from left to right in the tree.

    Recursive algorithm:
    get list of retrieve premises for left, then right, then
    append (child for children of root) to result

    :return retrieved: list[T][K_t]
    """

    def retrieved_helper(root):
      if len(root.children) == 0:
        return None

      # Recurse
      ids = [retrieved_helper(child) for child in root.children]

      # Take out NoneTypes
      ids = [id for id in ids if id is not None]

      # Flatten
      ids = [id for l in ids for id in l]

      # Add current list of retrieved premises
      ids.append([child.id for child in root.children])

      return ids

    return retrieved_helper(self.root)

  def _to_embedding(self, ids):
    """
    Given a list of m lists of m_i IDs,
    returns an list of m_i x d torch tensor of embeddings.

    e.g. [[id1 id2] [id3 id4 id5]] -> [tensor1 tensor2]
    where tensor1 is 2 x d and tensor2 is 3 x d.
    """
    pass

  def _parse_sentences(self, tree_json):
    """
    Extracts sentences from tree json from dataset.
    Each sentences corresponds to an id.
    """
    return {
        **tree_json["meta"]["triples"],
        **tree_json["meta"]["intermediate_conclusions"],
    }

  def _parse_embedding(self, id_to_sentence):
    """
    Returns embeddings for each sentence.
    """
    return {
        id: sentence_to_vec(sentence) for id, sentence in id_to_sentence.items()
    }

  def _parse_root(self, tokens, start, end):
    """
    Must have a single root node.
    """
    id = tokens[end - 1]
    children = self._parse_children(tokens, start + 1, end - 3)
    root = TreeNode(id, children)

    return root

  def _parse_children(self, tokens, start, end):
    """
    Parses children into an array.
    """
    children = []

    i = start + 1

    while i < end:
      # Case 1: base case (leaf node)
      if tokens[i] != "(":
        id = tokens[i]
        children.append(TreeNode(id, []))
      # Case 2: recursive case
      else:
        j = self._find_matching_parenthesis(tokens, i)
        child = self._parse_root(tokens, i, j)
        children.append(child)
        i = j

      i += 1

    return children

  def _find_matching_parenthesis(self, tokens, left_index):
    left_count = 0
    for i in range(left_index, len(tokens)):
      if tokens[i] == '(':
        left_count += 1
      elif tokens[i] == ')':
        left_count -= 1
        if left_count == 0:
          return i
    return -1  # No matching right parenthesis found
