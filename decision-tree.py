class Node:
    def __init__(self, is_leaf=False , value = None):
        self.is_leaf = is_leaf
        self.value = value
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self):
        self.root = Node(is_leaf=True)
    def add_split (self,leaf,signal,value):
        if not leaf.is_leaf:
            raise ValueError ("can't split non-leaf node")
        leaf.is_leaf = False
        leaf.split_feature = signal
        leaf.split_vale = value

        leaf.left = Node(is_leaf=True)
        leaf.right = Node(is_leaf=True)
        return leaf.right, leaf.left
    def set_leaf_value(self, leaf, value):
        if not leaf.is_leaf:
            raise ValueError("cant assign value to non-leaf node")
        leaf.value = value
    def eval (self, signals):
        node = self.Node()
        while not node.is_leaf:
            if signals[node.split_feature]<=node.split_vale:
                node = node.left
            else:
                node = node.right
        return node.value
