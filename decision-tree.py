import numpy as np

def gini_impurity(labels):
    """
    Compute Gini Impurity for a given set of integer class labels.
    """
    counts = np.bincount(labels)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)
import numpy as np

class DecisionTreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.label = None
        self.depth = depth
        self.max_depth = max_depth

    def is_leaf(self):
        return self.label is not None

def gini_impurity(labels):
    counts = np.bincount(labels)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def best_split(X, y):
    m, n = X.shape
    best_feature, best_threshold = None, None
    best_gini = 1.0  # worst possible
    best_left_idx, best_right_idx = None, None

    for feature_index in range(n):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_idx = X[:, feature_index] <= threshold
            right_idx = X[:, feature_index] > threshold

            if left_idx.sum() == 0 or right_idx.sum() == 0:
                continue

            y_left, y_right = y[left_idx], y[right_idx]
            gini = (len(y_left) / m) * gini_impurity(y_left) + \
                   (len(y_right) / m) * gini_impurity(y_right)

            if gini < best_gini:
                best_gini = gini
                best_feature = feature_index
                best_threshold = threshold
                best_left_idx = left_idx
                best_right_idx = right_idx

    return best_feature, best_threshold, best_left_idx, best_right_idx

def build_tree(X, y, depth=0, max_depth=None):
    node = DecisionTreeNode(depth=depth, max_depth=max_depth)

    # If pure or max depth reached, set label
    if gini_impurity(y) == 0 or (max_depth is not None and depth >= max_depth):
        node.label = np.bincount(y).argmax()
        return node

    feature_index, threshold, left_idx, right_idx = best_split(X, y)
    if feature_index is None:
        node.label = np.bincount(y).argmax()
        return node

    node.feature_index = feature_index
    node.threshold = threshold
    node.left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
    node.right = build_tree(X[right_idx], y[right_id_]()_]()_
