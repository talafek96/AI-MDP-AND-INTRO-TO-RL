import math

from DecisonTree import Leaf, Question, DecisionNode, class_counts
from utils import *

"""
Make the imports of python packages needed
"""


class ID3:
    def __init__(self, feature_names: list, min_for_pruning=0, target_attribute='diagnosis'):
        self.feature_names = feature_names
        self.target_attribute = target_attribute
        self.tree_root = None
        self.used_features = set()
        self.min_for_pruning = min_for_pruning

    @staticmethod
    def entropy(rows: np.ndarray, labels: np.ndarray):
        """
        Calculate the entropy of a distribution for the classes probability values.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: entropy value.
        """
        # TODO:
        #  Calculate the entropy of the data as shown in the class.
        #  - You can use counts as a helper dictionary of label -> count, or implement something else.

        counts = class_counts(rows, labels)
        impurity = 0.0

        # ====== YOUR CODE: ======
        num_samples = len(rows)
        
        for count in counts.values():
            p = count / num_samples
            impurity += -p * np.log2(p)
        # ========================

        return impurity

    def info_gain(self, left, left_labels, right, right_labels, current_uncertainty):
        """
        Calculate the information gain, as the uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        :param left: the left child rows.
        :param left_labels: the left child labels.
        :param right: the right child rows.
        :param right_labels: the right child labels.
        :param current_uncertainty: the current uncertainty of the current node
        :return: the info gain for splitting the current node into the two children left and right.
        """
        # TODO:
        #  - Calculate the entropy of the data of the left and the right child.
        #  - Calculate the info gain as shown in class.
        assert (len(left) == len(left_labels)) and (len(right) == len(right_labels)), \
            'The split of current node is not right, rows size should be equal to labels size.'

        info_gain_value = 0.0
        # ====== YOUR CODE: ======
        num_left = len(left)
        num_right = len(right)
        num_total = num_left + num_right

        info_gain_value = current_uncertainty \
            - (num_left / num_total) * self.entropy(left, left_labels) \
            - (num_right / num_total)  * self.entropy(right, right_labels)
        # ========================

        return info_gain_value

    def partition(self, rows, labels, question: Question, current_uncertainty):
        """
        Partitions the rows by the question.
        :param rows: array of samples
        :param labels: rows data labels.
        :param question: an instance of the Question which we will use to partition the data.
        :param current_uncertainty: the current uncertainty of the current node
        :return: Tuple of (gain, true_rows, true_labels, false_rows, false_labels)
        """
        # TODO:
        #   - For each row in the dataset, check if it matches the question.
        #   - If so, add it to 'true rows', otherwise, add it to 'false rows'.
        #   - Calculate the info gain using the `info_gain` method.

        gain, true_rows, true_labels, false_rows, false_labels = None, [], [], [], []
        assert len(rows) == len(labels), 'Rows size should be equal to labels size.'

        # ====== YOUR CODE: ======
        rows = np.array(rows)

        for row, label in zip(rows, labels):
            if question.match(row):
                true_rows.append(row)
                true_labels.append(label)
            
            else:
                false_rows.append(row)
                false_labels.append(label)
        
        gain = self.info_gain(false_rows, false_labels, true_rows, true_labels, current_uncertainty)
        # ========================

        return gain, true_rows, true_labels, false_rows, false_labels

    def find_best_split(self, rows, labels):
        """
        Find the best question to ask by iterating over every feature / value and calculating the information gain.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: Tuple of (best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels)
        """
        # TODO:
        #   - For each feature of the dataset, build a proper question to partition the dataset using this feature.
        #   - find the best feature to split the data. (using the `partition` method)
        best_gain = - math.inf  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        best_false_rows, best_false_labels = None, None
        best_true_rows, best_true_labels = None, None
        current_uncertainty = self.entropy(rows, labels)

        # ====== YOUR CODE: ======
        rows = np.array(rows)

        for i, feature in enumerate(rows.transpose()):
            sorted_feature = np.sort(feature)
            thresholds = [(sorted_feature[j] + sorted_feature[j + 1]) / 2 for j in range(len(sorted_feature) - 1)]
            questions = [Question(f'{self.feature_names[i]}^{threshold:.2f}', i, threshold) for threshold in thresholds]
            gains = np.array([self.partition(rows, labels, question, current_uncertainty)[0] for question in questions])

            max_gain_index = gains.argmax()
            max_gain = gains[max_gain_index]

            if max_gain >= best_gain:
                best_gain = max_gain
                best_question = questions[max_gain_index]
        
        _, best_true_rows, best_true_labels, best_false_rows, best_false_labels = \
            self.partition(rows, labels, best_question, current_uncertainty)
        # ========================

        return best_gain, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels

    def build_tree(self, rows, labels):
        """
        Build the decision Tree in recursion.
        :param rows: array of samples
        :param labels: rows data labels.
        :return: a Question node, This records the best feature / value to ask at this point, depending on the answer.
                or leaf if we have to prune this branch (in which cases ?)

        """
        # TODO:
        #   - Try partitioning the dataset using the feature that produces the highest gain.
        #   - Recursively build the true, false branches.
        #   - Build the Question node which contains the best question with true_branch, false_branch as children
        best_question = None
        true_branch, false_branch = None, None

        # ====== YOUR CODE: ======
        counts = class_counts(rows, labels)
        if len(counts) <= 1 or sum(counts.values()) <= self.min_for_pruning:
            return Leaf(rows, labels)

        _, best_question, best_true_rows, best_true_labels, best_false_rows, best_false_labels = \
            self.find_best_split(rows, labels)
        # ========================

        return DecisionNode(
            best_question,
            self.build_tree(best_true_rows, best_true_labels), 
            self.build_tree(best_false_rows, best_false_labels)
        )

    def fit(self, x_train, y_train):
        """
        Trains the ID3 model. By building the tree.
        :param x_train: A labeled training data.
        :param y_train: training data labels.
        """
        # TODO: Build the tree that fits the input data and save the root to self.tree_root

        # ====== YOUR CODE: ======
        self.tree_root = self.build_tree(x_train, y_train)
        # ========================

    def predict_sample(self, row, node: DecisionNode or Leaf = None):
        """
        Predict the most likely class for single sample in subtree of the given node.
        :param row: vector of shape (1,D).
        :return: The row prediction.
        """
        # TODO: Implement ID3 class prediction for set of data.
        #   - Decide whether to follow the true-branch or the false-branch.
        #   - Compare the feature / value stored in the node, to the example we're considering.

        if node is None:
            node = self.tree_root
        prediction = None

        # ====== YOUR CODE: ======
        while not isinstance(node, Leaf):
            if node.question.match(row):
                node = node.true_branch
            else:
                node = node.false_branch
        
        prediction = self._get_prediction(node.predictions)
        # ========================

        return prediction
    
    def _get_prediction(self, predictions: dict) -> str:
        assert 1 <= len(predictions) <= 2, 'Only support a binary target class'

        if len(predictions) == 2 and len(set(predictions.values())) == 1:
            # Two types of examples got to the leaf, and the same amount from each.
            # Return 'M' according to the Piazza forum.
            return 'M'
        
        # Otherwise, return the prediction with the highest count
        return max(predictions, key=lambda x: predictions[x])

    def predict(self, rows):
        """
        Predict the most likely class for each sample in a given vector.
        :param rows: vector of shape (N,D) where N is the number of samples.
        :return: A vector of shape (N,) containing the predicted classes.
        """
        # TODO:
        #  Implement ID3 class prediction for set of data.

        y_pred = None

        # ====== YOUR CODE: ======
        y_pred = np.array([self.predict_sample(row) for row in rows], dtype=object)
        # ========================

        return y_pred
