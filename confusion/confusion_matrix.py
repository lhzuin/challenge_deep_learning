import matplotlib.pyplot as plt
import numpy as np

class Confusion_matrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    def update(self, true_labels, pred_labels):
        for t, p in zip(true_labels, pred_labels):
            self.matrix[t][p] += 1

    def compute(self):
        return self.matrix

    def print_matrix(self, class_names=None, figsize=(8, 6), cmap="Blues", save_path=None):
        """Display the confusion matrix as a heatmap plot."""
        matrix = np.array(self.matrix)
        plt.figure(figsize=figsize)
        im = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title("Confusion Matrix")
        plt.colorbar(im)
        tick_marks = np.arange(self.num_classes)
        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Annotate each cell with the count
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, format(matrix[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if matrix[i, j] > thresh else "black")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def accuracy(self):
        correct = 0
        total = 0
        for i in range(self.num_classes):
            correct += self.matrix[i][i]
            total += sum(self.matrix[i])
        return correct / total if total > 0 else 0
    def precision(self):
        precision = []
        for i in range(self.num_classes):
            tp = self.matrix[i][i]
            fp = sum(self.matrix[j][i] for j in range(self.num_classes) if j != i)
            precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        return precision
    def recall(self):
        recall = []
        for i in range(self.num_classes):
            tp = self.matrix[i][i]
            fn = sum(self.matrix[i][j] for j in range(self.num_classes) if j != i)
            recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        return recall
    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        f1_scores = []
        for i in range(self.num_classes):
            if precision[i] + recall[i] > 0:
                f1_scores.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
            else:
                f1_scores.append(0)
        return f1_scores
    def print_metrics(self):
        accuracy = self.accuracy()
        precision = self.precision()
        recall = self.recall()
        f1_scores = self.f1_score()

        print("Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        for i in range(self.num_classes):
            print(f"Class {i}:")
            print(f"\tPrecision: {precision[i]:.4f}")
            print(f"\tRecall: {recall[i]:.4f}")
            print(f"\tF1 Score: {f1_scores[i]:.4f}")
        print()
