import numpy as np

class BinaryClassification:
    def __init__(self, dataset, bias):
        self.self = self
        self.dataset = dataset
        self.bias = bias

    #Path to the file, in this case will be a csv format.
    #Include the bias desired.
    def SimplePerceptron(self, dataset_path, bias):
        with open(dataset_path, "r") as f:
            dataset = f.read()
            f.close()
        pass
