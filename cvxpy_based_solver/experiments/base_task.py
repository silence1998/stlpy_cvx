from abc import ABC, abstractmethod
from Models.base_model import flatten_before_sub

class BaseTask(ABC):
    def __init__(self, K):
        self.K = K
        pass

    def flatten_before_sub(self, formula):

        return flatten_before_sub(formula, self.K)

    @abstractmethod
    def add_to_plot(self, ax):
        pass