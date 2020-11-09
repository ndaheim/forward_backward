from functools import reduce
import operator

import numpy as np
from scipy.special import logsumexp

class Semiring(object):
    """Abstract class defining components of a semiring."""

    def add(self, summands):
        """Addition of the given elements in the semiring."""
        pass

    def multiply(self, multiplicants):
        """Multiplication of the given elements in the semiring."""
        pass

    def one(self):
        """Returns the neutral element w.r.t multiplication."""
        pass

    def zero(self):
        """Returns the neutral element w.r.t to addition."""
        pass

class LogSemiring(Semiring):

    def add(self, summands):
        return logsumexp(summands)

    def multiply(self, multiplicants):
        return sum(multiplicants)

    def one(self):
        return 0.0

    def zero(self):
        return np.inf

class ProbabilitySemiring(Semiring):

    def add(self, summands):
        return sum(summands)

    def multiply(self, multiplicants):
        return reduce(operator.mul, multiplicants, self.one())

    def one(self):
        return 1.0

    def zero(self):
        return 0.0

class TropicalSemiring(Semiring):
    """
    Implements the tropical semiring, in which the forward-backward algorithm
    realizes Viterbi approximation.
    """

    def add(self, summands):
        return min(summands)

    def multiply(self, multiplicants):
        return sum(multiplicants)

    def one(self):
        return 0.0

    def zero(self):
        return np.inf
