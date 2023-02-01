import numpy as np
import torch
import random
from typing import Any, Callable, Generic, Iterable, List, Mapping, Optional, Sequence, Text, Tuple, TypeVar

device = "cuda" if torch.cuda.is_available() else "cpu"


def importance_sampling_weights(
    probabilities: np.ndarray,
    uniform_probability: float,
    exponent: float,
    normalize: bool,
) -> np.ndarray:
    """Calculates importance sampling weights from given sampling probabilities.

    Args:
    probabilities: Array of sampling probabilities for a subset of items. Since
      this is a subset the probabilites will typically not sum to `1`.
    uniform_probability: Probability of sampling an item if uniformly sampling.
    exponent: Scalar that controls the amount of importance sampling correction
      in the weights. Where `1` corrects fully and `0` is no correction
      (resulting weights are all `1`).
    normalize: Whether to scale all weights so that the maximum weight is `1`.
      Can be enabled for stability since weights will only scale down.

    Returns:
    Importance sampling weights that can be used to scale the loss. These have
    the same shape as `probabilities`.
    """
    if not 0. <= exponent <= 1.:
        raise ValueError('Require 0 <= exponent <= 1.')
    if not 0. <= uniform_probability <= 1.:
        raise ValueError('Expected 0 <= uniform_probability <= 1.')

    weights = (uniform_probability / probabilities)**exponent
    if normalize:
        weights /= np.max(weights)
    if not np.isfinite(weights).all():
        raise ValueError('Weights are not finite: %s.' % weights)
    return weights


class PrioritizedReplayMemory:
    def __init__(
        self,
        capacity: int,
        priority_exponent: float,
        importance_sampling_exponent: Callable[[int], float],
        uniform_sample_probability: float,
        normalize_weights: bool
    ):
        self._capacity = capacity
        self._priority_exponent = priority_exponent
        self._importance_sampling_exponent = importance_sampling_exponent
        self._uniform_sample_probability = uniform_sample_probability
        self._normalize_weights = normalize_weights
        self._num_added = 0
        self._storage = [None] * capacity
        self._max_seen_priority = 1.

        self._distribution = PrioritizedDistribution(
            capacity=capacity,
            priority_exponent=priority_exponent,
            uniform_sample_probability=uniform_sample_probability)

    def add(self, transition, priority) -> None:
        index = self._num_added % self._capacity
        self._storage[index] = transition
        self._distribution.set_priorities([index], [priority])
        self._num_added += 1

    def get(self, indices):
        return [self._storage[i] for i in indices]

    def sample(self, batch_size: int = 1):
        indices, probabilities = self._distribution.sample(batch_size)
        weights = importance_sampling_weights(
            probabilities,
            uniform_probability=1. / self.size,
            exponent=self._importance_sampling_exponent(self._num_added),
            normalize=self._normalize_weights)
        samples = self.get(indices)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        priorities = np.asarray(priorities)
        self._distribution.update_priorities(indices, priorities)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return min(self._num_added, self._capacity)

    @property
    def steps_done(self) -> int:
        return self._num_added

    @property
    def storage(self) -> list:
        return self._storage


class PrioritizedDistribution:
    """Distribution for weighted sampling."""

    def __init__(
        self,
        capacity: int,
        priority_exponent: float,
        uniform_sample_probability: float,
    ):
        self._priority_exponent = priority_exponent
        self._uniform_sample_probability = uniform_sample_probability
        self._sum_tree = SumTree(capacity)
        self._active_indices = []
        self._active_indices_mask = np.zeros(capacity, dtype=np.bool)

    def set_priorities(self, indices, priorities):
        """Sets priorities for indices, whether or not all indices already exist."""
        for idx in indices:
            if not self._active_indices_mask[idx]:
                self._active_indices.append(idx)
                self._active_indices_mask[idx] = True
        self._sum_tree.set(indices, np.power(priorities, self._priority_exponent))

    def update_priorities(self, indices, priorities):
        """Updates priorities for existing indices."""
        for idx in indices:
            if not self._active_indices_mask[idx]:
                raise IndexError('Index %s cannot be updated as it is inactive.' % idx)
        self._sum_tree.set(indices, np.power(priorities, self._priority_exponent))

    def sample(self, size):
        """Returns sample of indices with corresponding probabilities."""
        uniform_indices = random.sample(self._active_indices, size)
        targets = np.random.uniform(size=size) * self._sum_tree.root()
        prioritized_indices = np.asarray(self._sum_tree.query(targets))
        usp = self._uniform_sample_probability
        # Desny: if transition's probability is smaller than usp, then choose uniform sampling
        # to ensure transitions with low priority can be sampled
        indices = np.where(np.random.uniform(size=size) < usp, uniform_indices, prioritized_indices)
        priorities = self._sum_tree.get(indices)
        prioritized_probs = priorities / self._sum_tree.root()
        return indices, prioritized_probs


class SumTree:
    """A binary tree where non-leaf nodes are the sum of child nodes.

      Leaf nodes contain non-negative floats and are set externally. Non-leaf nodes
      are the sum of their children. This data structure allows O(log n) updates and
      O(log n) queries of which index corresponds to a given sum. The main use
      case is sampling from a multinomial distribution with many probabilities
      which are updated a few at a time.
      """

    def __init__(self, size):
        """Initializes an empty `SumTree`."""
        # When there are n values, the storage array will have size 2 * n. The first
        # n elements are non-leaf nodes (ignoring the very first element), with
        # index 1 corresponding to the root node. The next n elements are leaf nodes
        # that contain values. A non-leaf node with index i has children at
        # locations 2 * i, 2 * i + 1.
        self._capacity = 0
        # size => leaf num
        self._size = size
        # first leaf index
        self._first_leaf = 0
        self._storage = np.zeros(0, dtype=np.float64)

        assert size >= 0
        new_capacity = 1
        while new_capacity < size:
            new_capacity *= 2
        new_storage = np.empty((2 * new_capacity,), dtype=np.float64)
        self._storage = new_storage
        self._first_leaf = new_capacity
        self._size = size

    def root(self) -> float:
        """Returns sum of values."""
        return self._storage[1] if self.size > 0 else np.nan

    def set(self, indices, values):
        """Sets values at the given indices."""
        values = np.asarray(values)
        if not np.isfinite(values).all() or (values < 0.).any():
            raise ValueError('value must be finite and positive.')
        for i, idx in enumerate(np.asarray(indices) + self._first_leaf):
            self._storage[idx] = values[i]
            # update parent
            parent = idx // 2
            while parent > 0:
                self._storage[parent] = self._storage[2 * parent] + self._storage[2 * parent + 1]
                parent //= 2

    def query(self, targets: Sequence[float]) -> Sequence[int]:
        """Finds smallest indices where `target <` cumulative value sum up to index.

        Args:
          targets: The target sums.

        Returns:
          For each target, the smallest index such that target is strictly less than
          the cumulative sum of values up to and including that index.

        Raises:
          ValueError: if `target >` sum of all values or `target < 0` for any
            of the given targets.
        """
        return [self._query_single(t) for t in targets]

    def _query_single(self, target: float) -> int:
        """Queries a single target, see query for more detailed documentation."""
        if not 0. <= target < self.root():
            raise ValueError('Require 0 <= target < total sum.')

        storage = self._storage
        idx = 1  # Root node.
        while idx < self._first_leaf:
            # At this point we always have target < storage[idx].
            assert target < storage[idx]
            left_idx = 2 * idx
            right_idx = left_idx + 1
            left_sum = storage[left_idx]
            if target < left_sum:
                idx = left_idx
            else:
                idx = right_idx
                target -= left_sum

        assert idx < 2 * self.capacity
        return idx - self._first_leaf


