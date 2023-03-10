import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayMemory:
    def __init__(
        self,
        capacity: int,
    ):
        self._capacity = capacity
        self._num_added = 0
        self._storage = [None] * capacity

    def add(self, transition) -> None:
        self._storage[self._num_added % self._capacity] = transition
        self._num_added += 1

    def sample(self, batch_size: int = 1):
        indices = np.random.randint(0, self.size, batch_size)
        samples = [self._storage[i] for i in indices]
        return samples

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
