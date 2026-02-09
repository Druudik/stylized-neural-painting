import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


class RandomWrapper:
    def __init__(
        self,
        seed: int | None = None
    ):
        if seed is None:
            seed = torch.random.seed()

        self.g_map = {"cpu": torch.Generator("cpu").manual_seed(seed)}

    def state_dict(self) -> dict[str, Any]:
        state = {
            device: (generator.initial_seed(), generator.get_state())
            for device, generator in self.g_map.items()
        }
        return state

    def load_state_dict(self, state: dict[str, Any]):
        self.g_map = {}

        for device, (initial_seed, generator_state) in state.items():
            try:
                gen = torch.Generator(device).manual_seed(initial_seed)
                gen.set_state(generator_state.cpu())
                self.g_map[device] = gen
            except RuntimeError as e:
                warnings.warn(f"Could not create torch.Generator for device {device} when loading RandomGenerator state. Exception was: {e}")
                continue

    def rand(self, size: Sequence[int] | None = None, device: str | None = None) -> torch.Tensor:
        if size is None:
            size = 1
        device, generator = self._resolve_device_and_generator(device)
        r = torch.rand(size, generator=generator, device=device)
        return r

    def rand_int(self, low: int, high: int, size: Sequence[int], device: str | None = None) -> torch.Tensor:
        device, generator = self._resolve_device_and_generator(device)
        r = torch.randint(low=low, high=high, size=size, generator=generator, device=device)
        return r

    def normal(
        self,
        mean: Tensor | float,
        std: Tensor | float,
        size: Sequence[int],
        device: str | None = None
    ) -> torch.Tensor:
        device, generator = self._resolve_device_and_generator(device)
        # We're using torch.randn instead of torch.normal since torch.randn does not require synchronization when std is CUDA tensor.
        r = mean + std * torch.randn(size, generator=generator, device=device)
        return r

    def multinomial(
        self,
        probs: Tensor,
        num_samples: int,
        replacement: bool = False,
    ) -> Tensor:
        device, generator = self._resolve_device_and_generator(str(probs.device))
        return torch.multinomial(probs, num_samples, replacement=replacement, generator=generator)

    def _resolve_device_and_generator(self, device: str | None) -> tuple[str, torch.Generator]:
        if device is None:
            device = torch.get_default_device()

        if device not in self.g_map:
            seed = torch.randint(low=0, high=torch.iinfo(torch.int64).max, size=(1,), generator=self.g_map["cpu"], device="cpu").item()
            self.g_map[device] = torch.Generator(device).manual_seed(seed)

        return device, self.g_map[device]


class RandomVector(ABC):

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """
        Size of the random vector.
        """
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """Samples n random vectors.

        Args:
            n_samples (int): Number of random vectors to be sampled.

        Returns:
            Tensor of shape [n_samples, vector_size]
        """
        pass

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]):
        pass


class RandomUniformVector(RandomVector):

    def __init__(self, vector_size: int, scale: float = 1.0, device: str = "cpu", seed: int | None = None):
        self._vector_size = vector_size
        self.scale = scale
        self.device = device
        self.random = RandomWrapper(seed=seed)

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def sample(self, n_samples: int) -> torch.Tensor:
        random_vectors = self.random.rand((n_samples, self.vector_size), device=self.device) * self.scale
        return random_vectors

    def state_dict(self) -> dict[str, Any]:
        return self.random.state_dict()

    def load_state_dict(self, state: dict[str, Any]):
        return self.random.load_state_dict(state)
