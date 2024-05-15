from abc import abstractmethod

import torch


class BaseParamsGenerator:
    def __init__(self, nb_params, min_value, max_value, device: torch.device) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.nb_params = nb_params
        self.device = device

    @abstractmethod
    def generate(self, size) -> torch.Tensor:
        pass

    @abstractmethod
    def one(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get(self, count: int, same: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def get_nb_params(self) -> int:
        pass


class ParamsGenerator(BaseParamsGenerator):
    def __init__(self, nb_params: int, min_value: float, max_value: float, device: torch.device) -> None:
        super().__init__(nb_params, min_value, max_value, device)

    def generate(self, size) -> torch.Tensor:
        return torch.rand(size, dtype=torch.float32, device=self.device) * (self.max_value - self.min_value) + self.min_value

    def one(self) -> torch.Tensor:
        return self.generate(self.nb_params)

    def zeros(self, count: int = -1) -> torch.Tensor:
        if count == -1:
            size = self.nb_params
        else:
            size = [count, self.nb_params]
        return torch.zeros(size, device=self.device)

    def get(self, count: int, same: bool = False) -> torch.Tensor:
        params = self.generate((1 if same else count, self.nb_params))
        return params.repeat(count, 1) if same else params

    def get_nb_params(self) -> int:
        return self.nb_params


class MultiParamsGenerator(BaseParamsGenerator):
    def __init__(self, nb_params: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor, device: torch.device) -> None:
        BaseParamsGenerator.__init__(self, nb_params, min_value, max_value, device)
        self.params_generators = []
        for i in range(len(nb_params)):
            self.params_generators.append(ParamsGenerator(nb_params[i], min_value[i], max_value[i], device))

    @classmethod
    def from_params(cls, params: torch.Tensor | list, device: torch.device, deg2rad: bool = False) -> "MultiParamsGenerator":
        if isinstance(params, list):
            params = torch.tensor(params, device=device)
        if deg2rad:
            return cls(params[::3].to(torch.int), params[1::3].deg2rad(), params[2::3].deg2rad(), device)
        return cls(params[::3].to(torch.int), params[1::3], params[2::3], device)

    def generate(self, size) -> torch.Tensor:
        assert size[-1] == self.get_nb_params(), f'a size format is not good, size:{size} size[-1]=={self.get_nb_params()}'
        size = list(size)
        params = []
        for i in range(len(self.nb_params)):
            size[-1] = self.nb_params[i]
            params.append(self.params_generators[i].generate(size))
        return torch.cat(params, dim=1)

    def one(self) -> torch.Tensor:
        return self.get(1).flatten()

    def zeros(self, count: int = -1) -> torch.Tensor:
        if count == -1:
            size = self.get_nb_params()
        else:
            size = [count, self.get_nb_params()]
        return torch.zeros(size, device=self.device)

    def get(self, count: int, same: bool = False, keyframes: int | None = None) -> torch.Tensor:
        params = self.generate((1 if same else keyframes if keyframes is not None else count, self.get_nb_params()))
        if keyframes is not None:
            if same: params = params.repeat(keyframes, 1)
            start_values = params[:-1]
            end_values = params[1:]
            first = torch.linspace(0, 1, count // (keyframes - 1) + count % (keyframes - 1), device=self.device).view(-1, 1)
            other = torch.linspace(0, 1, count // (keyframes - 1), device=self.device).view(-1, 1, 1)
            step_one = start_values[:1].mul(first.flip(dims=[0])).add(end_values[:1].mul(first))
            step_other = start_values[1:].mul(other.flip(dims=[0])).add(end_values[1:].mul(other)).permute(1, 0, 2).flatten(end_dim=-2)
            params = torch.cat([step_one, step_other])
        elif same: params = params.repeat(count, 1)
        return params

    def get_nb_params(self) -> int:
        return int(self.nb_params.sum())
