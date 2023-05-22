import torch

class BaseParamsGenerator():
    def __init__(self, nb_params, min_value:float, max_value:float, device:torch.device) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.nb_params = nb_params
        self.device = device

    def generate(self, size) -> torch.Tensor:
        return None
    
    def one(self) -> torch.Tensor:
        return None
    
    def get(self, count:int, same:bool=False) -> torch.Tensor:
        return None

class ParamsGenerator(BaseParamsGenerator):
    def __init__(self, nb_params, min_value:float, max_value:float, device:torch.device) -> None:
        BaseParamsGenerator.__init__(self, nb_params, min_value, max_value, device)

    def generate(self, size) -> torch.Tensor:
        return torch.rand(size, dtype=torch.float32, device=self.device) * (self.max_value - self.min_value) + self.min_value
    
    def one(self) -> torch.Tensor:
        return self.generate(self.nb_params)
    
    def zeros(self, count:int=-1) -> torch.Tensor:
        if count==-1: size = self.nb_params
        else: size = [count, self.nb_params]
        return torch.zeros(size, device=self.device)
    
    def get(self, count:int, same:bool=False) -> torch.Tensor:
        params = self.generate((1 if same else count, self.nb_params))
        return params.repeat(count, 1) if same else params

class PoseParamsGenerator(ParamsGenerator):
    def __init__(self, min_jaw1_value:float, max_jaw1_value:float, min_jaw2_3_value:float, max_jaw_2_3_value:float, device:torch.device) -> None:
        ParamsGenerator.__init__(self, 6, min_jaw1_value, max_jaw1_value, device)
        self.min_jaw2_3_value = min_jaw2_3_value
        self.max_jaw2_3_value = max_jaw_2_3_value

    def generate(self, size) -> torch.Tensor:
        if isinstance(size, int): assert size%6==0
        else: assert size[1]==6
        pose_params = torch.zeros(6, dtype=torch.float32, device=self.device)
        if not isinstance(size, int): pose_params = pose_params[None].repeat(size[0],1)
        jaw1_params = ParamsGenerator.generate(self, 1 if isinstance(size, int) else (size[0], 1))
        jaw2_3_params = torch.rand(2 if isinstance(size, int) else (size[0], 2), dtype=torch.float32, device=self.device) * (self.max_jaw2_3_value - self.min_jaw2_3_value) + self.min_jaw2_3_value
        jaw_params = torch.cat([jaw1_params, jaw2_3_params], dim=1)
        if isinstance(size, int): pose_params[3:6] = jaw_params
        else: pose_params[:,3:6] = jaw_params
        return pose_params