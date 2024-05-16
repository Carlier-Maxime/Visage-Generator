from typing import Any
import yaml


class ConfigItems:
    def __init__(self, cfg):
        self.cfg = cfg

    def __iter__(self):
        for key in self.cfg:
            val = self.cfg[key]
            if isinstance(val, dict): val = Config(val)
            yield key, val


class Config(dict):
    @staticmethod
    def transform_out(out):
        return Config(out) if isinstance(out, dict) else None if out == 'None' else out

    def __getattr__(self, name: str) -> Any:
        try:
            out = self[name]
            return self.transform_out(out)
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __getitem__(self, item):
        out = dict.__getitem__(self, item)
        return self.transform_out(out)

    def get(self, key):
        return Config.transform_out(dict.get(self, key))

    def items(self):
        return ConfigItems(self)

    @staticmethod
    def str2list(value, type_element):
        value = value[1:-1].split(",")
        value = [type_element(v) for v in value]
        return value

    @staticmethod
    def __merge_dict(base_dict, dict2):
        for key, val in dict2.items():
            base_dict[key] = Config.__merge_dict(base_dict.get(key), val) if isinstance(val, dict) else val
        return base_dict

    @staticmethod
    def fromYml(path: str):
        default = yaml.safe_load(open("configs/default.yml", 'r'))
        cfg = yaml.safe_load(open(path, 'r'))
        return Config(Config.__merge_dict(default, cfg))
