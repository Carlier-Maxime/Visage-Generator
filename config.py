from typing import Any
import yaml


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

    @staticmethod
    def str2list(value, type_element):
        value = value[1:-1].split(",")
        value = [type_element(v) for v in value]
        return value

    @staticmethod
    def fromYml(path: str):
        return Config(yaml.safe_load(open(path, 'r')) | yaml.safe_load(open("configs/default.yml", 'r')))
