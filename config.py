from typing import Any

import yaml
import click


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            out = self[name]
            return Config(out) if isinstance(out, dict) else out
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __getitem__(self, item):
        out = dict.__getitem__(self, item)
        return Config(out) if isinstance(out, dict) else out

    @staticmethod
    def str2list(value, type_element):
        value = value[1:-1].split(",")
        value = [type_element(v) for v in value]
        return value


def click_yaml(config_file: str):
    cfg = Config(yaml.safe_load(open(config_file, 'r')))

    def set_option_callback(opt_name: str):
        keys = opt_name[2:].split('-')

        def _set_option(cfg, keys, value):
            cfg[keys[0]] = _set_option(cfg[keys[0]], keys[1:], value) if len(keys) > 1 else None if value == 'None' else value
            return cfg

        def set_option(ctx, param, value):
            _set_option(cfg, keys, value)
            ctx.params['cfg'] = cfg
            return value

        return set_option

    def add_options(cfg: dict, callback, prefix: str = '-'):
        for name, value in cfg.items():
            opt_name = f'{prefix}-{name}'
            if isinstance(value, dict):
                add_options(value, callback, opt_name)
            else:
                callback(opt_name, value)

    def use_config(config_file: str):
        cfg = Config(yaml.safe_load(open(config_file, 'r')))
        return cfg

    def _click_yaml(func):
        def _click_add_option(opt_name, value):
            click.option(opt_name, default=value, type=type(value), help='for help see config file : ' + config_file, callback=set_option_callback(opt_name))(func)

        add_options(cfg, _click_add_option)
        func = click.option('--cfg', default=config_file, callback=lambda ctx, param, value: use_config(value))(func)
        return func

    return _click_yaml
