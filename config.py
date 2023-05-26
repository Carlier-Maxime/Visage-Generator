class Config:
    def __init__(self, **kwargs):
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _str_to_list(value, type):
        value = value[1:-1].split(",")
        value = [type(v) for v in value]
        return value