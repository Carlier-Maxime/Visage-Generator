class Config:
    def __init__(self, **kwargs):
        self.set(**kwargs)

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def str2list(value, type_element):
        value = value[1:-1].split(",")
        value = [type_element(v) for v in value]
        return value
