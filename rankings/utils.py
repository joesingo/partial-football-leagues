def listify(func):
    def inner(*args, **kwargs):
        return list(func(*args, **kwargs))
    return inner
