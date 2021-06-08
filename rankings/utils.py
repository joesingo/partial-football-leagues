import itertools

def listify(func):
    def inner(*args, **kwargs):
        return list(func(*args, **kwargs))
    return inner

def kendall_tau_distance(l1: list, l2: list) -> int:
    """
    Compute the Kendall tau distance of two rankings l1 and l2.

    l1 and l2 should be lists of set same length which order a common set of
    elements (i.e. set(l1) == set(l2))
    """
    n = len(l1)
    assert len(l2) == n
    l2_ranks = {x: i for i, x in enumerate(l2)}

    tau = 0
    for i, j in itertools.combinations(range(n), 2):
        x = l1[i]
        y = l1[j]
        i2 = l2_ranks[x]
        j2 = l2_ranks[y]
        if (i < j and i2 > j2) or (i > j and i2 < j2):
            tau += 1
    return tau

def all_subclasses(cls):
    for child in cls.__subclasses__():
        yield child
        yield from all_subclasses(child)
