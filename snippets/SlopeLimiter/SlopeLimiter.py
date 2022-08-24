def minmod(r):
    phi = max(0, min(1.0, r))
    return phi


def vanLeer(r):
    absr = abs(r)
    phi = (r + absr) / (1 + absr)
    return phi


def Gregori(r):
    phi = max(0, min(2 * r, r ** 0.5, 2))
    return phi


def superbee(r):
    phi = max(0, min(2 * r, 1), min(r, 2))
    return phi


def osher(r, beta=1):
    assert 1 <= beta <= 2
    phi = max(0, min(r, beta))
    return phi


def monocentral(r):
    phi = max(0, min(2 * r, 0.5 * (1 + r), 2))
    return phi
