from math import cos, sin


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vec_scale(a: list[float], s: float) -> list[float]:
    return [a[0] * s, a[1] * s, a[2] * s]


def _dot(a: list[float], b: list[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(v: list[float]) -> float:
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


def _unit(v: list[float]) -> list[float]:
    n = _norm(v)
    if n == 0:
        return [0.0, 0.0, 0.0]
    return [v[0] / n, v[1] / n, v[2] / n]


def _rodrigues_rotate(
    p: list[float], k: list[float], angle: float, about: list[float]
) -> list[float]:
    # p' = about + R (p - about)
    v = _vec_sub(p, about)
    kx, ky, kz = k
    c = cos(angle)
    s = sin(angle)
    # rotation matrix components
    r00 = c + kx * kx * (1 - c)
    r01 = kx * ky * (1 - c) - kz * s
    r02 = kx * kz * (1 - c) + ky * s

    r10 = ky * kx * (1 - c) + kz * s
    r11 = c + ky * ky * (1 - c)
    r12 = ky * kz * (1 - c) - kx * s

    r20 = kz * kx * (1 - c) - ky * s
    r21 = kz * ky * (1 - c) + kx * s
    r22 = c + kz * kz * (1 - c)

    x = r00 * v[0] + r01 * v[1] + r02 * v[2]
    y = r10 * v[0] + r11 * v[1] + r12 * v[2]
    z = r20 * v[0] + r21 * v[1] + r22 * v[2]
    return _vec_add([x, y, z], about)
