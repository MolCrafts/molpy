"""
Low-level 3D geometry operations.

Pure Python implementations of vector math and rotations for molecular
coordinate manipulation. All functions use list[float] for 3D vectors.

Note:
    These are internal utilities. For high-performance operations,
    consider using numpy arrays instead.
"""
from math import cos, sin


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    """Add two 3D vectors: a + b."""
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    """Subtract two 3D vectors: a - b."""
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _vec_scale(a: list[float], s: float) -> list[float]:
    """Scale a 3D vector: a * s."""
    return [a[0] * s, a[1] * s, a[2] * s]


def _dot(a: list[float], b: list[float]) -> float:
    """
    Compute dot product of two 3D vectors.
    
    Args:
        a: First vector [x, y, z]
        b: Second vector [x, y, z]
    
    Returns:
        Scalar dot product a · b
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: list[float], b: list[float]) -> list[float]:
    """
    Compute cross product of two 3D vectors.
    
    Args:
        a: First vector [x, y, z]
        b: Second vector [x, y, z]
    
    Returns:
        Vector cross product a × b
    """
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(v: list[float]) -> float:
    """
    Compute Euclidean norm (magnitude) of a 3D vector.
    
    Args:
        v: Vector [x, y, z]
    
    Returns:
        ||v|| = sqrt(x² + y² + z²)
    """
    return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ** 0.5


def _unit(v: list[float]) -> list[float]:
    """
    Normalize a 3D vector to unit length.
    
    Args:
        v: Vector [x, y, z]
    
    Returns:
        Unit vector v/||v||, or [0,0,0] if ||v|| = 0
    """
    n = _norm(v)
    if n == 0:
        return [0.0, 0.0, 0.0]
    return [v[0] / n, v[1] / n, v[2] / n]


def _rodrigues_rotate(
    p: list[float], k: list[float], angle: float, about: list[float]
) -> list[float]:
    """
    Rotate a point using Rodrigues' rotation formula.
    
    Rotates point p around axis k by angle (radians) about center point.
    Uses rotation matrix R = I + sin(θ)K + (1-cos(θ))K²
    where K is the skew-symmetric matrix of k.
    
    Args:
        p: Point to rotate [x, y, z]
        k: Unit rotation axis [x, y, z] (must be normalized)
        angle: Rotation angle in radians
        about: Center of rotation [x, y, z]
    
    Returns:
        Rotated point coordinates [x', y', z']
    
    Examples:
        >>> # Rotate point [1,0,0] by 90° around z-axis
        >>> p = [1.0, 0.0, 0.0]
        >>> k = [0.0, 0.0, 1.0]  # z-axis
        >>> angle = 3.14159 / 2  # 90 degrees
        >>> rotated = _rodrigues_rotate(p, k, angle, [0,0,0])
        >>> # rotated ≈ [0, 1, 0]
    """
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

