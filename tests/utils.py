import numpy as np

rng = np.random.default_rng()

def random_array(shape: tuple[int]) -> np.ndarray:
    return rng.random(shape)