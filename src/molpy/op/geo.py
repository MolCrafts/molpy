import molpy as mp
import numpy as np
from .utils import op

@op(input_key=[mp.Alias.xyz], output_key=[mp.Alias.xyz])
def move(xyz:np.array, vec:np.ndarray):
    return xyz + vec

@op(input_key=[mp.Alias.xyz], output_key=[mp.Alias.xyz])
def rotate(xyz:np.array, angle:float, axis:np.ndarray):
    """rotate by using Rodrigues' rotation formula

    Args:
        xyz (np.array): _description_
        angle (float): _description_
        axis (np.ndarray): _description_
    """
    pass