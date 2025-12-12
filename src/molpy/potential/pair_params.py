"""
Specialized TypeIndexedArray for pair potentials with combining rules.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal

from .utils import TypeIndexedArray


class PairTypeIndexedArray(TypeIndexedArray):
    """
    Specialized TypeIndexedArray for pair potential parameters.
    
    Handles combining rules (Lorentz-Berthelot, geometric, etc.) for
    computing pair parameters from individual atom type parameters.
    
    For pair potentials, we store per-atom-type parameters (epsilon, sigma)
    and apply combining rules when indexing with two atom types.
    
    Examples:
        >>> # Create with per-atom-type parameters
        >>> epsilon = PairTypeIndexedArray(
        ...     {'opls_135': 0.066, 'opls_140': 0.030},
        ...     combining_rule='geometric'
        ... )
        >>> 
        >>> # Index with single type (self-interaction)
        >>> epsilon['opls_135']  # 0.066
        >>> 
        >>> # Index with two types (cross-interaction, uses combining rule)
        >>> epsilon[('opls_135', 'opls_140')]  # sqrt(0.066 * 0.030) = 0.0445
        >>> 
        >>> # Array indexing with pairs
        >>> pairs = np.array([['opls_135', 'opls_140'], ['opls_140', 'opls_140']])
        >>> epsilon[pairs]  # [0.0445, 0.030]
    """
    
    def __init__(
        self,
        data: dict[str, float] | NDArray[np.floating] | float,
        combining_rule: Literal['geometric', 'arithmetic', 'harmonic'] = 'geometric',
    ):
        """
        Initialize PairTypeIndexedArray.
        
        Args:
            data: Dictionary mapping atom type names to parameters, or array
            combining_rule: Rule for combining parameters:
                - 'geometric': sqrt(a * b) - for epsilon
                - 'arithmetic': (a + b) / 2 - for sigma
                - 'harmonic': 2 * a * b / (a + b) - alternative
        """
        super().__init__(data)
        self.combining_rule = combining_rule
    
    def _combine(self, val_i: float, val_j: float) -> float:
        """
        Apply combining rule to two parameter values.
        
        Args:
            val_i: Parameter for first atom type
            val_j: Parameter for second atom type
            
        Returns:
            Combined parameter value
        """
        if self.combining_rule == 'geometric':
            return np.sqrt(val_i * val_j)
        elif self.combining_rule == 'arithmetic':
            return (val_i + val_j) / 2.0
        elif self.combining_rule == 'harmonic':
            if val_i == 0 or val_j == 0:
                return 0.0
            return 2.0 * val_i * val_j / (val_i + val_j)
        else:
            raise ValueError(f"Unknown combining rule: {self.combining_rule}")
    
    def __getitem__(self, key):
        """
        Index with single type, pair of types, or arrays.
        
        Args:
            key: Can be:
                - Single type (str or int): returns self-interaction parameter
                - Tuple of two types: returns combined parameter
                - Array of single types: returns array of self-interaction parameters
                - Array of pairs (shape: (n, 2)): returns array of combined parameters
                
        Returns:
            Single value or array of values
        """
        # Handle tuple of two types (pair interaction)
        if isinstance(key, tuple) and len(key) == 2:
            type_i, type_j = key
            val_i = super().__getitem__(type_i)
            val_j = super().__getitem__(type_j)
            return self._combine(val_i, val_j)
        
        # Handle array of pairs
        elif isinstance(key, np.ndarray):
            # Check if it's a 2D array of pairs
            if key.ndim == 2 and key.shape[1] == 2:
                # Array of pairs: [(type_i, type_j), ...]
                results = []
                for i in range(len(key)):
                    type_i = key[i, 0]
                    type_j = key[i, 1]
                    val_i = super().__getitem__(type_i)
                    val_j = super().__getitem__(type_j)
                    results.append(self._combine(val_i, val_j))
                return np.array(results)
            else:
                # Regular array indexing (single types)
                return super().__getitem__(key)
        
        # Single type (self-interaction) or integer index
        else:
            return super().__getitem__(key)
    
    def get_pair_matrix(self) -> NDArray[np.floating]:
        """
        Get full pair parameter matrix for all type combinations.
        
        Returns:
            Matrix of shape (n_types, n_types) with combined parameters
            for all pairs of atom types
        """
        if not self._use_labels:
            raise ValueError("Cannot create pair matrix without type labels")
        
        n_types = len(self._values)
        matrix = np.zeros((n_types, n_types))
        
        for i in range(n_types):
            for j in range(n_types):
                matrix[i, j] = self._combine(self._values[i], self._values[j])
        
        return matrix
    
    def get_pair_array(self, type_pairs: NDArray) -> NDArray[np.floating]:
        """
        Get combined parameters for an array of type pairs.
        
        Args:
            type_pairs: Array of shape (n_pairs, 2) with type labels or indices
            
        Returns:
            Array of combined parameters for each pair
        """
        return self[type_pairs]


def create_lj_parameters(
    epsilon_dict: dict[str, float],
    sigma_dict: dict[str, float],
) -> tuple[PairTypeIndexedArray, PairTypeIndexedArray]:
    """
    Create LJ parameter arrays with standard Lorentz-Berthelot combining rules.
    
    Args:
        epsilon_dict: Per-atom-type epsilon values
        sigma_dict: Per-atom-type sigma values
        
    Returns:
        Tuple of (epsilon_array, sigma_array) with combining rules applied
        
    Example:
        >>> epsilon_dict = {'opls_135': 0.066, 'opls_140': 0.030}
        >>> sigma_dict = {'opls_135': 3.5, 'opls_140': 2.5}
        >>> epsilon, sigma = create_lj_parameters(epsilon_dict, sigma_dict)
        >>> 
        >>> # Get cross-interaction parameters
        >>> eps_ij = epsilon[('opls_135', 'opls_140')]  # sqrt(0.066 * 0.030)
        >>> sig_ij = sigma[('opls_135', 'opls_140')]    # (3.5 + 2.5) / 2
    """
    # Epsilon uses geometric mean (Lorentz-Berthelot)
    epsilon = PairTypeIndexedArray(epsilon_dict, combining_rule='geometric')
    
    # Sigma uses arithmetic mean (Lorentz-Berthelot)
    sigma = PairTypeIndexedArray(sigma_dict, combining_rule='arithmetic')
    
    return epsilon, sigma
