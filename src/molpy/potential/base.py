"""
Base classes for potential functions.
"""

from collections import UserList

import numpy as np

from ..core.forcefield import ForceField


class KernelMeta(type):
    def __new__(cls, clsname, bases, namespace, **kwargs):
        cls = super().__new__(cls, clsname, bases, namespace)
        typename = namespace.get("type", "root")
        if typename not in ForceField._kernel_registry:
            registry = ForceField._kernel_registry[typename] = {}
        else:
            registry = ForceField._kernel_registry[typename]
        registry[namespace.get("name", clsname)] = cls
        return cls


class Potential(metaclass=KernelMeta):
    """
    Base class for all potential functions in MolPy.

    This class provides a template for defining potential functions that can be used in molecular simulations.
    Each potential class should implement calc_energy and calc_forces methods with specific parameters.
    """

    def __call__(self, *args, **kwargs):
        """Evaluate the potential."""
        raise NotImplementedError("Subclasses must implement this method.")

    def calc_energy(self, *args, **kwargs) -> float:
        """
        Calculate the potential energy.

        Parameters
        ----------
        *args: Arguments specific to the potential type
        **kwargs: Keyword arguments specific to the potential type

        Returns
        -------
        float
            The potential energy.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def calc_forces(self, *args, **kwargs) -> np.ndarray:
        """
        Calculate the forces.

        Parameters
        ----------
        *args: Arguments specific to the potential type
        **kwargs: Keyword arguments specific to the potential type

        Returns
        -------
        np.ndarray
            An array of forces.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class Potentials(UserList[Potential]):
    """
    Collection of potential functions.

    This class provides a way to combine multiple potentials and calculate
    total energy and forces. However, since different potentials require
    different parameters, you need to call calc_energy and calc_forces
    for each potential separately and sum the results.

    For a simpler interface, use the helper functions in potential.utils
    to extract data from Frame objects.
    """

    def calc_energy(self, *args, **kwargs) -> float:
        """
        Calculate the total energy by summing energies from all potentials.

        Note: This assumes all potentials in the collection accept the same
        arguments. For different potential types, you should call calc_energy
        on each potential separately.

        Parameters
        ----------
        *args: Arguments passed to each potential's calc_energy method
        **kwargs: Keyword arguments passed to each potential's calc_energy method

        Returns
        -------
        float
            The total energy.
        """
        return sum(pot.calc_energy(*args, **kwargs) for pot in self)

    def calc_forces(self, *args, **kwargs) -> np.ndarray:
        """
        Calculate the total forces by summing forces from all potentials.

        Note: This assumes all potentials in the collection accept the same
        arguments and return forces with the same shape. For different potential
        types, you should call calc_forces on each potential separately and
        sum the results.

        Parameters
        ----------
        *args: Arguments passed to each potential's calc_forces method
        **kwargs: Keyword arguments passed to each potential's calc_forces method

        Returns
        -------
        np.ndarray
            An array of total forces.
        """
        if not self:
            raise ValueError(
                "Cannot determine force shape: no potentials in collection"
            )

        # Get forces from first potential to determine shape
        first_forces = self[0].calc_forces(*args, **kwargs)
        total_forces = np.zeros_like(first_forces)

        for pot in self:
            forces = pot.calc_forces(*args, **kwargs)
            total_forces += forces

        return total_forces
