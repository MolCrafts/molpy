import numpy as np
from ..simulator import Simulator


class Fix:
    """
    Basic class for simulator hooks
    """

    def on_step_begin(self, simulator: Simulator):
        pass

    def on_step_middle(self, simulator: Simulator):
        pass

    def on_step_end(self, simulator: Simulator):
        pass

    def on_step_finalize(self, simulator: Simulator):
        pass

    def on_step_failed(self, simulator: Simulator):
        pass

    def on_simulation_start(self, simulator: Simulator):
        pass

    def on_simulation_end(self, simulator: Simulator):
        pass


class Thermostat(Fix):
    """
    Basic thermostat hook for simulator class. This class is initialized based on the simulator and system
    specifications during the first MD step. Thermostats are applied before and after each MD step.

    Args:
        temperature_bath (float): Temperature of the heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs.
    """

    ring_polymer = False

    def __init__(self, temperature_bath: float, time_constant: float):
        super().__init__()
        self.T = temperature_bath
        # Convert from fs to internal time units
        self.time_constant = time_constant
        self._initialized = False

    @property
    def initialized(self):
        """
        Auxiliary property for easy access to initialized flag used for restarts
        """
        return self._initialized

    @initialized.setter
    def initialized(self, flag):
        """
        Make sure initialized is set to torch.tensor for storage in state_dict.
        """
        self._initialized = flag

    def on_simulation_start(self, simulator: Simulator):
        """
        Routine to initialize the thermostat based on the current state of the simulator. Reads the device to be used.
        In addition, a flag is set so that the thermostat is not reinitialized upon continuation of the MD.

        Main function is the `_init_thermostat` routine, which takes the simulator as input and must be provided for every
        new thermostat.

        Args:
            simulator (schnetpack.simulation_hooks.simulator.Simulator): Main simulator class containing information on
                                                                         the time step, system, etc.
        """
        if not self.initialized:
            self._init_thermostat(simulator)
            self.initialized = True

    def on_step_begin(self, simulator: Simulator):
        """
        First application of the thermostat before the first half step of the dynamics. Regulates temperature.

        Main function is the `_apply_thermostat` routine, which takes the simulator as input and must be provided for
        every new thermostat.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Apply thermostat
        self._apply_thermostat(simulator)

    def on_step_end(self, simulator: Simulator):
        """
        Application of the thermostat after the second half step of the dynamics. Regulates temperature.

        Main function is the `_apply_thermostat` routine, which takes the simulator as input and must be provided for
        every new thermostat.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Apply thermostat
        self._apply_thermostat(simulator)

    def _init_thermostat(self, simulator: Simulator):
        """
        Dummy routine for initializing a thermostat based on the current simulator. Should be implemented for every
        new thermostat. Has access to the information contained in the simulator class, e.g. number of replicas, time
        step, masses of the atoms, etc.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        pass

    def _apply_thermostat(self, simulator: Simulator):
        """
        Dummy routine for applying the thermostat to the system. Should use the implemented thermostat to update the
        momenta of the system contained in `simulator.system.momenta`. Is called twice each simulation time step.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        raise NotImplementedError


class Langevin(Thermostat):

    def __init__(self, T: float, time_constant: float):

        super().__init__(T, time_constant)

        self.T = T
        self.time_constant = time_constant

        self._initilalized = False

    @property
    def initilalized(self):
        return self._initilalized

    @initilalized.setter
    def initilalized(self, value: bool):
        self._initilalized = value

    def on_step_begin(self, simulator: Simulator):

        pass

    def _init_thermostat(self, simulator: Simulator):
        """
        Initialize the Langevin coefficient matrices based on the system and simulator properties.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        gamma = np.array([1]) / self.time_constant
        c1 = np.exp(-0.5 * simulator.integrator.timestep * gamma)
        c2 = np.sqrt(1 - c1**2)
        self.c1 = c1[:, None]
        self.c2 = c2[:, None]
        kb = 1
        self.thermostat_factor = np.sqrt(simulator.frame.mass * kb * self.T)

    def _apply_thermostat(self, simulator: Simulator):
        """
        Apply the stochastic Langevin thermostat to the systems momenta.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Get current momenta
        momenta = simulator.frame.momenta

        # Generate random noise
        thermostat_noise = np.random.randn(*momenta.shape)

        # Apply thermostat
        simulator.frame.momenta = (
            self.c1 * momenta + self.thermostat_factor * self.c2 * thermostat_noise
        )
