"""
Stochastic chain generator for G-BigSMILES polymer growth.

This module implements breadth-first stochastic growth for building polymers
with complex topologies using template-based monomer placement.
"""

from collections import deque
from typing import Sequence

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.builder.polymer.connectors import ReacterConnector
from molpy.builder.polymer.growth_kernel import GrowthKernel
from molpy.builder.polymer.port_utils import get_all_port_info, PortInfo
from molpy.builder.polymer.system import SupportsDP, SupportsMass
from molpy.builder.polymer.types import MonomerTemplate, StochasticChain


class StochasticChainGenerator:
    """BFS-based stochastic chain generator for G-BigSMILES polymers.
    
    This generator builds polymer chains using breadth-first search (BFS)
    growth, where each added monomer introduces new reactive ports that
    enter a queue. Growth proceeds until chain mass or DP target is reached.
    
    Key features:
    - Template-based monomer placement
    - Port-resolved stochastic growth
    - Mass-based or DP-based stopping criteria
    - BFS port queue management
    """
    
    def __init__(
        self,
        growth_kernel: GrowthKernel,
        monomer_templates: Sequence[MonomerTemplate],
        connector: ReacterConnector,
        distribution: SupportsDP | SupportsMass,
        seed_template: MonomerTemplate | None = None,
    ):
        """Initialize stochastic chain generator.
        
        Args:
            growth_kernel: GrowthKernel for choosing next monomers
            monomer_templates: Available monomer templates
            connector: ReacterConnector for applying reactions
            distribution: Distribution for sampling target (DP or mass)
            seed_template: Template for seed monomer (uses first template if None)
        """
        self.kernel = growth_kernel
        self.templates = list(monomer_templates)
        self.connector = connector
        self.distribution = distribution
        self.seed_template = seed_template or self.templates[0]
        
        # Determine stopping criterion based on distribution capabilities
        self.use_mass_criterion = isinstance(distribution, SupportsMass)
    
    def build_chain(self, rng: np.random.Generator | None = None) -> StochasticChain:
        """Build a single polymer chain using BFS growth.
        
        Args:
            rng: NumPy random number generator (creates default if None)
            
        Returns:
            StochasticChain with polymer, DP, mass, and growth history
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # 1. Sample target from distribution
        if self.use_mass_criterion:
            target_mass = self.distribution.sample_mass(rng)  # type: ignore
            target_dp = None
        else:
            target_dp = self.distribution.sample_dp(rng)  # type: ignore
            target_mass = None
        
        # 2. Initialize with seed monomer
        polymer = self.seed_template.instantiate()
        mass = self.seed_template.mass
        dp = 1
        
        # 3. Initialize active ports queue (BFS)
        # get_all_port_info now returns dict[str, list[PortInfo]], so we need to flatten
        active_ports: deque[PortInfo] = deque()
        for port_list in get_all_port_info(polymer).values():
            for port_info in port_list:
                active_ports.append(port_info)
        
        growth_history = []
        
        # 4. BFS growth loop
        while active_ports:
            # Check stopping criteria
            if target_mass is not None and mass >= target_mass:
                break
            if target_dp is not None and dp >= target_dp:
                break
            
            # Pop next port from queue (FIFO for BFS)
            port = active_ports.popleft()
            
            # Choose next monomer
            placement = self.kernel.choose_next_for_port(
                polymer, port, self.templates, rng
            )
            
            if placement is None:
                # Terminate this port (no monomer added)
                continue
            
            # 5. Instantiate new monomer
            new_monomer = placement.template.instantiate()
            target_descriptor = placement.template.get_port_by_descriptor(
                placement.target_descriptor_id
            )
            
            if target_descriptor is None:
                # Invalid descriptor ID, skip
                continue
            
            # Find the actual port on the instantiated monomer
            target_port_name = target_descriptor.port_name
            new_monomer_ports = get_all_port_info(new_monomer)
            
            if target_port_name not in new_monomer_ports:
                # Port not found on instantiated monomer, skip
                continue
            
            # Get the first port with this name (index 0)
            target_port = new_monomer_ports[target_port_name][0]
            
            # 6. Apply reaction via Connector
            try:
                result = self.connector.connect(
                    polymer,
                    new_monomer,
                    left_label=self.seed_template.label,  # Placeholder
                    right_label=placement.template.label,
                    port_L=port.name,
                    port_R=target_port.name,
                )
                
                polymer = result.product
                mass += placement.template.mass
                dp += 1
                
                # 7. Add new active ports to queue (excluding used port)
                new_ports = get_all_port_info(polymer)
                for port_name, port_list in new_ports.items():
                    for port_info in port_list:
                        # Only add ports that weren't in the original polymer
                        # and aren't the port we just used
                        if port_name != target_port.name:
                            # Check if this is a newly added port
                            # (Simple heuristic: if it's from the new monomer)
                            active_ports.append(port_info)
                
                growth_history.append({
                    "template": placement.template.label,
                    "port_used": port.name,
                    "target_descriptor": placement.target_descriptor_id,
                    "target_port": target_port.name,
                })
                
            except Exception as e:
                # Connection failed, skip this monomer
                # In production, might want to log this
                continue
        
        return StochasticChain(
            polymer=polymer,
            dp=dp,
            mass=mass,
            growth_history=growth_history,
        )
    
    def _choose_seed(self, rng: np.random.Generator) -> MonomerTemplate:
        """Choose seed monomer template.
        
        Args:
            rng: Random number generator
            
        Returns:
            MonomerTemplate for seed
        """
        if self.seed_template is not None:
            return self.seed_template
        
        # Default: choose first template
        return self.templates[0]
