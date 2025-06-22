from .base import Packer
try:
    import nlopt
except ImportError:  # allow running without nlopt
    nlopt = None
import numpy as np
import molpy as mp


class ObjectiveFunction:
    def __init__(self, constraints):
        self.constraints = constraints if isinstance(constraints, list) else [constraints]

    def value(self, x):
        points = x.reshape(-1, 3)
        penalty = sum(constraint.penalty(points) for constraint in self.constraints)
        return penalty

    def gradient(self, x):
        points = x.reshape(-1, 3)
        grad = np.zeros_like(points)
        for constraint in self.constraints:
            grad += constraint.dpenalty(points)
        return grad.flatten()

    def __call__(self, x, grad):
        if grad.size > 0:
            grad[:] = self.gradient(x)
        return self.value(x)


class NloptPacker(Packer):

    def __init__(self, method="LD_MMA"):
        if nlopt is None:
            raise ImportError("nlopt is required for NloptPacker")
        super().__init__()
        self.method = getattr(nlopt, method)
        
    def pack(self, targets=None, max_steps: int = 1000, seed: int | None = None) -> 'mp.Frame':
        """Pack molecules using nlopt optimization."""
        # Use self.targets if none provided
        targets_to_use = targets if targets is not None else self.targets
        
        if not targets_to_use:
            return mp.Frame()
            
        # Calculate total number of points from targets
        n_points = sum(target.n_points for target in targets_to_use)
        if n_points == 0:
            return mp.Frame()
            
        # Set up optimization problem
        opt = nlopt.opt(self.method, 3 * n_points)
        
        # Collect all constraints from targets
        constraints = []
        for target in targets_to_use:
            constraints.append(target.constraint)
            
        obj = ObjectiveFunction(constraints)
        opt.set_min_objective(obj)
        
        # Set bounds (can be customized)
        lower_bounds = np.full(3 * n_points, -10.0)
        upper_bounds = np.full(3 * n_points, 10.0)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        
        # Set tolerances
        opt.set_xtol_rel(1e-4)
        opt.set_maxeval(max_steps)
        
        # Initial guess (random or provided)
        if seed is not None:
            np.random.seed(seed)
        x0 = np.random.uniform(-5, 5, 3 * n_points)
        
        # Optimize
        try:
            result = opt.optimize(x0)
            optimized_points = result.reshape(-1, 3)
            
            # Create result frame by updating target coordinates
            result_frame = mp.Frame()
            all_atoms = []
            
            point_idx = 0
            for target in targets_to_use:
                target_points = target.points
                n_target_points = len(target_points)
                
                # Update coordinates with optimized positions
                new_points = optimized_points[point_idx:point_idx + n_target_points]
                
                # Add atoms to result frame
                for i, point in enumerate(new_points):
                    atom_data = {
                        'id': len(all_atoms),
                        'x': point[0],
                        'y': point[1], 
                        'z': point[2]
                    }
                    all_atoms.append(atom_data)
                    
                point_idx += n_target_points
                
            if all_atoms:
                result_frame["atoms"] = {
                    'id': [atom['id'] for atom in all_atoms],
                    'x': [atom['x'] for atom in all_atoms],
                    'y': [atom['y'] for atom in all_atoms],
                    'z': [atom['z'] for atom in all_atoms]
                }
                
            return result_frame
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return mp.Frame()