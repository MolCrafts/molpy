# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-10
# version: 0.0.1

from molpy.convert import toJAXMD
import numpy as np
import pytest

class TestJAXMD:
    
    def test_toJAXMD(self, C6):
        
        c6 = toJAXMD(C6)
        
        assert np.array_equal(c6['positions'].shape, (12, 3))
        assert len(c6['atomTypes']) == 12
        assert len(c6['elements']) == 12
        assert len(c6['bonds']) == 12
        
    # @pytest.mark.skipif(not find_spec('jax_md'), reason=f'jax-md is not installed')
    # def test_runJAXMD(self, C6):
    #     pass
        # import jax_md
        # from jax import jit, random
        # from jax_md import quantity, minimize, simulate, space, energy
        # c6 = toJAXMD(C6)
        # dimension = 3
        # box_size = quantity.box_size_at_number_density(c6['natoms'], 0.5, dimension)
        # dis_fn, shift_fn = space.periodic(box_size)
        
        
        # R = c6['positions']
        # pair_fn = energy.soft_sphere_pair(dis_fn)
        # bonds = c6['bonds']
        # bond_fn = energy.simple_spring_bond(dis_fn, bonds)
        # def energy_fn(R):
        #     return pair_fn(R)+bond_fn(R)

        # simulation_steps = 10
        # write_every = 100
        # dt = 1e-1
        # temperature = 1e-5

        # init_fn, apply_fn = simulate.brownian(energy_fn, shift_fn, dt, temperature)     
        # apply_fn = jit(apply_fn)  
        
        # key = random.PRNGKey(0) 
        # state = init_fn(key, R)
        # trajectory = []

        # for step in range(simulation_steps):
        #     state = apply_fn(state)
        #     if step % write_every == 0:
        #         trajectory += [state.position]

        # trajectory = np.stack(trajectory)