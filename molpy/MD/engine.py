# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-10-05
# version: 0.0.1

class MDEngine:

    def __init__(self, integrator):

        self.integrator = integrator

    def run(self, timesteps:int):
        """
        loop over N timesteps:
        """
        for i in range(timesteps):
            self._step()

    def _step(self):
        """
        run one timestep. The order of execution and name of method is following LAMMPS:

        if timeout condition: break
        
        ev_set()

        fix->initial_integrate()
        fix->post_integrate()

        nflag = neighbor->decide()
        if nflag:
            fix->pre_exchange()
            domain->pbc()
            domain->reset_box()
            comm->setup()
            neighbor->setup_bins()
            comm->exchange()
            comm->borders()
            fix->pre_neighbor()
            neighbor->build()
            fix->post_neighbor()
        else:
            comm->forward_comm()

        force_clear()
        fix->pre_force()

        pair->compute()
        bond->compute()
        angle->compute()
        dihedral->compute()
        improper->compute()
        kspace->compute()

        fix->pre_reverse()
        comm->reverse_comm()

        fix->post_force()
        fix->final_integrate()
        fix->end_of_step()

        if any output on this step:
            output->write()        

        """
        pass
