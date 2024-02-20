import logging
import molpy as mp

class BaseCalculator:
    
    def __init__(self):

        # self.logger = logging.getLogger(__name__)
        pass

    def dump_to(self, filename: str):
        """
        Dump the calculator to a file.

        Args:
            filename (str): File to dump the calculator to.
        """
        self.traj_saver = mp.io.TrajectorySaver(filename)

    def dump(self, frame: mp.Frame):
        """
        Dump the frame to the trajectory.

        Args:
            frame (schnetpack.md.Frame): Frame to be dumped.
        """
        self.traj_saver.dump(frame)