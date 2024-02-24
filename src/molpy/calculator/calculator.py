import logging
import molpy as mp

class BaseCalculator:
    
    def __init__(self, report_config: dict = None, dump_config: dict = None):

        # self.logger = logging.getLogger(__name__)
        self.report_rate = report_config.get('rate')
        if dump_config:
            self.dump_rate = dump_config.get('rate')
            self.dump_path = dump_config.get('path')
            self.dump_to(self.dump_path)

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