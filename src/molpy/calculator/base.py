import logging
import molpy as mp

class Calculator:
    
    def __init__(self, report_config: dict = None, dump_config: dict = None):

        self.logger = logging.getLogger(__name__)
        self.report_rate = report_config.get('rate')
        if dump_config:
            self.dump_rate = dump_config.get('rate')
            dump_path = dump_config.get('path')
            format = dump_config.get('format')
            self.dump_to(dump_path, format)

    def dump_to(self, filename: str, format: str):
        """
        Dump the calculator to a file.

        Args:
            filename (str): File to dump the calculator to.
        """
        self.traj_saver = mp.io.TrajectorySaver(filename, format)

    def dump(self, step:int, frame: mp.Frame):
        """
        Dump the frame to the trajectory.

        Args:
            frame (schnetpack.md.Frame): Frame to be dumped.
        """
        if self.dump_rate and step % self.dump_rate == 0:
            self.traj_saver.write(frame)

    def log(self, step:int, frame:mp.Frame):
        """
        Log the current state of the simulation.

        Args:
            step (int): Current simulation step.
            frame (schnetpack.md.Frame): Frame to be logged.
        """
        if self.report_rate and step % self.report_rate == 0:
            self.logger.info(f"Step {step}: {frame.energy}")