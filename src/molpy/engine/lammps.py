from typing import List
import subprocess

from .base import Engine


class LAMMPSEngine(Engine):
    """
    LAMMPS molecular dynamics engine.
    
    Example:
        >>> engine = LAMMPSEngine('lmp')
        >>> script = Script('in.lmp')
        >>> script.write('units real\\natom_style full\\n')  
        >>> engine.prepare('./calc', [script])
        >>> result = engine.run()
        >>> print(result.returncode)
        0
    """
    
    @property
    def name(self) -> str:
        """Return engine name."""
        return 'LAMMPS'
    
    def run(self, input_script: str = 'in.lmp', output_file: str = 'out.lmp',
            **kwargs) -> subprocess.CompletedProcess:
        """
        Execute LAMMPS calculation.
        
        Args:
            input_script: Name of the input script file
            output_file: Name of the output file
            **kwargs: Additional arguments passed to subprocess.run
            
        Returns:
            CompletedProcess object with execution results
        """
        if not hasattr(self, 'work_dir'):
            raise RuntimeError("Engine not prepared. Call prepare() first.")
        
        command = [self.executable, '-in', input_script, '-log', output_file]
        
        # Default subprocess arguments
        run_kwargs = {
            'cwd': self.work_dir,
            'capture_output': True,
            'text': True,
            'check': False
        }
        run_kwargs.update(kwargs)
        
        return subprocess.run(command, **run_kwargs)