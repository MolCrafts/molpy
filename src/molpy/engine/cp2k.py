from typing import List
import subprocess

from .base import Engine


class CP2KEngine(Engine):
    """
    CP2K quantum chemistry engine.
    
    Example:
        >>> engine = CP2KEngine('cp2k.psmp')
        >>> script = Script('cp2k.inp')
        >>> script.write('&GLOBAL\\n  PROJECT water\\n&END GLOBAL\\n')
        >>> engine.prepare('./calc', [script])
        >>> result = engine.run()
        >>> print(result.returncode)
        0
    """
    
    @property
    def name(self) -> str:
        """Return engine name."""
        return 'CP2K'
    
    def run(self, input_file: str = 'cp2k.inp', output_file: str = 'cp2k.out',
            **kwargs) -> subprocess.CompletedProcess:
        """
        Execute CP2K calculation.
        
        Args:
            input_file: Name of the input file
            output_file: Name of the output file
            **kwargs: Additional arguments passed to subprocess.run
            
        Returns:
            CompletedProcess object with execution results
        """
        if not hasattr(self, 'work_dir'):
            raise RuntimeError("Engine not prepared. Call prepare() first.")
        
        command = [self.executable, '-i', input_file, '-o', output_file]
        
        # Default subprocess arguments
        run_kwargs = {
            'cwd': self.work_dir,
            'capture_output': True,
            'text': True,
            'check': False
        }
        run_kwargs.update(kwargs)
        
        return subprocess.run(command, **run_kwargs)