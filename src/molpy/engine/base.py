import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Sequence
import shutil
from string import Template

if TYPE_CHECKING:
    from typing_extensions import Self

class Script:
    """
    Helper for generating and managing input scripts/files for engines.

    Example:
        >>> s = Script('in.lmp')
        >>> s.write('units real\natom_style full\n')
        >>> s.path.name
        'in.lmp'
        >>> s.read().startswith('units')
        True
        >>> s.append('pair_style lj/cut\n')
        >>> 'pair_style' in s.read()
        True
        >>> str(s) == s.as_posix()
        True
    """

    def __init__(self, path):
        self.path = Path(path)

    def write(self, content):
        self.path.write_text(content)

    def append(self, content):
        with self.path.open("a") as f:
            f.write(content)

    def read(self):
        return self.path.read_text()

    def exists(self):
        return self.path.exists()

    def as_posix(self):
        return self.path.as_posix()

    def as_path(self):
        return self.path

    def __str__(self):
        return self.path.as_posix()

    def __repr__(self):
        return f"<Script path='{self.path}'>"

    def __fspath__(self):
        return str(self.path)

    def __eq__(self, other):
        if isinstance(other, Script):
            return self.path == other.path
        return False

    def __hash__(self):
        return hash(self.path)

    def __enter__(self):
        self._file = self.path.open("r+")
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_file"):
            self._file.close()

    def substitute(self, **kwargs):
        """
        Substitute placeholders in the script content with provided values.
        
        Args:
            **kwargs: Key-value pairs for substitution
        
        Returns:
            Self: The Script instance for method chaining
        """
        content = self.read()
        template = Template(content)
        substituted_content = template.safe_substitute(**kwargs)
        # return copy of the script with substituted content
        new_script = Script(self.path)
        new_script.write(substituted_content)
        return new_script
    
    def save(self, new_path: Path):
        """
        Save the script to a new path.
        
        Args:
            new_path: Path to save the script
        
        Returns:
            Self: The Script instance for method chaining
        """
        new_script = Script(new_path)
        new_script.write(self.read())
        return new_script


class Engine(ABC):
    """
    Abstract base class for computational chemistry engines.
    
    Provides a common interface for running external programs like LAMMPS, CP2K, etc.
    Each engine handles setup, execution, and output processing for its specific program.
    
    Attributes:
        name: Name of the engine
        executable: Path or command to the executable
        work_dir: Working directory for calculations
        input_file: Primary input file for the calculation
        output_file: Primary output file from the calculation
        error_file: Error output file
    
    Example:
        >>> engine = LAMMPSEngine(executable='lmp', work_dir='./calc')
        >>> engine.prepare(input_script='in.lmp')
        >>> result = engine.run()
        >>> print(result.returncode)
        0
    """
    
    def __init__(self, executable: str):
        """
        Initialize the engine.
        
        Args:
            executable: Path or command to the executable
        """
        self.executable = executable
        self.check_executable()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the engine."""
        pass

    def check_executable(self):
        """
        Check if the executable is available in the system PATH.
        
        Raises:
            FileNotFoundError: If the executable is not found
        """
        if not shutil.which(self.executable):
            raise FileNotFoundError(f"Executable '{self.executable}' not found in PATH")
        
    def __repr__(self):
        return f"<{self.__class__.__name__}>"
    
    def prepare(self, work_dir: Path, scripts: Sequence[Script]) -> 'Self':
        """
        Prepare the engine for execution by setting up the working directory and input script.
        
        Args:
            work_dir: Path to the working directory
            scripts: Content of the input script
        
        Returns:
            Self: The engine instance for method chaining
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure scripts are Script instances
        self.scripts = [Script(s) if isinstance(s, Path) else s for s in scripts]
        
        # Write scripts to the working directory
        for script in self.scripts:
            script_path = self.work_dir / script.path.name
            script.save(script_path)
        
        return self
    
    @abstractmethod
    def run(self, **kwargs) -> subprocess.CompletedProcess:
        """
        Execute the engine calculation.
        
        Args:
            **kwargs: Additional arguments for the specific engine
            
        Returns:
            CompletedProcess object with execution results
        """
        pass
    
    def get_script(self, name: str) -> Optional[Script]:
        """
        Get a script by name.
        
        Args:
            name: Name of the script file
            
        Returns:
            Script object or None if not found
        """
        if not hasattr(self, 'scripts'):
            return None
            
        for script in self.scripts:
            if script.path.name == name:
                return script
        return None
    
    def clean(self, keep_scripts: bool = True):
        """
        Clean up calculation files.
        
        Args:
            keep_scripts: Whether to keep input scripts
        """
        if not hasattr(self, 'work_dir') or not self.work_dir.exists():
            return
            
        if not keep_scripts and hasattr(self, 'scripts'):
            for script in self.scripts:
                script_path = self.work_dir / script.path.name
                if script_path.exists():
                    script_path.unlink()
    
    def list_output_files(self) -> List[Path]:
        """
        List all output files in the working directory.
        
        Returns:
            List of output file paths
        """
        if not hasattr(self, 'work_dir') or not self.work_dir.exists():
            return []
            
        return [f for f in self.work_dir.iterdir() if f.is_file()]