'''
Directories handler. Reading, writing, creating directories.
file    : directories.py
author  : Maksymilian Kliczkowski
email   : maksymilian.kliczkowski@pwr.edu.pl
'''

import os
import random
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union, Iterator, Any, Dict

# Type alias for path-like objects
PathLike    = Union[str, Path]
kPS         = os.sep

#######################################################################################################################

class Directories(object):
    """ 
    Class representing a directory handler
    - static methods are represented with camel case
    - class methods are represented with underscore
    """
    
    def __init__(self, *parts: PathLike) -> None:
        """
        Initialize with one or more path components.
        >>> d = Directories("foo", "bar")  # -> Path("foo/bar")
        """
        self.path = Path(*parts)

    def __fspath__(self) -> str:
        return str(self.path)
    
    def __len__(self) -> int | str:
        """
        Return the number of items in the directory if it is a directory,
        """
        if self.path.is_dir():
            return self.size_human()
        return self.path.stat().st_size

    #! Operators

    def __add__(self, other: PathLike) -> "Directories":
        """
        Concatenate with another path component.
        >>> d = Directories("foo") + "bar"  # -> Path("foo/bar")
        """
        return self.join(other)

    def __iadd__(self, other: PathLike) -> "Directories":
        """
        In-place concatenation with another path component.
        >>> d = Directories("foo"); d += "bar"  # -> Path("foo/bar")
        """
        self.path = self.path.joinpath(other)
        return self
    
    def __radd__(self, other: PathLike) -> "Directories":
        """
        Concatenate with another path component.
        >>> d = "foo" + Directories("bar")  # -> Path("foo/bar")
        """
        return self.join(other)
    
    def __truediv__(self, other: PathLike) -> "Directories":
        """
        Concatenate with another path component using / operator.
        >>> d = Directories("foo") / "bar"  # -> Path("foo/bar")
        """
        return self.join(other)
    
    def __rtruediv__(self, other: PathLike) -> "Directories":
        """
        Concatenate with another path component using / operator.
        >>> d = "foo" / Directories("bar")  # -> Path("foo/bar")
        """
        return self.join(other)
    
    #! Comparison
    
    def __eq__(self, other: PathLike) -> bool:
        """
        Check equality with another path component.
        >>> d = Directories("foo") == "foo"  # -> True
        """
        if isinstance(other, str):
            return str(self) == other
        elif isinstance(other, Path):
            return self.path == other
        else:
            return False
        
    def __ne__(self, other: PathLike) -> bool:
        """
        Check inequality with another path component.
        >>> d = Directories("foo") != "bar"  # -> True
        """
        return not self.__eq__(other)
    
    #! Hashing
    
    def __hash__(self) -> int:
        """
        Hash the path for use in sets or dictionaries.
        >>> d = Directories("foo")  # -> hash(Path("foo"))
        """
        return hash(self.path)
    
    #! String representation    
    
    def __repr__(self) -> str:
        """
        Return a string representation of the path.
        >>> d = Directories("foo")  # -> "Directories('foo')"
        """
        return f"Directories({self.path!r})"
    
    def __str__(self) -> str:
        """
        Return a string representation of the path.
        >>> d = Directories("foo")  # -> "foo"
        """
        return str(self.path)
    
    ################################################################################
    #! Some standard filters
    ################################################################################
    
    @staticmethod
    def f_h5(p: List[Path]) -> List[str]:
        """Filter for .h5 files."""
        return [str(x) for x in p if str(x).endswith('.h5')]
    
    @staticmethod
    def f_csv(p: List[Path]) -> List[str]:
        """Filter for .csv files."""
        return [str(x) for x in p if str(x).endswith('.csv')]
    
    @staticmethod
    def f_nonempty(p: List[Path]) -> List[str]:
        """Filter for non-empty files."""
        return [str(x) for x in p if x.stat().st_size > 0]
    
    @staticmethod
    def f_contains(substr: str) -> Callable[[Path], bool]:
        """Return a filter that checks if the filename contains a substring."""
        def _filter(p: List[Path]) -> List[str]:
            return [str(x) for x in p if substr in str(x)]
        return _filter
    
    ################################################################################
    #! Construction / Navigation
    ################################################################################

    def join(self, *parts: PathLike, create: bool = False) -> "Directories":
        """
        Return a new Directories for self/path joined with parts.
        If create=True, mkdir(parents=True, exist_ok=True) is called.
        """
        new_path = self.path.joinpath(*parts)
        if create:
            new_path.mkdir(parents=True, exist_ok=True)
        return Directories(new_path)

    def parent(self) -> "Directories":
        """
        Return Directories for parent directory (..).
        """
        return Directories(self.path.parent)

    @classmethod
    def win(cls, raw: str) -> "Directories":
        """
        Parse a Windows-style backslash path into Directories.
        """
        return cls(*raw.split("\\"))
    
    def format(self, *args, **kwargs) -> "Directories":
        """
        Format the path using str.format() and return a new Directories.
        >>> d = Directories("foo").format("bar")  # -> Path("foo/bar")
        """
        formatted_path = self.path.as_posix().format(*args, **kwargs)
        return Directories(formatted_path)
    
    def resolve(self) -> "Directories":
        """
        Return a new Directories with the absolute resolved path.
        """
        return Directories(self.path.resolve())
    
    def endswith(self, suffix: str) -> bool:
        """
        Check if the path ends with the given suffix.
        """
        return str(self.path).endswith(suffix)
    
    ################################################################################
    #! Creation
    ################################################################################

    def mkdir(self, parents: bool = True, exist_ok: bool = True) -> "Directories":
        """
        Create this directory on disk.
        Returns self for chaining.
        """
        self.path.mkdir(parents=parents, exist_ok=exist_ok)
        return self

    @staticmethod
    def mkdirs(paths    : Iterable[PathLike], 
            parents     : bool = True, 
            exist_ok    : bool = True) -> None:
        """
        Create multiple directories.
        """
        for p in paths:
            Path(p).mkdir(parents=parents, exist_ok=exist_ok)
            
    ################################################################################
    #! Listing & Clearing
    ################################################################################

    def list_files(self,
                   *,
                include_empty   :   bool                            = True,
                filters         :   List[Callable[[Path], bool]]    = None,
                sort_key        :   Optional[Callable[[Path], any]] = None) -> List[Path]:
        """
        List files (not directories) in this directory.
        - include_empty : if False, skip files of size zero.
        - filters       : a list of callables Path->bool; all must pass.
        - sort_key      : key function for sorting.
        """
        try:
            files = [p for p in self.path.iterdir() if p.is_file()]
        except FileNotFoundError:
            return []
        except PermissionError:
            print(f"PermissionError: {self.path}")
            return []
        except OSError as e:
            print(f"OSError: {self.path} - {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {self.path} - {e}")
            return []
        
        if not include_empty:
            files = [p for p in files if p.stat().st_size > 0]
        if filters is None:
            filters = []
        elif not isinstance(filters, list):
            filters = [filters]
        
        for f in filters:
            try:
                files = list(filter(f, files))
            except Exception as e:
                print(f"Error applying filter {f.__name__}: {e}")
                continue
            
        if sort_key:
            files.sort(key=sort_key)

        return files

    def list_dirs(self,
                *,
                include_empty: bool = True,
                relative: bool = False,
                as_string: bool = False,
                filters: List[Callable[[Path], bool]] = [],
                sort_key: Optional[Callable[[Path], Any]] = None) -> List[Path]:
        """
        List directories in this directory.
        - include_empty : if False, skip empty directories.
        - filters       : a list of callables Path->bool; all must pass.
        - sort_key      : key function for sorting.
        """
        try:
            dirs = [p for p in self.path.iterdir() if p.is_dir()]
        except FileNotFoundError:
            return []
        except PermissionError:
            print(f"PermissionError: {self.path}")
            return []
        except OSError as e:
            print(f"OSError: {self.path} - {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {self.path} - {e}")
            return []

        if not include_empty:
            dirs = [p for p in dirs if not any(p.iterdir())]

        for f in filters:
            try:
                dirs = list(filter(f, dirs))
            except Exception as e:
                print(f"Error applying filter {f.__name__}: {e}")
                continue

        if sort_key:
            dirs.sort(key=sort_key)

        if relative:
            dirs = [p.relative_to(self.path) for p in dirs]
        
        if as_string:
            dirs = [str(p) for p in dirs]
        return dirs

    def clear_empty(self) -> List[Path]:
        """
        Remove all zero-length files in this directory.
        Returns list of files left after removal.
        """
        
        survivors: List[Path] = []
        for p in self.path.iterdir():
            if p.is_file() and p.stat().st_size == 0:
                p.unlink()
            else:
                survivors.append(p)
        return survivors

    def walk(self) -> Iterator[Path]:
        """
        Walk the directory tree and yield all files.
        """
        yield from self.path.rglob('*')

    def glob(self, pattern: str) -> List[Path]:
        """
        Return a list of all files matching the pattern in this directory.
        """
        return list(self.path.glob(pattern))
    
    ################################################################################
    #! Random file
    ################################################################################

    def random_file(self, condition: Callable[[Path], bool] = lambda _: True) -> Path:
        """
        Return a random Path in this directory satisfying condition.
        Raises ValueError if none match.
        """
        candidates = [p for p in self.path.iterdir() if p.is_file() and condition(p)]
        if not candidates:
            raise ValueError(f"No file satisfying condition in {self.path}")
        return random.choice(candidates)

    ################################################################################
    #! Transfer
    ################################################################################

    def copy_files(self,
                    dest        : PathLike,
                    condition   : Callable[[Path], bool],
                    overwrite   : bool = False) -> None:
        """
        Copy all files satisfying condition() from self to dest.
        Creates dest if needed.
        
        Parameters
        ----------
        dest : PathLike
            Destination directory.
        condition : Callable[[Path], bool]
            Function that takes a Path and returns True if the file should be copied.
        overwrite : bool, optional
            If True, overwrite existing files in the destination directory. Default is False.
        """
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        for p in self.path.iterdir():
            if p.is_file() and condition(p):
                target = dest_path / p.name
                if not overwrite and target.exists():
                    continue
                shutil.copy2(p, target)

    def transfer_files(self,
                    dest        : PathLike,
                    condition   : Callable[[Path], bool]) -> None:
        """
        Move all files satisfying condition() from self to dest.
        Creates dest if needed.
        """
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)

        for p in self.path.iterdir():
            if p.is_file() and condition(p):
                target = dest_path / p.name
                p.rename(target)

    ################################################################################
    #! Convenience
    ################################################################################

    def exists(self) -> bool:
        """
        Check if the path exists.
        """
        return self.path.exists()

    def as_path(self) -> Path:
        '''
        Return the path as a Path object.
        '''
        return self.path
    
    def is_empty(self) -> bool:
        """
        Check if the directory is empty.
        """
        return not any(self.path.iterdir())

    def is_dir(self) -> bool:
        """
        Check if the path is a directory.
        """
        return self.path.is_dir()
    
    def is_file(self) -> bool:
        """
        Check if the path is a file.
        """
        return self.path.is_file()
    
    def is_symlink(self) -> bool:
        """
        Check if the path is a symlink.
        """
        return self.path.is_symlink()
    
    def size(self) -> int:
        """
        Return the size of the directory in bytes.
        """
        return sum(f.stat().st_size for f in self.path.glob('*') if f.is_file())
    
    def size_human(self) -> str:
        """
        Return the size of the directory in a human-readable format.
        """
        size = self.size()
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"
    
    def disk_usage(self) -> str:
        """
        Return the disk usage of the directory in a human-readable format.
        """
        total, used, free = shutil.disk_usage(self.path)
        return f"Total: {total // (2**30)} GB, Used: {used // (2**30)} GB, Free: {free // (2**30)} GB"
    
    def checksum(self) -> str:
        """
        Return the checksum of the directory.
        """
        import hashlib
        hash_md5 = hashlib.md5()
        for f in self.path.glob('*'):
            if f.is_file():
                with open(f, "rb") as file:
                    for chunk in iter(lambda: file.read(4096), b""):
                        hash_md5.update(chunk)
        return hash_md5.hexdigest()

################################################################################

class DirectoriesData:
    """
    Collects directories across multiple machines and stores them in dictionaries.
    Only directories that exist are included in `existing`.
    Example:
    >>> dirs = DirectoriesData(
    >>>     klimak_um_only_f=("/media/.../klimak_um_only_f_t100000/uniform", "503"),
    >>>     klimak_all=("/media/.../klimak_um_plrb_all_t100000/uniform", "503"),
    >>>     locally=("data_project/uniform", "local")
    >>> )
    """

    def __init__(self, **dirs: str):
        """
        Initialize with named directory paths.
        Each value can be either a string (path) or a tuple (path, machine).
        """
        
        self.all: Dict[str, Directories] = {}
        for name, spec in dirs.items():
            if isinstance(spec, tuple):
                path, machine = spec
            else:
                path, machine = spec, "default"
            self.all[name] = (Directories(path), machine)

        #! track existing directories
        self.existing: Dict[str, Directories] = {
            name: d for name, (d, _) in self.all.items() if os.path.exists(d)
        }

        #! track machines
        self.machines: Dict[str, List[str]] = {}
        for name, (d, machine) in self.all.items():
            self.machines.setdefault(machine, []).append(name)

    ############################################################

    def get(self, name: str, only_existing: bool = True) -> Optional[Directories]:
        """
        Get a directory by name. Optionally restrict to existing ones.
        
        Parameters
        ----------
        name : str
            The name of the directory to retrieve.
        only_existing : bool, optional
            If True, only return the directory if it exists. Default is True.
        """
        
        if only_existing:
            return self.existing.get(name)
        return self.all.get(name)

    ############################################################

    def add(self, name: str, path: PathLike, machine: str = "default"):
        """Add a new directory."""
        self.all[name] = Directories(path, machine)
        if self.all[name].exists():
            self.existing[name] = self.all[name]
        self.machines.setdefault(machine, []).append(name)
    
    def remove(self, name: str) -> None:
        """Remove a directory entry by name."""
        if name in self.all:
            machine = self.all[name].machine
            del self.all[name]
            self.existing.pop(name, None)
            if machine in self.machines and name in self.machines[machine]:
                self.machines[machine].remove(name)
                if not self.machines[machine]:
                    del self.machines[machine]
    
    ############################################################

    def _match(self, name: str, filters: list[Union[str, Callable[[str], bool]]]) -> bool:
        """Check if a name matches any filter."""
        for f in filters:
            if isinstance(f, str):
                if f in name:
                    return True
            elif callable(f):
                if f(name):
                    return True
        return False
    
    def filter_names(self, filters: list[Union[str, Callable[[str], bool]]], only_existing: bool = True) -> list[str]:
        """
        Return names that match any filter.
        Filters can be substrings or callables (e.g. regex matchers, lambdas).
        """
        source = self.existing if only_existing else self.all
        return [name for name in source.keys() if self._match(name, filters)]

    def filter_dirs(self, filters: list[Union[str, Callable[[str], bool]]], only_existing: bool = True) -> dict[str, Directories]:
        """
        Return {name: Directories} for names matching any filter.
        Filters can be substrings or callables (e.g. regex matchers, lambdas).
        """
        source = self.existing if only_existing else self.all
        return {name: d for name, d in source.items() if self._match(name, filters)}
    
    ############################################################

    def list_existing(self) -> List[str]:
        """List names of existing directories."""
        return list(self.existing.keys())
    
    def list_existing_dirs(self) -> List[Directories]:
        """List existing Directories objects."""
        return list(self.existing.values())

    def list_all(self) -> List[str]:
        """List all directory names provided."""
        return list(self.all.keys())

    def list_all_dirs(self) -> List[Directories]:
        """List all Directories objects provided."""
        return list(self.all.values())

    def list_machines(self) -> List[str]:
        """List all machines."""
        return list(self.machines.keys())

    ############################################################

    def on(self, machine: str, only_existing: bool = True) -> Dict[str, Directories]:
        """
        Get directories for a specific machine.
        Parameters
        ----------
        machine : str
            The machine name to filter directories.
        only_existing : bool, optional
            If True, only return existing directories. Default is True.
        Returns
        -------
        Dict[str, Directories]
            A dictionary of directory names to Directories objects.
        Raises
        -------
        KeyError
            If the machine is not known.
        """
        names = self.machines.get(machine, [])
        if only_existing:
            return {n: self.all[n] for n in names if n in self.existing}
        return {n: self.all[n] for n in names}

    ############################################################

    def register_machine(self, machine: str):
        """Ensure machine is known (for clarity, optional)."""
        self.machines.setdefault(machine, [])

    ############################################################

    def __repr__(self):
        return (
            f"DirectoriesData(\n"
            f"  machines={list(self.machines.keys())},\n"
            f"  all={list(self.all.keys())},\n"
            f"  existing={list(self.existing.keys())}\n"
            f")"
        )
        
    def __str__(self):
        lines = ["DirectoriesData:"]
        for name, d in self.all.items():
            status = "exists" if d.exists() else "missing"
            lines.append(f"  {name}: {d} [{status}]")
        return "\n".join(lines)
    
    def __len__(self):
        return len(self.all)
    
    def __getitem__(self, name: str) -> Directories:
        return self.all[name]
    
    def __contains__(self, name: str) -> bool:
        return name in self.all
    
    def __iter__(self):
        return iter(self.all.items())

    def __add__(self, other: "DirectoriesData") -> "DirectoriesData":
        """Return a new DirectoriesData with merged contents."""
        new = DirectoriesData(**{})  # empty
        # copy self
        for name, d in self.all.items():
            new.add(name, str(d.path), d.machine)
        # add from other
        for name, d in other.all.items():
            new.add(name, str(d.path), d.machine)
        return new

    def __iadd__(self, other: "DirectoriesData") -> "DirectoriesData":
        """In-place merge of other into self."""
        for name, d in other.all.items():
            self.add(name, str(d.path), d.machine)
        return self

    def __radd__(self, other: "DirectoriesData") -> "DirectoriesData":
        """Allow sum([...]) to work by reusing __add__."""
        return self.__add__(other)

################################################################################
#! EOF