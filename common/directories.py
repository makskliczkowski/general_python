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
from typing import Callable, Iterable, List, Optional, Union, Iterator
PathLike = Union[str, Path]

from general_python.common.flog import printV

kPS = os.sep

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
    
    def __len__(self) -> int:
        """
        """
        if self.path.is_dir():
            return self.size_human()
        return len(self.path)

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
                filters         :   List[Callable[[Path], bool]]    = [],
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

        for f in filters:
            try:
                files = list(filter(f, files))
            except Exception as e:
                print(f"Error applying filter {f.__name__}: {e}")
                continue

        if sort_key:
            files.sort(key=sort_key)

        return files

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
#! EOF