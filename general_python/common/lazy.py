"""
Lazy import utilities.
"""
import importlib

class LazyImporter:
    """
    Helper class to manage lazy imports.
    """
    def __init__(self, module_name, lazy_imports, lazy_cache=None):
        self.module_name = module_name
        self._lazy_imports = lazy_imports
        self._lazy_cache = lazy_cache if lazy_cache is not None else {}

    def lazy_import(self, name: str):
        """
        Lazily import a module or attribute based on configuration.
        """
        if name in self._lazy_cache:
            return self._lazy_cache[name]

        if name not in self._lazy_imports:
            raise AttributeError(f"module {self.module_name!r} has no attribute {name!r}")

        module_path, attr_name = self._lazy_imports[name]

        # Handle module-local classes (if applicable, though typically handled before calling this)
        if module_path is None:
             # This might need adjustment depending on how local classes are handled in specific modules
             # but generally lazy imports are for external/submodule things.
            raise AttributeError(f"Local class {name!r} must be accessed after class definition")

        try:
            # Import the module
            # We use the package of the caller if relative import
            package = self.module_name
            module = importlib.import_module(module_path, package=package)

            # If attr_name is None, we want the module itself
            if attr_name is None:
                result = module
            else:
                result = getattr(module, attr_name)

            self._lazy_cache[name] = result
            return result
        except ImportError as e:
            raise ImportError(f"Failed to import lazy module '{name}' from '{module_path}': {e}") from e

    def __getattr__(self, name: str):
        return self.lazy_import(name)

    def __dir__(self):
        return sorted(list(self._lazy_imports.keys()))
