'''
This is a module for solving ordinary differential equations (ODEs)
It provides a set of classes and methods to define and solve initial value problems (IVPs)

-----------------------------------------------
File    : general_python/common/ode.py
Author  : Maksymilian Kliczkowski
email   : maxgrom97@gmail.com
-----------------------------------------------
'''

import  time
import  numpy as np
import  warnings
import  inspect
from    typing import Union, Any, Tuple, Callable
from    abc import ABC, abstractmethod

try:
    from scipy.integrate import solve_ivp
except ImportError as e:
    raise ImportError("Failed to import scipy.integrate module. Ensure general_python package is correctly installed.") from e

try:
    import          jax
    import          jax.numpy as jnp
    from            jax import jit
    JAX_AVAILABLE   = True
except ImportError:
    jnp             = None
    jit             = None
    jax             = None
    JAX_AVAILABLE   = False
    
########################################################################
#! General class for ODE integration
########################################################################

class IVP(ABC):
    r"""
    Abstract initial value problem solver interface.

    Methods
    -------
    step(f, t, y, **rhs_args)
        Compute one integration step without modifying internal state.
    update(y, h, f, t, **rhs_args)
        Update and return new state given current y and step size h.
    dt(h, i)
        Return the time step used (may depend on h or step index i).

    Attributes
    ----------
    xp
        Array module (numpy or jax.numpy) selected by backend.
    """
    
    def __init__(self, backend: str = 'numpy', rhs_prefactor: float = 1.0, dt: float = 1e-3):
        """
        Initialize the ODE solver with a specified backend.

        Parameters
        ----------
        backend : str
            Backend to use for numerical operations ('numpy' or 'jax').
        rhs_prefactor : float
            Prefactor for the right-hand side of the ODE.
        """
        
        try:
            from ..algebra.utils import get_backend
        except ImportError:
            def get_backend(backend_str: str): return np if isinstance(backend_str, str) and backend_str.lower() == 'numpy' else jnp if JAX_AVAILABLE and backend_str.lower() == 'jax' else np
        
        self.backend        = backend
        self.xp             = get_backend(backend)
        self.backendstr     = 'numpy' if self.xp is np else 'jax'
        
        if self.xp is None:
            raise ValueError(f"Backend '{backend}' is not supported. Choose 'numpy' or 'jax'.")
        
        self._isjax         = not (self.xp is np)
        self._isnpy         = not self._isjax
        self._dt            = dt
        self._rhs_prefactor = rhs_prefactor
    
    def _call_rhs(self, f, t: float, y, int_step: int = 0, **rhs_args):
        """
        Call the user-provided RHS function `f`, handling different signatures.

        f may accept:
            - positional or keyword args for state (y or y0), time t, int_step
            - additional rhs_args
        It may return:
            - (dy, info, other)
            - dy only (scalar or array)
        """
        if self._isjax:
            #! assume f is a jax function with signature f(y, t, **rhs_args, int_step=int_step)
            out     = f(y, t, **rhs_args, int_step=int_step)
        else:
            sig     = inspect.signature(f)
            kwargs  = {}
            # bind state
            if 'y' in sig.parameters:
                kwargs['y']     = y
            elif 'y0' in sig.parameters:
                kwargs['y0']    = y
            else:
                pass
            
            #! bind time
            if 't' in sig.parameters:
                kwargs['t']         = t
            #! bind int_step
            if 'int_step' in sig.parameters:
                kwargs['int_step']  = int_step
            #! bind additional args
            for name, val in rhs_args.items():
                if name in sig.parameters:
                    kwargs[name] = val
            out = f(**kwargs)
            
        #! normalize outputs
        if isinstance(out, tuple):
            if len(out) == 3:
                return out # (dy, info, other)
            elif len(out) == 2:
                dy, info = out
                return dy, info, None
            else:
                dy = out[0]
                return dy, None, None
        else:
            return out, None, None
    
    # -------------------------------------------------------
    
    def dt(self, h: float = 0.0, i: int = 0) -> float:
        return self._dt
    
    def set_dt(self, dt: float):
        """
        Set the time step for the integration.

        Parameters
        ----------
        dt : float
            The new time step to set.
        """
        self._dt = dt
    
    # -------------------------------------------------------
    
    @abstractmethod
    def step(self, f, t: float, y, **rhs_args):
        '''
        Compute one integration step without modifying internal state.
        This method should be implemented by subclasses.
        '''
        raise NotImplementedError

    def update(self, y, h: float, f, t: float, **rhs_args):
        # Default: call step then return state
        yout, _ = self.step(f, t, y, **rhs_args)
        return yout

    @property
    def order(self) -> int:
        '''
        Return the order of the integration method.
        This method should be implemented by subclasses.
        '''
        return 1

    @property
    def is_jax(self) -> bool:
        """
        Check if the backend is JAX.

        Returns
        -------
        bool
            True if the backend is JAX, False otherwise.
        """
        return self._isjax
    
    @property
    def is_numpy(self) -> bool:
        """
        Check if the backend is NumPy.

        Returns
        -------
        bool
            True if the backend is NumPy, False otherwise.
        """
        return self._isnpy
    
    def __repr__(self):
        """
        Return a string representation of the IVP object.
        """
        return f"SimpleIVP(backend={self.backendstr})"
    
    def __str__(self):
        """
        Return a string representation of the IVP object.
        """
        return self.__repr__()
    
    def __call__(self, f, t, y, **rhs_args):
        """
        Call the step method to compute one integration step.

        Parameters
        ----------
        f : callable
            Function representing the right-hand side of the ODE.
        t : float
            Current time.
        y : array-like
            Current state.
        **rhs_args : keyword arguments
            Additional arguments to pass to the function f.

        Returns
        -------
        yout : array-like
            New state after one integration step.
        dt : float
            The step size used.
        """
        return self.step(f, t, y, **rhs_args)
    
    def __len__(self):
        """
        Return the length of the IVP object.
        """
        return self.order

#######################################################################
#! Euler integration
#######################################################################

class Euler(IVP):
    r"""
    Simple forward Euler integrator.

    Parameters
    ----------
    dt : float
        Fixed step size for the integration.
    backend : str
        'numpy' or 'jax'
    """
    
    def __init__(self, dt: float = 1e-3, backend: str = 'numpy', rhs_prefactor: float = 1.0):
        """
        Initializes the object with a specified time step and computational backend.

        Args:
            dt (float, optional):
                The time step to use for the ODE solver. Defaults to 1e-3.
            backend (str, optional):
                The computational backend to use (e.g., 'numpy'). Defaults to 'numpy'.
            rhs_prefactor (float, optional):
                A prefactor to multiply with the right-hand side of the ODE. Defaults to 1.0.
        """
        super().__init__(backend, rhs_prefactor=rhs_prefactor, dt=dt)

    def step(self, f, t: float, y, **rhs_args):
        r"""
        Compute one Euler step: y_{n+1} = y_n + \Delta t * f(y_n, t).

        Returns
        -------
        yout
            New state after one Euler step.
        dt
            The step size used.
        """
        #! Evaluate derivative through function
        dy, step_info, other    = self._call_rhs(f, t, y, int_step=0, **rhs_args)
        yout                    = y + (self._dt * self._rhs_prefactor) * dy
        return yout, self._dt, (step_info, other)

    def __repr__(self):
        """
        Return a string representation of the Euler object.
        """
        return f"Euler(dt={self._dt}, backend={self.backendstr}, rhs_p={self._rhs_prefactor})"

########################################################################
#! Heun integration
########################################################################

class Heun(IVP):
    r"""
    Second-order Heun (explicit trapezoidal) integrator.

    Parameters
    ----------
    dt : float
        Fixed step size delta t (can be adapted externally).
    backend : str
        'numpy' or 'jax'
    """
    
    def __init__(self, dt: float = 1e-3, backend: str = 'numpy', rhs_prefactor: float = 1.0):
        super().__init__(backend, dt=dt, rhs_prefactor=rhs_prefactor)

    def step(self, f, t: float, y, **rhs_args):
        r"""
        Compute one Heun step:
        .. math::
            y_{n+1} = y_n + \frac{\Delta t}{2} \left( f(y_n, t) + f(y_n + \Delta t f(y_n, t), t + \Delta t) \right)
        where :math:`\Delta t` is the time step.
        This is a second-order Runge-Kutta method.
        
        Parameters
        ----------
        f : callable
            Function representing the right-hand side of the ODE.
        t : float
            Current time.
        y : array-like
            Current state.

        >>> k0      = f(y, t)                   # slope at t
        >>> y_pred  = y + dt * k0               # predictor step   
        >>> k1      = f(y_pred, t + dt)         # slope at t + dt
        >>> yout    = y + (dt / 2) * (k0 + k1)  # corrector step

        Returns
        -------
            yout and dt.
        """
        dt                      = self._dt
        multiplier              = self._rhs_prefactor * dt
        # Predictor slope
        k0, step_info, other    = self._call_rhs(f, t, y, int_step=0, **rhs_args)
        # Predictor step
        y_pred                  = y + multiplier * k0
        # Corrector slope
        k1, step_info, other    = self._call_rhs(f, t + dt, y_pred, int_step=1, **rhs_args)
        # Combine as average
        yout                    = y + 0.5 * multiplier * (k0 + k1)
        return yout, dt, (step_info, other)

    def __repr__(self):
        """
        Return a string representation of the Heun object.
        """
        return f"Heun(dt={self._dt}, backend={self.backendstr}, rhs_p={self._rhs_prefactor})"

########################################################################
#! Adaptive Heun integration
########################################################################

class AdaptiveHeun(IVP):
    """
    Adaptive second-order Heun integrator with error control.

    Parameters
    ----------
    dt : float
        Initial time step delta t.
    tol : float
        Error tolerance.
    max_step : float
        Maximum allowed time step.
    backend : str
        'numpy' or 'jax'
    """
    def __init__(self,
                dt              : float = 1e-3,
                tol             : float = 1e-8,
                max_step        : float = 1.0,
                backend         : str   = 'numpy',
                rhs_prefactor   : float = 1.0):
        super().__init__(backend, dt=dt, rhs_prefactor=rhs_prefactor)
        self.tolerance  = tol
        self.max_step   = max_step

    def step(self, f, t: float, y, norm_fun=None, **rhs_args):
        if norm_fun is None:
            norm_fun = self.xp.linalg.norm

        dt  = self._dt
        y0  = y
        fe  = 0.0
        mult= self._rhs_prefactor * dt
        
        #! Adapt until accepted
        while fe < 1.0:
            # Full step - 1st order (just the simple Heun step)
            k0, step_info, other = f(y0=y0, t=t, **rhs_args, int_step=0)
            y_full = y0 + dt * k0
            k1, step_info, other = f(y0=y_full, t=t + dt, **rhs_args, int_step=1)
            dy_full = 0.5 * mult * (k0 + k1)

            # Two half steps - 2nd order (search for a better step)
            k0h     = k0
            y_half  = y0 + 0.5 * mult * k0h
            k1h, step_info, other = f(y0=y_half, t=t + 0.5 * dt, **rhs_args, int_step=2)
            dy_half = 0.5 * mult * k1h
            y_half2 = y_half + 0.5 * mult * k1h
            k2h, step_info, other = f(y0=y_half2, t=t + dt, **rhs_args, int_step=3)
            dy_half = 0.25 * mult * (k0h + k2h)

            #! Error estimate
            err     = norm_fun(dy_half - dy_full)       # absolute error
            fe      = self.tolerance / (err + 1e-15)    # relative error 

            #! Step size control
            fac     = 0.9 * fe**(1/3)
            fac     = self.xp.clip(fac, 0.2, 2.0)
            dt_new  = dt * fac
            dt      = min(dt_new, self.max_step)

        # Accept step
        self._dt    = dt
        yout        = y0 + dy_half
        return yout, dt, (step_info, other)

    def __repr__(self):
        """
        Return a string representation of the AdaptiveHeun object.
        """
        return f"AdaptiveHeun(dt={self._dt}, tol={self.tolerance}, max_step={self.max_step}, backend={self.backendstr})"

#########################################################################
#! General Runge-Kutta integration
#########################################################################

class RK(IVP):
    """
    General explicit Runge-Kutta integrator with arbitrary Butcher tableau.

    Parameters
    ----------
    a : array-like of shape (s, s)
        Lower-triangular matrix of stage coefficients.
    b : array-like of length s
        Weights for final combination.
    c : array-like of length s
        Nodes (c = sum of rows of a).
    dt : float
        Fixed step size dt.
    backend : str
        'numpy' or 'jax'

    Notes
    -----
    For common orders, use the `from_order` classmethod.
    """
    def __init__(self,
                a               : list, # Butcher tableau
                b               : list, # Weights
                c               : list, # Nodes
                dt              : float = 1e-3,
                backend         : str   = 'numpy',
                rhs_prefactor   : float = 1.0):
        
        # Initialize base with rhs_prefactor for API consistency
        super().__init__(backend, rhs_prefactor=rhs_prefactor, dt=dt)
        xp          = self.xp
        # Convert tableau to arrays
        self.a      = xp.array(a, dtype=xp.float64)
        self.b      = xp.array(b, dtype=xp.float64)
        self.c      = xp.array(c, dtype=xp.float64)
        self.stages = len(self.b)

    @property
    def order(self) -> int:
        return len(self.b)

    @classmethod
    def from_order(cls, order: int, dt: float = 1e-3, backend: str = 'numpy', rhs_prefactor: float = 1.0):
        """
        Create a Runge-Kutta method instance from a specified order.
        Parameters:
            order (int):
                The order of the Runge-Kutta method. Supported values are 1 (Euler), 2 (RK2), and 4 (RK4).
            dt (float, optional):
                The time step size. Defaults to 1e-3.
            backend (str, optional):
                The computational backend to use (e.g., 'numpy'). Defaults to 'numpy'.
        Returns:
            cls:
                An instance of the class initialized with the appropriate Butcher tableau for the specified order.
        Raises:
            ValueError: If the specified order is not supported.
        """

        # Define tableau for orders 1,2,4
        if order == 1:
            a = [[0.0]]
            b = [1.0]
            c = [0.0]
        elif order == 2:
            a = [[0.0, 0.0], [1.0, 0.0]]
            b = [0.5, 0.5]
            c = [0.0, 1.0]
        elif order == 4:
            a = [[0.0, 0.0, 0.0, 0.0],
                 [0.5, 0.0, 0.0, 0.0],
                 [0.0, 0.5, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0.0, 0.5, 0.5, 1.0]
        else:
            raise ValueError(f"Unsupported order: {order}")
        return cls(a, b, c, dt=dt, backend=backend, rhs_prefactor=rhs_prefactor)

    def step(self, f, t: float, y, **rhs_args):
        """
        Perform one Runge-Kutta step.
        
        Returns
        -------
        yout :
            new state
        dt :
            step size used
        """
        
        xp  = self.xp
        h   = self._dt
        k   = [None] * self.stages
        
        #! Compute stages
        for i in range(self.stages):
            ti = t + self.c[i] * h
            yi = y
            for j in range(i):
                yi = yi + h * self.a[i, j] * k[j]
            k[i], step_info, other = f(yi, ti, **rhs_args, int_step=i)

        #! Combine
        yout = y
        for i in range(self.stages):
            yout = yout + (self._rhs_prefactor * h) * self.b[i] * k[i]
        return yout, h, (step_info, other)

    def __repr__(self):
        """
        Return a string representation of the RK object.
        """
        return f"RK(order={self.order}, dt={self._dt}, backend={self.backendstr})"

#########################################################################
#! From scipy.integrate import solve_ivp
#########################################################################

class ScipyRK(IVP):
    """
    Wrapper for scipy.integrate.solve_ivp with explicit Runge-Kutta methods.

    Parameters
    ----------
    dt : float
        Initial and maximum time step delta t.
    tol : float
        Relative and absolute tolerance for solver.
    max_step : float or None
        Maximum allowed step size; if None uses dt.
    method : str
        Solver method: one of 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'.
    backend : str
        'numpy' or 'jax'. JAX backend is not supported and will fallback to NumPy.

    """
    def __init__(self,
                dt           : float = 1e-3,
                tol          : float = 1e-6,
                max_step     : float = None,
                method       : str   = 'RK45',
                backend      : str   = 'numpy',
                rhs_prefactor: float = 1.0):
        super().__init__(backend, rhs_prefactor=rhs_prefactor, dt=dt)
        self._dt                = float(dt)
        self.tol                = float(tol)
        self.max_step           = float(max_step) if max_step is not None else None
        self.method             = method
        self.supported_methods  = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
        if method not in self.supported_methods:
            raise ValueError(f"Method '{method}' not supported. Choose from {self.supported_methods}.")
        if backend == 'jax':
            warnings.warn("ScipyRK does not support JAX backend; using NumPy internally.")
            self.xp = np

    def step(self, f, t: float, y, **rhs_args):
        """
        Perform one adaptive step using solve_ivp over [t, t + dt].

        Returns
        -------
        yout : ndarray
            State at end of interval.
        dt_actual : float
            Actual time step taken.
        """
        def _ode_system(t_i, y_i):
            # Multiply RHS by prefactor for consistency with other integrators
            return self._rhs_prefactor * f(y_i, t_i, **rhs_args)[0]

        t_span = (t, t + self._dt)
        sol = solve_ivp(
            fun         = _ode_system,
            t_span      = t_span,
            y0          = y,
            method      = self.method,
            rtol        = self.tol,
            atol        = self.tol,
            max_step    = self.max_step or self._dt
        )
        if not sol.success:
            raise RuntimeError(f"SciPy solver failed: {sol.message}")

        yout        = sol.y[:, -1]
        dt_actual   = sol.t[-1] - t
        self._dt    = dt_actual
        return yout, dt_actual, (None, None)
    
    def __repr__(self):
        """
        Return a string representation of the ScipyRK object.
        """
        return f"ScipyRK(method={self.method}, dt={self._dt}, tol={self.tol}, max_step={self.max_step}, backend={self.backendstr})"

#########################################################################

class OdeTypes:
    """
    Enum-like class for ODE types.
    """
    EULER      = 'euler'
    HEUN       = 'heun'
    RK2        = 'rk2'
    RK4        = 'rk4'
    ADAPTIVE   = 'adaptive'
    SCIPY      = 'scipy'

def choose_ode(ode_type: Union[str, int, OdeTypes], *, dt: float = 1e-1, rhs_prefactor: float = 1.0, backend: Any = 'numpy', **kwargs) -> IVP:
    """
    Choose an ODE solver based on the specified type.

    Parameters
    ----------
    ode_type : str or int
        Type of ODE solver to use.
    dt : float, optional
        Time step size. Default is 1e-1.
    backend : str, optional
        Computational backend to use. Default is 'numpy'.
    **kwargs : keyword arguments
        Additional arguments for the ODE solver.

    Returns
    -------
    IVP
        An instance of the selected ODE solver.
    """
    
    if isinstance(ode_type, int):
        ode_type = OdeTypes(ode_type)
    
    if isinstance(ode_type, str):
        ode_type = ode_type.lower()
    
    if ode_type == OdeTypes.EULER:
        return Euler(dt=dt, backend=backend, rhs_prefactor=rhs_prefactor, **kwargs)
    elif ode_type == OdeTypes.HEUN:
        return Heun(dt=dt, backend=backend, rhs_prefactor=rhs_prefactor, **kwargs)
    elif ode_type == OdeTypes.RK2:
        return RK.from_order(2, dt=dt, backend=backend, rhs_prefactor=rhs_prefactor, **kwargs)
    elif ode_type == OdeTypes.RK4:
        return RK.from_order(4, dt=dt, backend=backend, rhs_prefactor=rhs_prefactor, **kwargs)
    elif ode_type == OdeTypes.ADAPTIVE:
        return AdaptiveHeun(dt=dt, backend=backend, rhs_prefactor=rhs_prefactor, **kwargs)
    elif ode_type == OdeTypes.SCIPY:
        return ScipyRK(dt=dt, backend=backend, rhs_prefactor=rhs_prefactor, **kwargs)

    raise ValueError(f"Unknown ODE type: {ode_type}")

#########################################################################
#! End of file
#########################################################################