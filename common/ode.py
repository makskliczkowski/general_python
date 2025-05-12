'''
file    : general_python/common/ode.py


'''

import numpy as np
import numba as nb
import warnings
from abc import ABC, abstractmethod
from general_python.algebra.utils import get_backend, JAX_AVAILABLE
from scipy.integrate import solve_ivp

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from jax import jit
else:
    jnp             = None
    jit             = None
    jax             = None
    
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
    
    def __init__(self, backend: str = 'numpy'):
        """
        Initialize the ODE solver with a specified backend.

        Parameters
        ----------
        backend : str
            Backend to use for numerical operations ('numpy' or 'jax').
        """
        self.backend    = backend
        self.xp         = get_backend(backend)
        if self.xp is None:
            raise ValueError(f"Backend '{backend}' is not supported. Choose 'numpy' or 'jax'.")
        self._isjax     = not (self.xp is np)
        self._isnpy     = not self._isjax
        self._dt        = None
    
    @abstractmethod
    def step(self, f, t: float, y, **rhs_args):
        '''
        Compute one integration step without modifying internal state.
        This method should be implemented by subclasses.
        '''
        raise NotImplementedError

    def dt(self, h: float, i: int) -> float:
        return self._dt
    
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
        return f"SimpleIVP(backend={self.backend})"
    
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
    
    def __init__(self, dt: float = 1e-3, backend: str = 'numpy'):
        """
        Initializes the object with a specified time step and computational backend.

        Args:
            dt (float, optional):
                The time step to use for the ODE solver. Defaults to 1e-3.
            backend (str, optional):
                The computational backend to use (e.g., 'numpy'). Defaults to 'numpy'.
        """
        super().__init__(backend)
        self._dt = dt # does not need to be float, but should be a scalar

    def step(self, f, t: float, y, **rhs_args):
        """
        Compute one Euler step: y_{n+1} = y_n + \Delta t * f(y_n, t).

        Returns
        -------
        yout
            New state after one Euler step.
        dt
            The step size used.
        """
        #! Evaluate derivative through function
        dy      = f(y, t, **rhs_args, intStep=0)
        yout    = y + self._dt * dy
        return yout, self._dt

    def dt(self, h: float = None, i: int = None) -> float:
        """
        Return the fixed time-step size.
        Parameters
        ----------
        h : float, optional
            Not used in this implementation.
        i : int, optional
            Not used in this implementation.
        Returns
        -------
        float
            The fixed time-step size.
        """
        return self._dt

    def __repr__(self):
        """
        Return a string representation of the Euler object.
        """
        return f"Euler(dt={self._dt}, backend={self.backend})"
    
########################################################################
#! Heun integration
########################################################################

class Heun(IVP):
    r"""
    Second-order Heun (explicit trapezoidal) integrator.

    Parameters
    ----------
    dt : float
        Fixed step size Δt (can be adapted externally).
    backend : str
        'numpy' or 'jax'
    """
    
    def __init__(self, dt: float = 1e-3, backend: str = 'numpy'):
        super().__init__(backend)
        self._dt = dt

    def step(self, f, t: float, y, **rhs_args):
        """
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
        dt      = self._dt
        # Predictor slope
        k0      = f(y, t, **rhs_args, intStep=0)
        # Predictor step
        y_pred  = y + dt * k0
        # Corrector slope
        k1      = f(y_pred, t + dt, **rhs_args, intStep=1)
        # Combine as average
        yout    = y + 0.5 * dt * (k0 + k1)
        return yout, dt

    def __repr__(self):
        """
        Return a string representation of the Heun object.
        """
        return f"Heun(dt={self._dt}, backend={self.backend})"
    
########################################################################
#! Adaptive Heun integration
########################################################################

class AdaptiveHeun(IVP):
    """
    Adaptive second-order Heun integrator with error control.

    Parameters
    ----------
    dt : float
        Initial time step Δt.
    tol : float
        Error tolerance.
    max_step : float
        Maximum allowed time step.
    backend : str
        'numpy' or 'jax'
    """
    def __init__(self,
                dt   : float = 1e-3,
                tol         : float = 1e-8,
                max_step    : float = 1.0,
                backend     : str   = 'numpy'):
        super().__init__(backend)
        self._dt        = dt
        self.tolerance  = tol
        self.max_step   = max_step

    def step(self, f, t: float, y, norm_fun=None, **rhs_args):
        if norm_fun is None:
            norm_fun = self.xp.linalg.norm

        dt  = self._dt
        y0  = y
        fe  = 0.0
        
        #! Adapt until accepted
        while fe < 1.0:
            # Full step - 1st order (just the simple Heun step)
            k0      = f(y0, t, **rhs_args, intStep=0)
            y_full  = y0 + dt * k0
            k1      = f(y_full, t + dt, **rhs_args, intStep=1)
            dy_full = 0.5 * dt * (k0 + k1)

            # Two half steps - 2nd order (search for a better step)
            k0h     = k0
            y_half  = y0 + 0.5 * dt * k0h
            k1h     = f(y_half, t + 0.5 * dt, **rhs_args, intStep=2)
            dy_half = 0.5 * dt * k1h
            y_half2 = y_half + 0.5 * dt * k1h
            k2h     = f(y_half2, t + dt, **rhs_args, intStep=3)
            dy_half = 0.25 * dt * (k0h + k2h)

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
        return yout, dt
    
    def __repr__(self):
        """
        Return a string representation of the AdaptiveHeun object.
        """
        return f"AdaptiveHeun(dt={self._dt}, tol={self.tolerance}, max_step={self.max_step}, backend={self.backend})"
    
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
        Fixed step size Δt.
    backend : str
        'numpy' or 'jax'

    Notes
    -----
    For common orders, use the `from_order` classmethod.
    """
    def __init__(self,
                a           : list,
                b           : list,
                c           : list,
                dt          : float = 1e-3,
                backend     : str   = 'numpy'):
        super().__init__(backend)
        xp          = self.xp
        # Convert tableau to arrays
        self.a      = xp.array(a, dtype=xp.float64)
        self.b      = xp.array(b, dtype=xp.float64)
        self.c      = xp.array(c, dtype=xp.float64)
        self._dt    = dt
        self.stages = len(self.b)

    @property
    def order(self) -> int:
        return len(self.b)

    @classmethod
    def from_order(cls, order: int, dt: float = 1e-3, backend: str = 'numpy'):
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
        return cls(a, b, c, dt=dt, backend=backend)

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
            k[i] = f(yi, ti, **rhs_args, intStep=i)
        
        #! Combine
        yout = y
        for i in range(self.stages):
            yout = yout + h * self.b[i] * k[i]
        return yout, h

    def __repr__(self):
        """
        Return a string representation of the RK object.
        """
        return f"RK(order={self.order}, dt={self._dt}, backend={self.backend})"
    
#########################################################################
#! From scipy.integrate import solve_ivp
#########################################################################

class ScipyRK(IVP):
    """
    Wrapper for scipy.integrate.solve_ivp with explicit Runge-Kutta methods.

    Parameters
    ----------
    dt : float
        Initial and maximum time step Δt.
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
                dt          : float = 1e-3,
                tol         : float = 1e-6,
                max_step    : float = None,
                method      : str = 'RK45',
                backend     : str = 'numpy'):
        super().__init__(backend)
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
            return f(y_i, t_i, **rhs_args)

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
        return yout, dt_actual
    
    def __repr__(self):
        """
        Return a string representation of the ScipyRK object.
        """
        return f"ScipyRK(method={self.method}, dt={self._dt}, tol={self.tol}, max_step={self.max_step}, backend={self.backend})"

#########################################################################
#! End of file
#########################################################################