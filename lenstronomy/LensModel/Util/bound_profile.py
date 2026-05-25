import inspect
import warnings
import numpy as np
from functools import wraps


class OutOfBoundError(Exception):
    # specific error: raise if particles are ouf of bounds
    def __init__(self, error):
        self.error   = str(error)
        self.message = f"OoB Error: {self.error}"
        super().__init__(self.message)
    def __str__(self):
        return self.message


def enforce_bounds(func):
    """Bounds checker for lens profiles"""
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            if name in self.lower_limit_default:
                v = np.asarray(value, dtype=float)
                lo = self.lower_limit_default[name]
                hi = self.upper_limit_default[name]

                if np.any(v < lo) or np.any(v > hi):
                    err_message = f"{func.__name__}: parameter '{name}' out of bounds [{lo}, {hi}], got [{v.min()}, {v.max()}]"
                    if not getattr(self,"ignore_OoBErr",False):
                        raise OutOfBoundError(err_message)
                    else:
                        warnings.warn(err_message)

        return func(self, *args, **kwargs)

    return wrapper


def enforce_bounds_parall(param_names):
    """High-performance bounds checker for parallelised lens profiles"""


    def decorator(func):
        # resolve parameter positions ONCE
        code = func.__code__
        arg_names = code.co_varnames[: code.co_argcount]

        indices = []
        for name in param_names:
            if name not in arg_names:
                raise ValueError(f"{name} not in {func.__name__} signature")
            indices.append(arg_names.index(name))

        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]

            lower = self.lower_limit_default
            upper = self.upper_limit_default

            for idx, name in zip(indices, param_names):
                if idx < len(args):
                    v = args[idx]
                else:
                    v = kwargs.get(name)

                if v is None:
                    continue

                # Fast scalar path
                if np.isscalar(v):
                    if v < lower[name] or v > upper[name]:
                        err_message = f"{func.__name__}: {name}={v} outside [{lower[name]}, {upper[name]}]"
                        if not getattr(self,"ignore_OoBErr",False):
                            raise OutOfBoundError(err_message)
                        else:
                            warnings.warn(err_message)
                    continue

                # Array path
                arr = np.asarray(v)
                vmin = arr.min()
                vmax = arr.max()

                if vmin < lower[name] or vmax > upper[name]:
                    err_message = f"{func.__name__}: {name} outside bounds [{lower[name]}, {upper[name]}], got [{vmin}, {vmax}]"
                    if not getattr(self,"ignore_OoBErr",False):
                        raise OutOfBoundError(err_message)
                    else:
                        warnings.warn(err_message)
                        
            return func(*args, **kwargs)

        return wrapper

    return decorator

