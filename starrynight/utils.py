from .configdefaults import config
import jax


# Integration codes
FLUX_ZERO = 0
FLUX_DAY_OCC = 1
FLUX_DAY_VIS = 2
FLUX_NIGHT_OCC = 3
FLUX_NIGHT_VIS = 4
FLUX_SIMPLE_OCC = 5
FLUX_SIMPLE_REFL = 6
FLUX_SIMPLE_OCC_REFL = 7
FLUX_TRIP_DAY_OCC = 8
FLUX_TRIP_NIGHT_OCC = 9
FLUX_QUAD_DAY_VIS = 10
FLUX_QUAD_NIGHT_VIS = 11

# Maximum number of iterations when computing `el2`
STARRY_EL2_MAX_ITER = 100

# Maximum number of iterations & tolerance when computing 2F1
STARRY_2F1_MAXITER = 200
STARRY_2F1_TOL = 1e-15

# Square root of the desired precision in `el2`
STARRY_EL2_CA = 1e-8

# Replace `inf` with this value in argument to `el2`
STARRY_HUGE_TAN = 1e15

# If |sin(theta)| or |cos(theta)| is less than this, set = 0
STARRY_T_TOL = 1e-12

# Low, medium, and high tolerance for root polishing
STARRY_ROOT_TOL_LOW = 1e-2
STARRY_ROOT_TOL_MED = 1e-10
STARRY_ROOT_TOL_HIGH = 1e-15

# Tolerance for duplicate roots
STARRY_ROOT_TOL_DUP = 1e-8

# Maximum number of root polishing iterations
STARRY_ROOT_MAX_ITER = 50

# If |b| is less than this value, set = 0
STARRY_B_ZERO_TOL = 1e-8

# Tolerance for various functions that calculate phi, xi, and lam
STARRY_ANGLE_TOL = 1e-13

# Hacks. Determining the integration paths close to the singular
# points of the occultation is quite hard, and the solution can
# often oscillate between two regimes. These tolerances prevent us
# from entering those regimes, at the cost of precision loss near
# these singular points.
STARRY_COMPLETE_OCC_TOL = 1e-8
STARRY_NO_OCC_TOL = 1e-8
STARRY_GRAZING_TOL = 1e-8

STARRY_PAL_BO_EQUALS_RO_TOL = 1e-6


def pairdiff(x):
    """Return the sum over pairwise differences of an array.

    This is used to evaluate a (series of) definite integral(s) given
    the antiderivatives at each of the integration limits.
    """
    return sum(-config.np.array(x)[::2] + config.np.array(x)[1::2])


def custom_gradient(func_and_grad):
    """Function wrapper enabling custom gradients for black-box functions.
    
    Based on https://github.com/google/jax/issues/1142#issuecomment-522283030.

    Args:
        func_and_grad (callable): A function taking in any number of positional
            arguments and returning the tuple (`f`, `df`) of the function value 
            `f` (a scalar) and its gradient `df` (a tuple of scalars, one per
            argument).
    
    Returns:
        callable: A jax-compatible function with its first derivative implemented.
    """
    if not config.use_jax:

        # Return just the function value
        return lambda *args: func_and_grad(*args)[0]

    else:

        # Declare
        func_p = jax.core.Primitive("func")

        # Bind
        def func(*args):
            return func_p.bind(*args)

        # Evaluation
        func_p.def_impl(lambda *args: func_and_grad(*args)[0])

        # Gradient
        def func_vjp(*args):
            val, grad = func_and_grad(*args)
            return val, lambda g: grad

        # Bind the gradient
        jax.interpreters.ad.defvjp_all(func_p, func_vjp)

        return func
