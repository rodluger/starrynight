import warnings
import starry
import jax

warnings.simplefilter("ignore")
starry.config.quiet = True
starry.config.lazy = True
jax.config.update("jax_enable_x64", True)

from .configdefaults import config
from .starrynight import *
from .viz import visualize
from .starrynight_version import __version__

__uri__ = "https://github.com/rodluger/starrynight"
__author__ = "Rodrigo Luger"
__email__ = "rodluger@gmail.com"
__license__ = "MIT"
__description__ = "Analytic occultation light curves in reflected light"
