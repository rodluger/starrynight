import numpy
import jax


class _Config(object):
    def __init__(self):
        self._use_jax = False
        self.np = numpy

    @property
    def use_jax(self):
        return self._use_jax

    @use_jax.setter
    def use_jax(self, value):
        self._use_jax = bool(value)
        if self._use_jax:
            self.np = jax.numpy
        else:
            self.np = numpy


config = _Config()
