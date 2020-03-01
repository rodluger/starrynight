"""Install script for `starry`."""
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import setuptools
import subprocess
import sys
import os
import glob


STARRY_PATH = "/Users/rluger/src/starry/"


class get_pybind_include(object):
    """
    Helper class to determine the pybind11 include path.

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


include_dirs = []
ext_modules = [
    Extension(
        "starrynight._c_ops",
        ["starrynight/include/interface.cpp"],
        include_dirs=include_dirs
        + [
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            STARRY_PATH + "starry/_core/ops/lib/include",
            STARRY_PATH + "starry/_core/ops/lib/vendor/eigen_3.3.5",
        ],
        language="c++",
    )
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ["-std=c++17", "-std=c++14", "-std=c++11"]

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError("Unsupported compiler -- at least C++11 support " "is needed!")


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {"msvc": ["/EHsc"], "unix": []}
    l_opts = {"msvc": [], "unix": []}

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append("-fcolor-diagnostics")
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, "-fvisibility=hidden"):
                opts.append("-fvisibility=hidden")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


setup(
    name="starrynight",
    author="Rodrigo Luger",
    author_email="rodluger@gmail.com",
    url="https://github.com/rodluger/starrynight",
    description="Analytic occultation light curves for astronomy.",
    license="MIT",
    packages=find_packages(),
    ext_modules=ext_modules,
    use_scm_version={
        "write_to": os.path.join("starrynight", "starrynight_version.py"),
        "write_to_template": '__version__ = "{version}"\n',
    },
    install_requires=[
        "setuptools_scm",
        "numpy>=1.13.0",
        "scipy>=1.2.1",
        "astropy>=3.1",
        "matplotlib>=3.0.2",
        "pybind11>=2.4",
        "theano>=1.0.4",
        "ipython",
        "pillow",
        "exoplanet>=0.2.0",
        "packaging",
    ],
    setup_requires=["setuptools_scm", "pybind11>2.4"],
    cmdclass={"build_ext": BuildExt},
    include_package_data=True,
    zip_safe=False,
)
