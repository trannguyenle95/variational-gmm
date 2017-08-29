from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("src/streaming_gmm/features/streaming_aov.pyx")
)
