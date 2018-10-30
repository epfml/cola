import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from sklearn._build_utils import get_blas_info

cblas_libs, blas_info = get_blas_info()

if os.name == 'posix':
    cblas_libs.append('m')

include_dirs = [numpy.get_include()] + blas_info.pop('include_dirs', [])

ext_modules = [
    Extension("fast_cd.svm", sources=["fast_cd/svm.pyx"], libraries=["m"], include_dirs=include_dirs),
    Extension("fast_cd.elasticnet", sources=["fast_cd/elasticnet.pyx"], libraries=["m"], include_dirs=include_dirs)
]

setup(name="CoLA",
      ext_modules=cythonize(ext_modules),
      # include_dirs=include_dirs,
      packages=[
          'cola',
          'fast_cd'])
