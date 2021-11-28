import os
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
# from sklearn._build_utils import get_blas_info
from numpy.distutils.system_info import get_info


def get_blas_info():
    def atlas_not_found(blas_info_):
        def_macros = blas_info.get('define_macros', [])
        for x in def_macros:
            if x[0] == "NO_ATLAS_INFO":
                # if x[1] != 1 we should have lapack
                # how do we do that now?
                return True
            if x[0] == "ATLAS_INFO":
                if "None" in x[1]:
                    # this one turned up on FreeBSD
                    return True
        return False

    blas_info = get_info('blas_opt', 0)
    if (not blas_info) or atlas_not_found(blas_info):
        cblas_libs = ['cblas']
        blas_info.pop('libraries', None)
    else:
        cblas_libs = blas_info.pop('libraries', [])

    return cblas_libs, blas_info




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
