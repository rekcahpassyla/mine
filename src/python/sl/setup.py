import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize

include_dirs = ['.', np.get_include()]
library_dirs = []
libraries = []

percytron = Extension(
    'percytron',
    ['./percytron.pyx'],
    language='C',
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
    extra_link_args=['-fopenmp']
)


alcygos = Extension(
    'alcygos',
    ['./alcygos.pyx'],
    language='C',
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
    extra_link_args=['-fopenmp']
)

setup(
    name='comp0078',
    version='1.0.0',
    ext_modules=cythonize([percytron, alcygos])
)
