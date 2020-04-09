'''
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(ext_modules = cythonize('usetest.pyx'),
        include_dirs = [np.get_include()])

'''
from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension

import os
os.environ['CC'] = 'g++'
os.environ['CXX'] = 'g++'

ext_modules = [
    Extension(
        name="TestingUSECython",
        sources=["cytest.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-O2", "-march=native", '-stdlib=libc++', '-std=c++11'],
        extra_link_args=["-O2", "-march=native", '-stdlib=libc++'],
        language="c",
        include_dirs=["."],
    )
]

setup(
    name="TestingUSECython", ext_modules=ext_modules, cmdclass={"build_ext": build_ext}
)