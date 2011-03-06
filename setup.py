from setuptools import setup
setup(
      name = 'ufunc_gen',
      version=".2",
      description="Cython utilities for building numpy ufuncs.",
      author="John Salvatier",
      author_email="jsalvati@u.washington.edu",
      url="",
      license="BSD",
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering',
         ],
      requires=['NumPy (>=1.3)',],
      long_description="",
      packages = ['ufunc_gen'],
)