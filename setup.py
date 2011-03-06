from setuptools import setup
setup(
      name = 'ufunc_gen',
      version=".2",
      description="Code generation utility for creating UFuncs.",
      author="John Salvatier",
      author_email="jsalvatier@gmail.com",
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
      requires=['jinja2',],
      long_description="",
      packages = ['ufunc_gen'],
)