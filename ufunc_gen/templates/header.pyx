#macro header(includes)

import numpy as np
cimport numpy as np
cimport cython

#for I in includes
from {{I}} cimport *
#endfor

np.import_ufunc()
np.import_array()

cdef double inf = <double>np.inf 
cdef double pi = <double>np.pi

cdef extern from "numpy/arrayobject.h":
    ctypedef struct PyArrayIterObject: 
        np.npy_intp strides[np.NPY_MAXDIMS]
        
#endmacro 