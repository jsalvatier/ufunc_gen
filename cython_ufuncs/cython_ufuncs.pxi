from numpy cimport *

import_ufunc() # must come first!!!!

DEF _UFUNCS_MAX = 300
DEF _UFUNCS_S_MAX = 1000

cdef:
	PyUFuncGenericFunction _ufuncs_functions[_UFUNCS_MAX]
	void* _ufuncs_data[_UFUNCS_MAX]
	char _ufuncs_signatures[_UFUNCS_S_MAX]
	int _ufuncs_set = 0
	int _ufuncs_s_set = 0
	
cdef inline PyUFuncGenericFunction * functions():
	return &_ufuncs_functions[_ufuncs_set]
	
cdef inline void* *data():
	return &_ufuncs_data[_ufuncs_set]
	
cdef inline char *signatures():

	return &_ufuncs_signatures[_ufuncs_s_set]
        
cdef ufuncs_add_def(void *data, PyUFuncGenericFunction function,
		char* signatures, int num_signatures):
	cdef int i
	global _ufuncs_set, _ufuncs_s_set
	_ufuncs_functions[_ufuncs_set] = function
	_ufuncs_data[_ufuncs_set] = data
	_ufuncs_set += 1
	for i in range(num_signatures):
		_ufuncs_signatures[_ufuncs_s_set] = signatures[i]
		_ufuncs_s_set += 1
        
       
#must call in this order
#testf = PyUFunc_FromFuncAndData(functions(), data(), signatures(), 1, 1, 1,PyUFunc_None, "testf", "a test function", 0)
#ufuncs_add_def(<void*>f, PyUFunc_long_double, sig1, 2)
