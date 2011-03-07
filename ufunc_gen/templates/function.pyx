#macro ufunc(f)
@cython.cdivision(True)
def {{f.name}} ({{f.arglist}}):
    #if f.docstring
    """
    {{f.docstring}}
    """
    #endif
    cdef np.broadcast it
    
    # for var in f.vars 
    cdef {{var.type}} {{var.e_name}}
    cdef {{var.type}} *{{var.e_name}}_larray
    # endfor 
    
    cdef {{f.otype}} out = <{{f.otype}}>0
    
    # for var in f.vars 
    {{var.name}} = np.asarray({{var.name}},{{var.dtype}})
    # endfor 
    
    # if f.array_out 
    Aout = np.empty(np.broadcast({{f.arglist}}).shape,{{f.odtype}})
    # endif
    
    it = np.broadcast({{f.arglist}}{% if f.array_out %},Aout {% endif %})
    
    cdef int saxis = np.PyArray_RemoveSmallest(it)
    
    #for var in f.vars
    cdef int saxis_stride{{loop.index0}} = (<PyArrayIterObject*>(it.iters[{{loop.index0}}])).strides[saxis]
    # endfor
    
    #if f.array_out 
    cdef int saxis_stride{{f.vars|length}} = (<PyArrayIterObject*>(it.iters[{{f.vars|length}}])).strides[saxis]
    #endif 
    
    cdef int i
    cdef int saxis_dim 
    if it.nd > 0:
        saxis_dim = it.shape[saxis]
    else:
        saxis_dim = 1
    cdef int constraint_break = 0
    
    with nogil:
        while np.PyArray_MultiIter_NOTDONE(it):
            
            #for var in f.vars
            {{var.e_name}}_larray = (<{{var.type}}*>np.PyArray_MultiIter_DATA(it, {{loop.index0}}))
            # endfor 
            for i in range(saxis_dim):
                
                #for var in f.vars
                {{var.e_name}} = (<{{var.type}}*>({{var.e_name}}_larray + i*saxis_stride{{loop.index0}}))[0]
                # endfor 
        
                # if f.constraints
                if not ({{f.constraints}}):
                    contraint_break = 1
                    break
                # endif
                
                {{f.loop_code|indent(8)}}
                
                # if array_out
                (<{{f.out_type}}*>((<char*>np.PyArray_MultiIter_DATA(it, {{f.vars|length}})) + i*saxis_stride{{f.vars|length}}))[0] = out
                # endif
            
            np.PyArray_MultiIter_NEXT(it) 
    
    if constraint_break != 0:
        return -inf
    #if f.array_out
    return Aout
    # else 
    return out
    # endif
# endmacro