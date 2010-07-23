'''
Created on Jul 22, 2010

@author: johnsalvatier
'''
from utilities import flatten, add_newlines

__all__ = ['create_generic_loops']
def create_generic_loops(file_prefix, input_types, output_types, loop_names = None):
    # create the header 
    n = len(input_types)
    assert len(output_types) == n and (loop_names is None or len(loop_names) == n)
    
    if loop_names is None:
        loop_names = ['']  * n
        
    for i in xrange(n):
        if loop_names[i] == '':
            loop_names[i] = 'PyUFunc_' + ''.join(input_types[i]) + '_' + output_types[i]
        
    
    
    headerc_file_name = file_prefix + '.h'
    
    guard_name = file_prefix + 'H'
    headerc =[
    '#ifndef ' + guard_name,
    '#define ' + guard_name,
    '#include "Python.h"',
    '#include "numpy/arrayobject.h"',
    ['void ' + name + '(char **args, npy_intp *dimensions, npy_intp *steps, void *func) ;' for name in  loop_names],    
    '#endif']
    
    
    loop_defs_file_name = file_prefix + '.c'
    loop_defs = [
    '#include "Python.h"',
    '#include "numpy/arrayobject.h"',
    '#include "loops.h"',
    '',
    ['typedef ' + otype + ' ' + name + 'f(' + ','.join(itypes) + ');'         for itypes, otype, name in zip(input_types, output_types, loop_names)],   
    '']
    for itypes, otype, name in zip(input_types, output_types, loop_names):
        nt = len(itypes)
        loop_defs.append(
        ['void ' + name + '(char **args, npy_intp *dimensions, npy_intp *steps, void *func)',
        '{',
        '    npy_intp n = dimensions[0];',
        ['    npy_intp istep' + str(i) + ' = steps[' + str(i) +'];' for i in xrange(nt)], 
        '    npy_intp ostep = steps[' + str(n) + '];',
        
        ['    char *ip' + str(i) + ' = args[' + str(i) +'];'    for i in xrange(nt)],
        '    char *op = args[' + str(n) + '];',
        '    ' + name + 'f *f = (' + name +'f *)func;',
        '    npy_intp i;',
        '    for(i = 0; i < n; i++, ' + ','.join(['ip' + str(i) + '+= istep' + str(i) for i in xrange(nt)]) + ', op += ostep)',
        '    {',
        ['        ' + itypes[i] + '*in' + str(i) +' = (' + itypes[i] +' *)ip' + str(i) + ';' for i in xrange(nt)],
        '        ' + otype +' *out = (' + otype +' *)op;',
        '        *out = f(' + ','.join(['*in' + str(i) for i in xrange(nt)]) +');',
        '    }',
        '}'])
        
        
        
    headercy_file_name = file_prefix + '.pxi'
    headercy = [
    'from numpy cimport npy_intp',
    'cdef extern from "'+ headerc_file_name +'":',
    ['    void ' + name + '(char **args, npy_intp *dimensions, npy_intp *steps, void *func)' for name in  loop_names]]
    
    
    write(headerc_file_name,headerc)
    
    write(loop_defs_file_name,loop_defs)
    
    write(headercy_file_name, headercy)      

def write(file, lines): 
    with open(file, 'w') as f:
        f.writelines(add_newlines(flatten(lines)))
         