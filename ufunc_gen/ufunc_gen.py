'''
Created on Jul 22, 2010

@author: johnsalvatier
'''
from utilities import flatten 
from jinja2 import Environment, PackageLoader, DebugUndefined
import string
import re

__all__ = ['generate_ufuncs','UFuncDefinition','NumpyVarDefinition'] 

env = Environment(loader=PackageLoader('ufunc_gen', 'templates'), line_statement_prefix = '#', undefined = DebugUndefined)

def generate_ufuncs(file, function_definitions, includes = []): 
    template = env.get_template('function_file.pxy')
    context = {'functions' : flatten(function_definitions),
               'includes' : includes}
    
    with open(file, 'w') as f:
        f.write(template.render(**context))

class NumpyVarDefinition(object):
    def __init__(self, name, type):
        self.name = string.strip(name)
        self.e_name = self.name +'_v'
        self.type =  c_type(type)
        self.dtype = string.strip(type)

class UFuncDefinition(object):
    def __init__(self, name,vars,odtype,constraints, calc, docstring = None, array_out=True):
        self.name = name
        self.docstring = docstring 
        self.vars = vars
        self.array_out = array_out
        self.arglist = ', '.join([var.name for var in self.vars])
        self.odtype = odtype
        self.otype = c_type(self.odtype)
        self.constraints = replace_var_identifiers(constraints, self.vars)
        self.loop_code =  replace_var_identifiers(calc, self.vars)

c_type_dict = {'float' : 'double',
               'int'   : 'int',
               'bool'  : 'bool'}

def c_type(dtype):
    return c_type_dict[string.strip(dtype)]

python_identifier = '[_A-Za-z][_A-Za-z1-9]*'

def replace_var_identifiers(code, vars):
    vardict = {}
    for var in vars:
        vardict[var.name ] = var.e_name
    
    #if a python identifier is in the dict of variables, replace it with the e_name otherwise do nothing
    return re.sub(python_identifier, lambda s: vardict.get(s.group(0), s.group(0)), code)