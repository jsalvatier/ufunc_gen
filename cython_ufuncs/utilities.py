'''
Created on Jul 22, 2010

@author: johnsalvatier
'''
import collections

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el
            
newline = '\n'
def add_newlines(lines):
    return [line + newline for line in lines]