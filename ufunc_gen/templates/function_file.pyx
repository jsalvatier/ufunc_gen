# import "header.pyx" as header
# import "function.pyx" as function

{{header.header(includes)}}


# for f in functions
{{function.ufunc(f)}}

#endfor