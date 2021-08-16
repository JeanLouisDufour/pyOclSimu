from c_lexer import CLexer

def lex_error_func(msg, line, column):
	print(msg, line, column)

scope_stack = [{}]

def lex_on_lbrace_func():
	scope_stack.append({})

def lex_on_rbrace_func():
	scope_stack.pop()

def is_type_in_scope(name):
	" Is *name* a typedef-name in the current scope? "
	for scope in reversed(scope_stack):
		# If name is an identifier in this scope it shadows typedefs in
		# higher scopes.
		in_scope = scope.get(name)
		if in_scope is not None: return in_scope
	return False

def lex_type_lookup_func(name):
	""" Looks up types that were previously defined with
            typedef.
            Passed to the lexer for recognizing identifiers that
            are types.
	"""
	is_type = is_type_in_scope(name)
	return is_type

clex = CLexer(
            error_func=lex_error_func,
            on_lbrace_func=lex_on_lbrace_func,
            on_rbrace_func=lex_on_rbrace_func,
            type_lookup_func=lex_type_lookup_func)

clex.build(optimize=False)

# from ply.lex import LexToken

clex.input('hello/*foo*/12.0 world//bar')
tl = list(iter(clex.token, None)) # un CLexer n'est pas iterable
tyl = [i.type for i in tl]
assert tyl == ['ID', 'FLOAT_CONST', 'ID'], tyl

l = clex.lexer
l.input('hello/*foo*/12.0 world//bar')
tl = list(l) # un Lexer est iterable
tyl = [i.type for i in tl]
assert tyl == ['ID', 'FLOAT_CONST', 'ID'], tyl