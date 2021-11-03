import functools, json
from collections import ChainMap

none_predicate = lambda _1,_2: None
stmt_predicate = expr_predicate = none_predicate
 
fn_getitem2 = lambda o, i: o[i]
def get(tree,path, default=None):
	#
	try:
		return functools.reduce(fn_getitem2,path,tree)
	except IndexError:
		return default

def iter_in_expr(e, p=[]):
	""
	expr_predicate(e, p)
	if isinstance(e,list):
		assert isinstance(e[0],str) and not e[0].isidentifier() and isinstance(e[-1],dict), e
		for i,ei in enumerate(e[1:-1], start=1):
			iter_in_expr(ei, p+[i])
		if 'Stmt' in e[-1]:
			assert e[0] == 'StmtExpr.', e
			iter_in_stmt(e[-1]['Stmt'], p + [-1,'Stmt'])
	elif isinstance(e,str):
		assert e.isidentifier(), e
	else:
		assert isinstance(e,(bool,float,int)), e

def iter_in_stmt(stmt, p=[]):
	""
	assert isinstance(stmt,list) and isinstance(stmt[-1], dict)
	sk = stmt[0]
	assert sk.endswith('Stmt')
	stmt_predicate(stmt,p)
	if sk == 'AttributedStmt':
		[s1] = stmt[1:-1]
		iter_in_stmt(s1, p+[1])
	elif sk in ('BreakStmt','ContinueStmt','NullStmt'):
		[] = stmt[1:-1]
	elif sk == 'CompoundStmt':
		for i, si in enumerate(stmt[1:-1], start=1):
			iter_in_stmt(si, p+[i])
	elif sk == 'DeclStmt':
		for i, (vn, [vt,vi]) in enumerate(stmt[1:-1], start=1):
			if vi is not None:
				iter_in_expr(vi, p+[i,1,1])
	elif sk == 'DoStmt':
		[B,C] = stmt[1:-1]
		iter_in_stmt(B, p+[1])
		iter_in_expr(C, p+[2])
	elif sk == 'ExprStmt':
		[e] = stmt[1:-1]
		iter_in_expr(e, p+[1])
	elif sk == 'ForStmt':
		[Start,Stop,Step,Body] = stmt[1:-1]
		if Start is not None:
			iter_in_stmt(Start, p+[1])
		if Stop is not None:
			iter_in_expr(Stop, p+[2])
		if Step is not None:
			iter_in_expr(Step, p+[3])
		iter_in_stmt(Body, p+[4])
	elif sk == 'IfStmt':
		[If_cond, If_then, If_else] = stmt[1:-1]
		iter_in_expr(If_cond, p+[1])
		iter_in_stmt(If_then, p+[2])
		if If_else is not None:
			iter_in_stmt(If_else, p+[3])
	elif sk == 'ReturnStmt':
		if len(stmt) > 2:
			[s1] = stmt[1:-1]
			iter_in_expr(s1, p+[1])
	elif sk == 'SwitchStmt':
		iter_in_expr(stmt[1], p+[1])
		for i, st in enumerate(stmt[2:-1], start=2):
			if st[0] == 'CaseStmt':
				assert len(st) == 3 and isinstance(st[-1],dict), st
				iter_in_expr(st[1], p+[i,1])
			else:
				iter_in_stmt(st, p+[i])
	elif sk == 'WhileStmt':
		[C,B] = stmt[1:-1]
		iter_in_stmt(B, p+[2])
		iter_in_expr(C, p+[1])
	else:
		assert False, sk	

##############################################


##############################################


list_of_decl = current_decl = kernels = non_kernels = None

def chk_atomics(decl):
	""
	global stmt_predicate, expr_predicate
	stmt_predicate = none_predicate
	expr_predicate = lambda e,p: print((e,p)) if isinstance(e,str) and e.startswith('atomic_') else None
	assert decl[0] == 'FunctionDecl'
	[_k,name, decl_value, decl_loc] = decl
	[r_ty, pl, body_stmt, attr_l] = decl_value
	iter_in_stmt(body_stmt)
	

def expr_barrier(e,p):
	"" 
	if e == "barrier":
		#print(e,p)
		is_kernel = 'OpenCLKernelAttr' in current_decl[2][-1]
		#if not is_kernel:
		#	print('!!! not a kernel !!!!!')
		father = get(current_decl, p[:-1])
		assert father[0] == 'ImplicitCastExpr.FunctionToPointerDecay', father
		gf = get(current_decl, p[:-2])
		assert gf[0] == '.(...)', gf
		ggf = get(current_decl, p[:-3])
		assert ggf[0] == 'ExprStmt', ggf
		g3f = get(current_decl, p[:-4])
		assert g3f[0] == 'CompoundStmt', g3f
		if len(p) == 6:
			hist = []  # p[:-4] == [2,2]
		elif len(p) == 7:
			assert False, p
		elif len(p) == 8:
			assert get(current_decl,p[:-5])[0] in ('IfStmt','ForStmt','WhileStmt'), get(current_decl,p[:-5])
			assert get(current_decl,p[:-6])[0] == 'CompoundStmt'
			hist = get(current_decl,p[:-5])[:1]
		elif len(p) == 9:
			assert False, p
		elif len(p) == 10:
			assert get(current_decl,p[:-5])[0] in ('ForStmt',), get(current_decl,p[:-5])
			assert get(current_decl,p[:-6])[0] == 'CompoundStmt'
			assert get(current_decl,p[:-7])[0] in ('IfStmt',), get(current_decl,p[:-7])
			assert get(current_decl,p[:-8])[0] == 'CompoundStmt'
			hist = get(current_decl,p[:-7])[:1] + get(current_decl,p[:-5])[:1] # -5 : For  -6 : Compound  -7 If    -8  Compound
		else:
			assert False, p
		print(e,p,is_kernel, hist)
		if (not is_kernel) and hist != []:
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!') 
		

def chk_barrier(decl):
	""
	global stmt_predicate, expr_predicate
	stmt_predicate = none_predicate
	expr_predicate = expr_barrier
	assert decl[0] == 'FunctionDecl'
	[_k,name, decl_value, decl_loc] = decl
	[r_ty, pl, body_stmt, attr_l] = decl_value
	iter_in_stmt(body_stmt, [2,2])

###########################

def expr_collect_decl(e,p):
	""
	pass

def stmt_collect_decl(s,p):
	""
	if s[0] == 'CompoundStmt':
		d = {}; cpt = 0
		for st in s[1:-1]:
			if st[0] == 'DeclStmt':
				d.update((vn , {'type':vt}) for vn, [vt,_vi] in st[1:-1])
				cpt += len(st)-2
		assert len(d) == cpt
		assert 'Decl' not in s[-1]
		if d != {}:
			s[-1]['Decl'] = d
	elif s[0] == 'DeclStmt': # verif de coherence
		father = get(current_decl, p[:-1])
		assert father[0] in ('CompoundStmt','ForStmt'), father
	elif s[0] == 'ForStmt':
		st = s[1]
		assert st is None or st[0] in ('DeclStmt','ExprStmt'), st
		if st is not None and st[0] == 'DeclStmt':
			d = {vn : {'type':vt} for vn, [vt,_vi] in st[1:-1]}
			assert len(d) == len(st)-2
			assert 'Decl' not in s[-1]
			s[-1]['Decl'] = d
	
def collect_decl(decl):
	"""
	construit les environnements locaux pour les blocs et les for
	"""
	global stmt_predicate, expr_predicate
	stmt_predicate = stmt_collect_decl
	expr_predicate = none_predicate
	assert decl[0] == 'FunctionDecl'
	[_k,name, decl_value, decl_loc] = decl
	[r_ty, pl, body_stmt, attr_l] = decl_value
	assert 'Decl' not in decl_loc
	decl_loc['Decl'] = dict(pl)
	iter_in_stmt(body_stmt, [2,2])
	

############################

unique_names_set = None
def stmt_unique_names(s,p):
	""
	if s[0] == 'DeclStmt':
		sl = [vn for vn,_ in s[1:-1]]
		ss = set(sl)
		assert len(ss) == len(sl)
		if ss & unique_names_set != set():
			print('!!!! VAR CLASH !!! ', ss & unique_names_set,s[-1])
		unique_names_set.update(ss)

def unique_names(decl):
	""
	global stmt_predicate, expr_predicate
	global unique_names_set
	stmt_predicate = stmt_unique_names
	expr_predicate = none_predicate
	assert decl[0] == 'FunctionDecl'
	[_k,name, decl_value, decl_loc] = decl
	[r_ty, pl, body_stmt, attr_l] = decl_value
	sl = [p[0] for p in pl] + [name]
	ss = set(sl)
	assert len(ss) == len(sl)
	unique_names_set = ss
	iter_in_stmt(body_stmt, [2,2])

############################

modified_variables_set = {}

def expr_mv(e,p):
	""
	pass

def stmt_mv(s,p):
	""
	pass

def modified_variables(stmt, p):
	""
	global stmt_predicate, expr_predicate
	stmt_predicate = stmt_mv
	expr_predicate = expr_mv
	iter_in_stmt(stmt,p)

if __name__ == "__main__":
	if True:
		fd = open('c:/Temp/ocl_dump.json','r')
		js = json.load(fd)
		fd.close()
	for [filename, cpp_args, stderr, list_of_decl] in js:
		print('***** ', filename, ' ********')
		assert all(len(decl)==4 and '__constant' in decl[2][0] for decl in list_of_decl if \
			 decl[0] == 'VarDecl')
		global_constants = [decl[1] for decl in list_of_decl if \
			decl[0] == 'VarDecl']
		print('global constants : ', global_constants)
		kernels = [decl[1] for decl in list_of_decl if \
			 decl[0] == 'FunctionDecl' and 'OpenCLKernelAttr' in decl[2][3]]
		print('kernels : ', kernels)
		non_kernels = [decl[1] for decl in list_of_decl if \
			 decl[0] == 'FunctionDecl' and 'OpenCLKernelAttr' not in decl[2][3]]
		print('non kernels : ', non_kernels)
		for decl in list_of_decl:
			if decl[0] != 'FunctionDecl':
				continue
			print('---------- ', decl[1], ' ---------')
			current_decl = decl
			#chk_barrier(decl)
			collect_decl(decl)
# 			unique_names(decl)
# 			[_k,name, decl_value, decl_loc] = decl
# 			[r_ty, pl, body_stmt, attr_l] = decl_value
# 			mvs = modified_variables(body_stmt, [2,2])
			