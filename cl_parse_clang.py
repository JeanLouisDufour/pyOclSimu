from subprocess import check_output, run, PIPE
import json, os, re, tempfile, traceback

ctx_OpenCL = True

clang_path = r'c:/Program Files/LLVM/bin/clang.exe'
#clang_path = 'clang'
clang_options = '-cl-std=CL2.0 -Xclang'.split()
# clang_options += '-finclude-default-header -target spir64 -c -O0 -emit-llvm -o c:/Temp/tmp.bc'.split() # compil
#clang_options += ['-include','opencl-c.h']
# /cygdrive/c/Program Files (x86)/IntelSWTools/system_studio_2020/OpenCL/sdk/bin/x64/opencl-c.h
# /cygdrive/c/"Program Files"/LLVM/lib/clang/11.0.0/include
# recopie (avec ...base...) dans le dossier courant
clang_options += '-ast-dump=json -fsyntax-only'.split() # parse only
clang_options += " -include opencl-c.h".split()

id2obj = name2obj = rec2obj = td2obj = None

def loc_clean(loc):
	""
	return {x:loc[x] for x in ('col','line') if x in loc}

def parse_enum(f):
	""
	global rec2obj
	assert f['kind'] == 'EnumDecl'
	name = f.get('name')
	if name is None:
		assert f['id'] not in rec2obj
		rec2obj[f['id']] = f
	else:
		assert name not in rec2obj
		rec2obj[name] = f
	inner = f.get('inner')
	if inner is None:
		r = None
	else:
		r = []
		for decl in inner:
			assert decl['kind']=='EnumConstantDecl'
			val = chk_expr(decl['inner'][0]) if 'inner' in decl else None
			r.append([decl['name'], val])
	return r

def ty_rec_sz(d):
	""
	assert d['kind'] == 'RecordDecl' and d['tagUsed'] == 'struct'
	assert d['completeDefinition']
	sz = 0
	for f in d['inner']:
		if f['kind'] not in ('FieldDecl','RecordDecl'):
			assert f['kind'].endswith(('Attr','Comment')), f['kind']
			continue
		if f['kind'] == 'RecordDecl':
			assert 'type' not in f, f
		else:
			fty = f['type']
			assert 'qualType' in fty, fty
	return sz

def chk_typ(ty):
	"""
	const __global char *__private
	__private int
	void
	__global char *__private
	__local int *__private
	const __private int
	__global float *const __private
	"""
	global id2obj, name2obj, rec2obj, td2obj
	if 'attribute' in ty:
		_ = 2+2
	return

	is_pointer = False
	tl = ty.split()
	if tl[-1] == '__attribute__((cdecl))':
		tl = tl[:-1]
	
	td = {}
	i0 = 0
	while tl[i0] in ('const','__global','__local','__private'):
		i0 +=1
	i = i0 # i0 = i = 1 if tl[0] == 'const' else 0;
	while i < len(tl) and (tl[i].isidentifier()):
		td[tl[i]] = td.get(tl[i],0) + 1
		i += 1
	while i < len(tl) and tl[i] in ('*','*const','*__private','*restrict','*volatile','**','**restrict','(*)[11]','(*)[n]','(*)[20]','(*)[50]')+('***','*[1]','[]','[8]','[9]','[12]','[24]','[50]','[64]','[101]','[101][101]','[101][102]','[102][102]'):
		is_pointer = True
		i += 1
	if i < len(tl):
		assert tl[i].startswith('(*)('), ty # (*)(void), (*)(const void *, const void *)
		is_pointer = True
	#
	if 'struct' in td:
		assert 'union' not in td and 'struct' == tl[i0], tl
		sn = tl[i0+1]
		if sn not in name2obj:
			assert is_pointer
			return 'P'
		snd = name2obj[sn]
		assert snd['kind'] == 'RecordDecl' and snd['tagUsed'] == 'struct', tl
		assert len(tl) in (i0+2,i0+3), tl
		if len(tl) == i0+3:
			assert tl[-1] in ('*','*restrict','**'), tl
			return 'P'
		else:
			sz = ty_rec_sz(snd)
			return 'X' if sz <= 8 else 'Y'
	elif 'union' in td:
		assert 'union' == tl[i0], tl
		sn = tl[i0+1]
		snd = name2obj[sn]
		assert snd['kind'] == 'RecordDecl' and snd['tagUsed'] == 'union', tl
		assert False, ty
	else:
		_ = 2+2

def is_ocl_vect_ty(ty):
	""
	return ty.startswith(('float','char','int','short','uchar','uint','ushort')) and ty.endswith(('2','4','8','16'))

def parse_rec(e):
	""
	global rec2obj
	assert e['kind'] == "RecordDecl"
	assert e['tagUsed'] in ('struct','union'), e.get('tagUsed')
	r = [e['tagUsed']]
	name = e.get('name')
	if name is None:
		assert 'inner' in e, e
		assert e['id'] not in rec2obj
		rec2obj[e['id']] = e
	else:
		if name in rec2obj:
			assert rec2obj[name]['tagUsed'] == e['tagUsed']
			# assert 'inner' not in rec2obj[name], rec2obj[name]
		rec2obj[name] = e
	if 'completeDefinition' not in e:
		assert 'inner' not in e or (len(e['inner'])==1 and e['inner'][0]['kind']=='MaxFieldAlignmentAttr'), e
	else:
		for f in e['inner']:
			fn = f.get('name')
			fk = f['kind']
			fr = [fk,fn]
			if fk == 'EnumDecl':
				fr.append(parse_enum(f))
			elif fk == 'FieldDecl':
				assert ('name' in f) == ('isImplicit' not in f), f
				fty = f['type']
				assert 'qualType' in fty and 'inner' not in fty, fty
				fr.append(fty['qualType'])
			elif fk == 'IndirectFieldDecl':
				assert f['isImplicit'] and 'name' in f and 'type' not in f, f
			elif fk == 'RecordDecl':
				fr.append(parse_rec(f))
			else:
				assert fk in ('AlignedAttr','MaxFieldAlignmentAttr','FullComment'), f
			r.append(fr)
	return r

def is_expr(e):
	""
	return e['kind'].endswith(('Expr','Literal','Operator'))

types_int_base = ['char','int','long','short']
types_int_uxx = ['u'+x for x in types_int_base]
types_scal = types_int_base + types_int_uxx + ['double','float','half']
types_v2 = [x+'2' for x in types_scal]
types_v4 = [x+'4' for x in types_scal]
types_v8 = [x+'8' for x in types_scal]
types_v16 = [x+'16' for x in types_scal]
types_full = set(types_scal + types_v2 + types_v4 + types_v8 + types_v16)

def chk_expr(e, lvalue=False):
	""
	ek = e['kind']
	assert ek.endswith(('Expr','Literal','Operator'))
	r = [ek+'.'] # pour que ce ne soit pas un identifier
	inner = e.get('inner')
	e_ty = e['type']
	ty = e_ty['qualType']
	if 'desugaredQualType' in e_ty:
		dty = e_ty['desugaredQualType']
		if (ty,dty) not in (
				('__global uchar','__global unsigned char'),
				('__private float4','float __private __attribute__((ext_vector_type(4)))'),
				('const __global uchar','const __global unsigned char'),
				('float4','float __attribute__((ext_vector_type(4)))'),
				('int2','int __attribute__((ext_vector_type(2)))'),
				('size_t','unsigned long'),
				('uchar','unsigned char'),
				('uchar4','uchar __attribute__((ext_vector_type(4)))'), # uchar non expanse !!!
				('uint','unsigned int')):
			_ = 2+2
	for t in ty.split():
		if t in ('__global','__local','__private','const'):
			pass
		elif t in types_full:
			pass
		elif t in ('size_t','unsigned'):
			pass
		else:
			_ = 2+2
	if ek == 'ArraySubscriptExpr':
		[arr, idx] = inner
		A = chk_expr(arr, lvalue)
		I = chk_expr(idx)
		r = ['.[.]',A,I]
	elif ek == 'AsTypeExpr': # opencl-c.h:6401: #define as_float2(x) __builtin_astype((x),  float2)
		[e1] = inner
		E = chk_expr(e1)
		r.append(E)
	elif ek == 'BinaryOperator':
		[x,y] = inner
		oc = e['opcode'] # =, ',', *, ...
		X = chk_expr(x, lvalue) if oc != '=' else chk_expr(x, True)
		Y = chk_expr(y)
		assert not oc.isidentifier(), oc
		r = [oc, X, Y]
	elif ek == 'CallExpr':
		assert len(inner) >= 1
		fn = inner[0]
		assert fn['kind'] == 'ImplicitCastExpr'
		assert '(*)' in fn['type']['qualType']
		F = chk_expr(fn)
		r = ['.(...)',F]
		for x in inner[1:]:
			r.append(chk_expr(x))
	elif ek == 'CompoundAssignOperator':
		# r = chk_stmt(e)
		oc = e['opcode']
		assert oc in ('+=','-=','>>=', '<<=', '%=','*=', '/='), e
		[lv,e] = inner
		#assert lv['kind'] in ('ArraySubscriptExpr', 'DeclRefExpr'), lv
		L = chk_expr(lv, lvalue=True)
		V = chk_expr(e)
		assert not oc.isidentifier(), oc
		r = [oc, L, V]
	elif ek == 'CompoundLiteralExpr':
		[e1] = inner
		assert e1['kind'] == 'InitListExpr', e
		r.append(chk_expr(e1))
	elif ek == 'ConditionalOperator':
		[c,x,y] = inner
		C = chk_expr(c)
		X = chk_expr(x)
		Y = chk_expr(y)
		r = ['?:',C,X,Y]
	elif ek == 'ConstantExpr':
		[e1] = inner
		if e1['kind'] != 'IntegerLiteral':
			if e1['kind'] == 'ImplicitCastExpr':
				assert e1['castKind'] == 'IntegralCast', e1
				assert e1['inner'][0]['kind'] == 'IntegerLiteral', e1
		r.append(chk_expr(e1))
	elif ek == 'CStyleCastExpr':
		ck = e['castKind']
		r[0] = r[0]+ck
		[foo] = inner
		if ck == 'AddressSpaceConversion':
			assert ty[-2:] == ' *', ty
		elif ck == 'BitCast':
			assert ty[-2:] == ' *' or is_ocl_vect_ty(ty), ty
		elif ck == 'FloatingCast':
			assert ty == 'float', ty
			assert foo['kind'] == 'FloatingLiteral', foo
		elif ck == 'FloatingToIntegral':
			assert ty == 'int', ty
		elif ck == 'IntegralCast': # UnaryExprOrTypeTraitExpr-sizeof
			assert ty.endswith(('char','int','long','short')), ty
		elif ck == 'IntegralToFloating':
			assert ty == 'float', ty
		elif ck == 'NoOp':
			# ty: tout est possible
			pass
		elif ck == 'VectorSplat':
			assert is_ocl_vect_ty(ty), ty
		else:
			assert False, e
		r.append(chk_expr(foo))
	elif ek == 'CXXBoolLiteralExpr':
		assert ty == 'bool', e
		r = e['value']
		assert isinstance(r,bool)
	elif ek == 'DeclRefExpr':
		vd = e['referencedDecl']
		assert vd['kind'] in ('EnumConstantDecl','FunctionDecl','ParmVarDecl','VarDecl'), vd
		name = vd['name']
		r = name
	elif ek == 'ExtVectorElementExpr':
		[e1] = inner
		r.append(chk_expr(e1))
	elif ek == 'FloatingLiteral':
		assert ty in ('double','float'), e
		v = e['value']
		if ty == 'double':
			r = float(v)
		else:
			r.append(float(v))
	elif ek == 'ImplicitCastExpr':
		ck = e['castKind']
		r[0] = r[0]+ck
		[e1] = inner
		#ty = e['type']['qualType']
		if ck == 'AddressSpaceConversion':
			assert ty[-2:] == ' *', ty
		elif ck == 'ArrayToPointerDecay':
			assert ty[-1] in '*]', ty
		elif ck == 'FloatingCast':
			assert ty in ('double','float'), ty
		elif ck == 'FloatingToIntegral':
			assert ty in ('int',), ty
		elif ck == 'FunctionToPointerDecay':
			assert ' (*)(' in ty, ty
		elif ck == 'IntegralCast':
			assert ty in ('cl_mem_fence_flags', 'size_t', \
				 'char','int','short','uchar', 'uint','ushort', \
				 'unsigned int', 'unsigned long'), ty
		elif ck == 'IntegralToBoolean':
			assert ty == 'bool', ty
		elif ck == 'IntegralToFloating':
			assert ty == 'float', ty
		elif ck == 'IntToOCLSampler':
			assert ty == 'sampler_t', ty
		elif ck == 'LValueToRValue':
			pass # ty: tout est possible
		elif ck == 'NoOp':
			pass # ty: tout est possible
		elif ck == 'VectorSplat':
			assert is_ocl_vect_ty(ty), ty
		else: #  :   : 
			assert False, e
		r.append(chk_expr(e1))
	elif ek == 'InitListExpr':
		for x in inner:
			r.append(chk_expr(x))
	elif ek == 'IntegerLiteral':
		assert ty in ('int','unsigned int'), e
		v = e['value']
		if ty == 'int':
			r = int(v)
		else:
			r.append(int(v))
	elif ek == 'MemberExpr':
		isArrow = e['isArrow']
		r = ['->' if isArrow else '.']
		[e1] = inner
		r.append(chk_expr(e1))
		r.append({'name': e['name']})
	elif ek == 'OpaqueValueExpr':
		print("!!!!!!!!!!!!!!!!!!! WARNING : OpaqueValueExpr !!!!!!!!!!!!!!")
		assert False
	elif ek == 'ParenExpr':
		assert len(inner)==1
		E = chk_expr(inner[0], lvalue)
		r = ['(.)',E]
	elif ek == 'StmtExpr':
		[stmt] = inner
		r.append({'Stmt' : chk_stmt(stmt)})
	elif ek == 'StringLiteral': # non OpenCL
		assert False, e
	elif ek == 'UnaryExprOrTypeTraitExpr':
		assert e['name'] == 'sizeof', e
		assert ty == 'unsigned long', ty
		if inner is not None:
			assert 'argType' not in e
			[e1] = inner
			r = ['sizeof.expr', chk_expr(e1)]
		else:
			aty = e['argType']
			assert set(aty) in ({'qualType'},{'desugaredQualType','qualType','typeAliasDeclId'}), aty
			r = ['sizeof.type', {'argType' : aty['qualType']}]
	elif ek == 'UnaryOperator':
		oc = e['opcode']
		assert oc in ('*','++','-','!','&','~','+','--'), e
		assert len(inner) == 1
		E = chk_expr(inner[0], lvalue)
		r = [oc, E]
	else:
		assert False, ek
	if isinstance(r,list):
		assert isinstance(r[0], str) and not r[0].isidentifier(), r
		if isinstance(r[-1], dict):
			if 'type' in r[-1]:
				assert False, (e,r)
			r[-1]['type'] = ty
		else:
			r.append({'type':ty})
	elif isinstance(r,str):
		assert r.isidentifier(),r
	else:
		assert isinstance(r, (bool,float,int)), r
	return r

def chk_varDecl(x):
	""
	xk = x['kind']
	assert xk in ('VarDecl','ParmVarDecl')
	vn = x.get('name')
	if vn == 'lid':
		_ = 2+2
	ty = x['type']
	ty_id = ty.get('typeAliasDeclId')
	if ty_id:
		assert id2obj[ty_id]['kind'] == "TypedefDecl"
	vt = ty['qualType'] # ty.get('desugaredQualType',ty['qualType'])
	vi = None
	if 'inner' in x:
		assert x.get('init') in ('c',None), x # None : canny.cl : __local float mag[...]
		foo = is_expr(x['inner'][0])
		assert foo == ('init' in x), x
		if foo:
			vi = chk_expr(x['inner'][0])
		for y in x['inner'][foo:]:
			yk = y['kind']
			assert yk in ('FullComment','ParagraphComment','DLLImportAttr','OpenCLAccessAttr'), y
			if yk == 'OpenCLAccessAttr':
				assert set(y) == {'id','kind','range'}, x
				if all(z not in vt for z in ('__read_only', '__read_write', '__write_only')):
					assert False, x
	else:
		assert 'init' not in x, x
	if xk == 'VarDecl':
		assert vn is not None
		return [vn,[vt,vi]]
	else:
		assert vi is None
		return [vn,vt]
		

def chk_stmt(stmt):
	""
	sk = stmt['kind']
	r = [sk]
	inner = stmt.get('inner')
	if inner is None:
		inner = []
	else:
		assert len(inner) > 0, stmt
	stmt_loc = chk_loc(stmt)
	if sk.endswith('Stmt'):
		assert 'type' not in stmt, stmt
	else:
		assert 'type' in stmt, stmt
	if sk == 'AttributedStmt':
		assert all(x['kind'].endswith('Attr') for x in inner[:-1])
		stk = inner[-1]['kind']
		assert stk.endswith('Stmt')
		S = chk_stmt(inner[-1])
		if stk == 'ForStmt':
			assert len(inner)==2
			for x in inner[:-1]:
				assert x['kind'] == 'LoopHintAttr'
				assert x['inner'] == [{}] and x['implicit'] is True, x
		else:
			assert False, stk
		r.append(S)
# 	elif sk == 'BinaryOperator':
# 		oc = stmt['opcode']
# 		assert oc in ('=',','), stmt
# 		r = [oc]
# 		if oc == '=':
# 			[lv,e] = inner
# 			lvk = lv['kind']
# 			#assert lvk in ('ArraySubscriptExpr', 'DeclRefExpr','ParenExpr','UnaryOperator'), lv # desind, var, *
# 			L = chk_expr(lv, lvalue=True)
# 			V = chk_expr(e)
# 			r.extend([L,V])
# 		else:
# 			for x in inner:
# 				r.append(chk_stmt(x))
	elif sk == 'BreakStmt':
		assert set(stmt) == {'id','kind','range'}
# 	elif sk == 'CallExpr':
# 		r = chk_expr(stmt)
# 	elif sk == 'CompoundAssignOperator':
# 		oc = stmt['opcode']
# 		assert oc in ('+=','-=','>>=', '%=','*='), stmt
# 		[lv,e] = inner
# 		#assert lv['kind'] in ('ArraySubscriptExpr', 'DeclRefExpr'), lv
# 		L = chk_expr(lv, lvalue=True)
# 		V = chk_expr(e)
# 		r = [oc, L, V]
	elif sk == 'CompoundStmt':
		#r = [';']
		for st in inner:
			r.append(chk_stmt(st))
	elif sk == 'ContinueStmt':
		assert set(stmt) == {'id','kind','range'}
	elif sk == 'DeclStmt':
		for x in inner:
			_ = chk_loc(x)
			[vn,[vt,vi]] = chk_varDecl(x)
			r.append([vn,[vt,vi]])
	elif sk == 'DoStmt':
		[w_body,w_cond] = inner
		C = chk_expr(w_cond)
		B = chk_stmt(w_body)
		r.extend([B,C])
	elif sk == 'ForStmt':
		if len(inner) == 5:
			[start, strange, stop, step, body] = inner
			assert strange == {}, strange
			Start = chk_stmt(start) if start != {} else None
			Stop  = chk_expr(stop)  if stop  != {} else None
			Step  = chk_expr(step)  if step  != {} else None
			Body = chk_stmt(body)
			r.extend([Start,Stop,Step,Body])
		else:
			assert False, inner
	elif sk == 'IfStmt':
		if len(inner) == 2:
			[if_cond, if_then] = inner; if_else = None
		else:
			[if_cond, if_then, if_else] = inner
		If_cond = chk_expr(if_cond)
		If_then = chk_stmt(if_then)
		if if_else is not None:
			If_else = chk_stmt(if_else)
		else:
			If_else = None
		r.extend([If_cond, If_then, If_else])
# 	elif sk == 'ImplicitCastExpr': # lrn.cl : for(<<index>>; index < nthreads; index += tmp)
# 		r = chk_expr(stmt)
	elif sk == 'NullStmt':
		assert set(stmt) == {'id','kind','range'}
# 	elif sk == 'ParenExpr':
# 		r = chk_expr(stmt)
	elif sk == 'ReturnStmt':
		if 'inner' in stmt:
			assert len(stmt['inner']) == 1
			e = chk_expr(stmt['inner'][0])
			r.append(e)
	elif sk == 'SwitchStmt':
		[sw_cond,sw_body] = stmt['inner']
		r.append(chk_expr(sw_cond))
		assert sw_body['kind'] == 'CompoundStmt'
		for x in sw_body['inner']:
			xk = x['kind']
			# changement de structure
			if xk == 'CaseStmt':
				case_loc = loc_clean(chk_loc(x))
				[e,s] = x['inner']
				E = chk_expr(e)
				S = chk_stmt(s)
				r.append(['CaseStmt',E,case_loc])
				r.append(S)
			else:
				r.append(chk_stmt(x))
# 	elif sk == 'UnaryOperator':
# 		oc = stmt['opcode']
# 		assert oc in ('++',), oc
# 		r = chk_expr(stmt)
	elif sk == 'WhileStmt':
		[w_cond,w_body] = stmt['inner']
		C = chk_expr(w_cond)
		B = chk_stmt(w_body)
		r.extend([C,B])
	else:
		assert not sk.endswith('Stmt') # expression
		e = chk_expr(stmt)
		r = ['ExprStmt',e]
	assert r[0].endswith('Stmt')
	r.append({x:stmt_loc[x] for x in ('col','line') if x in stmt_loc})
	return r

def chk_loc(e):
	""
	ek = e['kind']
	loc = e.get('loc')
	assert (loc is not None) == (ek[-4:] == 'Decl'), e
	range_ = e['range']
	assert (loc == {}) == (range_ == {'begin': {}, 'end': {}}), (loc,range_)
	assert loc is None or loc == {} \
		or set(loc) == {'spellingLoc','expansionLoc'} \
		or (set(loc)&{'spellingLoc','expansionLoc'}==set() and set(loc) >= {'offset', 'col', 'tokLen'}), loc
	if loc is None:
		foo = range_['begin']
	elif set(loc) == {'spellingLoc','expansionLoc'}:
		## spellingLoc indique le #define, donc typiquement dans un .h
		foo = loc['expansionLoc']
	else:
		foo = loc 
	return foo # {'line': foo.get('line'), 'col': foo.get('col')}

def parse(filename, cpp_args="", mode_fast = True):
	"""
	"""
	global id2obj, name2obj, rec2obj, td2obj
	res = []
	path_list = [clang_path] + clang_options + cpp_args.split() + [filename]
	#path_list = ('clang -cl-std=CL2.0 -Xclang -ast-dump=json -fsyntax-only'
	#	' c:/opencv-4.5.1/sources/modules/imgproc/src/opencl/canny.cl').split()
	#text = check_output(path_list, universal_newlines=True)
	use_pipe = False
	use_tempfile = True
	print('COMPILING', path_list)
	if use_pipe:
		p = run(path_list, stdout=PIPE, stderr=PIPE, universal_newlines=True)
	else:
		if use_tempfile:
			fd = tempfile.TemporaryFile()
		else:
			fd = open('c:/Temp/dump.json','w')
		p = run(path_list, stdout=fd, stderr=PIPE, universal_newlines=True)
		if not use_tempfile:
			fd.close()
	stderr = p.stderr
	if p.returncode == 0: # or p.stdout or p.stderr:
		assert 'error:' not in stderr
	elif p.returncode == 1:
		assert 'error:' in stderr, stderr
		print('**************** CLANG ERROR **********************')
		assert False, stderr
	else:
		assert False, p.returncode
	print('READING json')
	if use_pipe:
		jss = p.stdout
		assert jss[0] == '{' and jss[-1] == '}'
		r = json.loads(jss)
		jss = "" # gc
	else:
		if use_tempfile:
			fd.seek(0)
		else:
			fd = open('c:/Temp/dump.json','r')
		r = json.load(fd)
		fd.close()
	if False:
		fd = open('c:/Temp/dump.json','w')
		fd.write(jss)
		fd.close()
	print('ANALYZING')
	###############################
	fdir, fname_c = os.path.split(filename)
	fn_c = filename
	chk_src = False
	#
	assert r['kind'] == "TranslationUnitDecl"
	# print(sorted(set(e['kind'] for e in r['inner'])))
	id2obj = {e['id']:e for e in r['inner']}
	name2obj = {e.get('name',''):e for e in r['inner']}   # le dernier objet concerne
	name2obj_c = {e.get('name',''):e for e in r['inner'] if \
			 e.get('loc',{}).get('file','').endswith('.c') or \
			 e.get('loc',{}).get('includedFrom',{}).get('file','').endswith('.c')}
	rec2obj_full = {e.get('name',''):e for e in r['inner'] if \
			 e['kind'] == "RecordDecl"}
	td2obj_full = {e.get('name',''):e for e in r['inner'] if \
			 e['kind'] == "TypedefDecl"}
	rec2obj = {}
	overload_rec_td = set(rec2obj_full) & set(td2obj_full)
	if overload_rec_td:
		pass # print(['!!! OVERLOAD !!! : ']+list(overload_rec_td)) # flt, ldbl, dbl, fp ; edgenode ...
	td2obj = {}
	syntax_error = False
	prev_e = None
	### apparemment le fichier courant loc['file'] n'est pas present quand il ne change pas
	### --> parsing_file le maintient en permanence
	parsing_file = ''
	file_ok_end = ('/'+fname_c,'\\'+fname_c)
	for e_idx, e in enumerate(r['inner']):
		# fichier courant, pour skip rapide
		loc = e['loc']
		prev_parsing_file = parsing_file
		if 'file' in loc:
			parsing_file = loc['file']
		elif 'expansionLoc' in loc:
			if 'file' in loc['expansionLoc']:
				parsing_file = loc['expansionLoc']['file']
			elif 'file' in loc['spellingLoc']:
				parsing_file = loc['spellingLoc']['file']
		if parsing_file != prev_parsing_file:
			print(f'parsing_file : {prev_parsing_file} -> {parsing_file}')
		if mode_fast and not parsing_file.endswith(('.c','.cl')):
			continue
		#
		decl_value = None
		decl_loc = chk_loc(e)
		k = e['kind']
		name = e.get('name')
		if name == 'mvn1':
			_ =2+2
		inner = e.get('inner')
		range_ = e['range']
		assert (loc == {}) == (range_ == {'begin': {}, 'end': {}})
		assert loc == {} \
			or set(loc) == {'spellingLoc','expansionLoc'} \
			or (set(loc)&{'spellingLoc','expansionLoc'}==set() and set(loc) >= {'offset', 'col', 'tokLen'}), loc # pas line
		
		builtin = loc == {} or parsing_file.endswith(('opencl-c.h','opencl-c-base.h'))
		file_ok = parsing_file.endswith(fname_c) # file_ok_end
		assert builtin != file_ok
		if mode_fast and builtin:
			continue # DEBUG
		if file_ok:
			if 'includedFrom' in loc: ### jeu tordu d'inclusions (.h -> .c)
				assert 'presumedFile' in loc and loc['presumedFile'].endswith(file_ok_end)
			assert parsing_file == fname_c or parsing_file.endswith(file_ok_end), parsing_file
			if set(loc) == {'spellingLoc','expansionLoc'}:
				# cas tordu du nom de proc defini par #define
				# ex : #define inp next_inp   puis  void inp() ...
				assert loc['expansionLoc']['file'].endswith(fname_c) if 'file' in loc['expansionLoc'] \
					else loc['spellingLoc']['file'].endswith(fname_c), e
				## spellingLoc indique le #define, donc typiquement dans un .h
				#decl_loc = {'line': loc['expansionLoc'].get('line'), 'col': loc['expansionLoc'].get('col')}
			else:
				pass #decl_loc = {'line': loc.get('line'), 'col': loc.get('col')}
		stcl = e.get('storageClass')
		assert stcl in (None, 'extern', 'static'), stcl
		inline = e.get('inline')
		assert inline in (None, True), inline # donc traitable comme un booleen
		isImplicit = e.get('isImplicit')
		assert isImplicit in (None, True), isImplicit # donc traitable comme un booleen
		#
		if k == "EmptyDecl":
			#print(k)
			assert set(e) == {'id','kind','loc','range'}, e
		elif k == "EnumDecl":
			decl_value = parse_enum(e)
		elif k == "FunctionDecl": # name: ..., type: {'qualType':'int (int)'}, inner
			#print('{} {}'.format(k, name))
			f_ty = e['type']
			ty_id = f_ty.get('typeAliasDeclId')
			if ty_id:
				assert id2obj[ty_id]['kind'] == "TypedefDecl"
# 			anr = ' __attribute__((noreturn))'
# 			acd = ' __attribute__((cdecl))'
# 			#avs = '__attribute__((__vector_size__(2 * sizeof(int)))) '
# 			f_ty['qualType'] = f_ty['qualType'].replace(anr,'').replace(acd,'')#.replace(avs,'')
# 			if '__attribute__' in f_ty['qualType']:
# 				assert '__attribute__((__vector_size__' in f_ty['qualType'], f_ty['qualType']
# 				assert name.startswith('__builtin_ia32_'), name
# 				#print(name)
# 				continue
			#file_ok = ('file' not in loc and 'includedFrom' not in loc) or ('file' in loc and loc['file'].endswith('.c'))
			file_locs = None
			#
			body_stmt = r_ty = pl = None; attr_l = [] # les return-values
			has_body = False
			is_kernel = False
			if inner is None:
				nbp = 0
			else:
				nbp = sum(x['kind']=='ParmVarDecl' for x in inner)
				assert all(x['kind']=='ParmVarDecl' for x in inner[:nbp])
				if nbp < len(inner):
					has_body = inner[nbp]['kind'].endswith('Stmt')
					assert all(x['kind'].endswith(('Attr','Comment')) for x in inner[nbp+has_body:])
					attr_l = [x['kind'] for x in inner[nbp+has_body:] if x['kind'].endswith('Attr')]
					is_kernel = any(x['kind'] == 'OpenCLKernelAttr' for x in inner[nbp+has_body:])
			#
			if isImplicit:
				assert not has_body
				assert stcl == 'extern', stcl
				stcl = 'implicit'
				if inner is not None:
					assert len(inner) >= 1
					assert all(x['loc']=={} and 'name' not in x for x in inner[:nbp]), inner
					assert all(x['kind'].endswith('Attr') \
						and ('loc' not in x or x['loc']=={}) for x in inner[nbp:]), inner
					if nbp >= 1:
						pl = [('',x['type']['qualType']) for x in inner[:nbp]]
						ps0 = ', '.join(t for _,t in pl)
						ps = '('+ps0+')'
						if not f_ty['qualType'].endswith(ps):
							ps = '('+ps0+', ...)'
					else:
						pl = []; ps = '(void)'
				else:
					pl = []; ps = '(void)'
				if not f_ty['qualType'].endswith(ps):
					assert ps == '(void)' and f_ty['qualType'] == 'void ()' and name.startswith(('__sync_','__builtin_')), (ps,f_ty['qualType'],name)
					#print(name)
					assert not file_ok
					continue
				r_ty = f_ty['qualType'][:-len(ps)]
				if r_ty[-1] != '*':
					assert r_ty[-1] == ' ' and r_ty[-2] != ' ', r_ty
					r_ty = r_ty[:-1]
				chk_typ(r_ty)
			else:
				#print('\t'+name)
				if file_ok:
#					if not loc.get('file',fname_c).endswith(fname_c):
#						continue
					if 'line' in loc:
						f_line,f_col, f_tokLen = loc['line'], loc['col'], loc['tokLen']
					else:
						# on a affaire au nom de la macro
						assert set(loc) == {'expansionLoc','spellingLoc'}, e
						print('*** nom MACRO-defini : '+name)
						f_line,f_col, f_tokLen = loc['expansionLoc']['line'], loc['expansionLoc']['col'], loc['expansionLoc']['tokLen']
					prec_line = f_line; prec_col = f_col
					if chk_src:
						assert f_line-1 < len(src)
						assert f_col-1+f_tokLen <= len(src[f_line-1])
						if 'line' in loc:
							assert src[f_line-1][f_col-1:f_col-1+f_tokLen] == name, (loc, src[f_line-1], name)
						else:
							pass ### c'est le nom de la macro qui est dans le .c
					#
					loc_beg = e['range']['begin'] # donne le 1ier token,par exemple 'int'
					if 'col' not in loc_beg:
						assert set(loc_beg) == {'expansionLoc','spellingLoc'}
						loc_exp = loc_beg['expansionLoc']
						a_line,a_col, a_tokLen = loc_exp['line'], loc_exp['col'], loc_exp['tokLen']
					else:
						a_line,a_col, a_tokLen = loc_beg.get('line',f_line), loc_beg['col'], loc_beg['tokLen']
					if not( a_line == f_line and a_col <= f_col ): # fft.cl ligne 23
						assert a_line < f_line and a_col <= f_col, (a_line, f_line, a_col, f_col)
					loc_end = e['range']['end']
					z_line,z_col, z_tokLen = loc_end.get('line',f_line), loc_end.get('col',-1), loc_end.get('tokLen',-1)
					if z_tokLen == 0:
						syntax_error = True
						print('\t!!!!!!!! syntax error !!!!!!!!')
						continue
					if chk_src:
						if not src[z_line-1][z_col-1:z_col-1+z_tokLen] in ('}',')'):
							assert False,  src[z_line-1]
				# ) pour les declarations
				#nbp = 0
				if inner is None:
					inner = []
				pl = [] # INV : nbp == len(pl)
				for p in inner[:nbp]:
					_ = chk_loc(p)
					_ = chk_varDecl(p)
					p_ty = p['type']
					ty_id = p_ty.get('typeAliasDeclId')
					if ty_id:
						assert id2obj[ty_id]['kind'] == "TypedefDecl"
					chk_typ(p_ty['qualType'])
					if file_ok:
						p_line, p_col, p_tokLen = p['loc'].get('line',prec_line), \
												  p['loc'].get('col', prec_col), \
												  p['loc'].get('tokLen',None)
						if chk_src and 'name' in p:
							if not src[p_line-1][p_col-1:p_col-1+p_tokLen] == p['name']:
								p_line += 1
								assert src[p_line-1][p_col-1:p_col-1+p_tokLen] == p['name']
						prec_line = p_line; prec_col = f_col
					pc = (p.get('name'),p_ty['qualType'])
					pl.append(pc)
					#nbp += 1
				if nbp:
					ps = ', '.join(t for _,t in pl)
					ps = '('+ps+')'
				elif f_ty['qualType'].endswith('()'):
					ps = '()'
				else:
					ps = '(void)'
				if f_ty['qualType'].endswith(ps): # , (f_ty['qualType'],ps)
					r_ty = f_ty['qualType'][:-len(ps)]
				else:
					assert f_ty['qualType'][-1] == ')'
					idx = f_ty['qualType'].index('(')
					idx1 = f_ty['qualType'].find('(*)')
					assert idx1==-1 or idx1> idx, f_ty['qualType']
					params = f_ty['qualType'][idx+1:-1]
					if '(*)(' in params:
						pass
					elif ', ...' in params:
						assert params.count(',') == len(pl), f_ty['qualType']
					else:
						assert params.count(',') == len(pl)-1, f_ty['qualType']
					r_ty = f_ty['qualType'][:idx]
				if r_ty[-1] != '*':
					assert r_ty[-1] == ' ' and r_ty[-2] != ' ', r_ty
					r_ty = r_ty[:-1]
				chk_typ(r_ty)
				#
				if nbp < len(inner):
					if has_body: # inner[nbp]['kind'] == 'CompoundStmt':  ### aussi dans les .h et .inl
						stmt = inner[nbp]
						if file_ok:
							assert 'file' not in loc or loc['file'].endswith(fname_c), (fn_c,loc['file'])
							if chk_src:
								assert src[z_line-1][z_col-1:z_col-1+z_tokLen] == '}'
							#
							assert 'loc' not in stmt
							s_range = stmt['range']
							s_beg = s_range['begin']
							s_line, s_col, s_tokLen = s_beg.get('line',prec_line), s_beg.get('col'), s_beg.get('tokLen')
							if chk_src:
								assert src[s_line-1][s_col-1:s_col-1+s_tokLen] == '{'
							#
							s_end = s_range['end']
							assert s_end == loc_end
							file_locs = [a_line,a_col, s_line,s_col, z_line, z_col]
						body_stmt = chk_stmt(stmt)
						#print(foo)
						#decl_value = foo
					else:
						k1 = [inn['kind'] for inn in inner[nbp:]]
						if not all(k != 'CompoundStmt' for k in k1):
							assert False
				assert all(inn['kind'] in ('AlwaysInlineAttr','CompoundStmt','ConstAttr', 'ConvergentAttr', \
			   'DeprecatedAttr','DLLImportAttr','FormatAttr','FullComment','GNUInlineAttr', \
			   'MinVectorWidthAttr','NoDebugAttr','NonNullAttr','NoThrowAttr','PureAttr', \
			   'ReturnsTwiceAttr','TargetAttr','UnusedAttr', \
			   'OpenCLIntelReqdSubGroupSizeAttr','OpenCLKernelAttr','OverloadableAttr','ReqdWorkGroupSizeAttr') \
			   for inn in inner[nbp:]), [inn['kind'] for inn in inner[nbp:]]
			#info = [name, stcl, r_ty, pl, file_locs]
			assert has_body == (body_stmt is not None)
			decl_value = [r_ty, pl, body_stmt, attr_l]
			#res.append(info)

		elif k == "RecordDecl": ## struct foo; mais peut aussi etre anonyme
			#print('{} {}'.format(k, name or '?????????????????'))
			decl_value = parse_rec(e)
			
		elif k == "TypedefDecl": # name: ..., type: {'qualType':'...'}
			assert name
			if name in td2obj:
				if td2obj[name]['type'] != e['type']:
					if 'desugaredQualType' in td2obj[name]['type'] and 'desugaredQualType' in e['type']:
						assert td2obj[name]['type']['desugaredQualType'] == e['type']['desugaredQualType'], (td2obj[name]['type'] , e['type'])
					else:
						assert td2obj[name]['type']['qualType'] == e['type']['desugaredQualType'], (td2obj[name]['type'] , e['type'])
				old_decl = td2obj[name]
			else:
				old_decl = None
			td2obj[name] = e
			if len(inner) != 1:
				assert all(x['kind'].endswith(('Attr','Comment')) for x in inner[1:]), [x['kind'] for x in inner[1:]]
			tk = inner[0]['kind'] # BuiltinType ElaboratedType PointerType RecordType TypedefType
			assert tk.endswith('Type'), tk
			assert tk[:-4] in ('Atomic','Attributed', 'Builtin', 'Complex', 'ConstantArray', \
					  'Elaborated', 'ExtVector', 'FunctionProto', 'Paren', 'Pointer', 'Qual', \
					  'Record', 'Typedef','Vector'), tk
			tqt = e['type']['qualType']
			assert tqt == inner[0]['type']['qualType'] ## mais pas e['type'] == inner[0]['type'] ; e['type'] plus gros
			decl_value = [tk, tqt]
			if tk == 'AtomicType': #
				i0tq = inner[0]['type']['qualType'] # '_Atomic(int)'
				i0i0tq = inner[0]['inner'][0]['type']['qualType'] # 'int'
				assert i0tq == f'_Atomic({i0i0tq})', (i0tq,i0i0tq)
			elif tk == 'AttributedType':
				assert not ctx_OpenCL
			elif tk == 'BuiltinType': #
				tqtl = tqt.split()
				assert all(t in ('__int128','char','int','long','short','signed','unsigned') for t in tqtl) or \
					tqt in ('double','float','void') or \
					tqt in ('clk_event_t','event_t','queue_t','reserve_id_t','sampler_t') or \
					(tqt.startswith('intel_sub_group_avc_') and tqt.endswith('_t')), tqt
			elif tk == 'ComplexType':
				assert not ctx_OpenCL
			elif tk == 'ConstantArrayType':
				assert not ctx_OpenCL
			elif tk == 'ElaboratedType':
				ttag, tname = tqt.split()
				assert tqt.startswith(('enum', 'struct ','union ')) and len(tqt.split()) == 2 and tqt.split()[1].isidentifier(), tqt
				if 'ownedTagDecl' in inner[0]:
					assert set(inner[0]['ownedTagDecl']) == {'id','kind','name'}
					assert inner[0]['ownedTagDecl']['kind'] in ('EnumDecl','RecordDecl'), inner[0]['ownedTagDecl']['kind']
					assert inner[0]['ownedTagDecl']['name'] in ('',tname)
				else:
					_ = 2+2
				if name == tname:
					if tname not in rec2obj: # rec2obj_full:
						# typedef struct /* ANON */ { ... } name;
						assert inner[0]['ownedTagDecl']['id'] in rec2obj, inner[0]['ownedTagDecl']['id']
						assert inner[0]['ownedTagDecl']['name'] == ''
					else:
						# typedef struct name {... } name;
						""" skiena/netflow :
typedef struct {
	int v;				/* neighboring vertex */
	int capacity;			/* capacity of edge */
	int flow;			/* flow through edge */
	int residual;			/* residual capacity of edge */
	struct edgenode *next;          /* next edge in list */
} edgenode;
						"""
						# inner[0]['ownedTagDecl']['name'] != '' ou pas
						assert 'tagUsed' not in rec2obj[tname] or ttag == rec2obj[tname]['tagUsed'], (ttag , rec2obj[tname]['tagUsed'])
				else:
					# 
					pass # assert ttag == rec2obj[tname].get('tagUsed','struct')
			elif tk == 'ExtVectorType':
				i0i0k = inner[0]['inner'][0]['kind']
				assert i0i0k in ('BuiltinType','TypedefType'), i0i0k
			elif tk == 'FunctionProtoType':
				assert not ctx_OpenCL
			elif tk == 'ParenType':
				assert not ctx_OpenCL
			elif tk == 'PointerType': #
				i0i0k = inner[0]['inner'][0]['kind']
				assert i0i0k in ('BuiltinType','QualType'), i0i0k
			elif tk == 'QualType':
				assert not ctx_OpenCL
			elif tk == 'RecordType': #
				assert name == '__NSConstantString' and tqt == 'struct __NSConstantString_tag'
			elif tk == 'TypedefType':
				#assert not ctx_OpenCL
				assert tqt.isidentifier(), tqt
				assert name != tqt
				assert tqt in td2obj
			elif tk == 'VectorType':
				assert not ctx_OpenCL
			else:
				assert False, tk
		elif k == "VarDecl":
			_, vt_vi = chk_varDecl(e)
			decl_value = vt_vi
			#
# 			assert name
# 			ty = e['type']
# 			ty_id = ty.get('typeAliasDeclId')
# 			if ty_id:
# 				assert id2obj[ty_id]['kind'] == "TypedefDecl"
# 			assert '__attribute__' not in ty['qualType'], ty['qualType']
# 			r_ty = ty.get('desugaredQualType',ty['qualType'])
# 			info = [name, stcl, r_ty]
# 			res.append(info)
# 			if inner:
# 				assert all(inn['kind'] in ('BinaryOperator','DLLImportAttr', \
# 				   'FloatingLiteral','FullComment',\
# 				   'ImplicitCastExpr','InitListExpr','IntegerLiteral',\
# 				   'ParagraphComment','StringLiteral','UnaryOperator') for inn in inner), inner
		else:
			assert False, k
		decl_loc = {x:decl_loc[x] for x in ('col','line') if x in decl_loc}
		resultat = [k,name, decl_value, decl_loc]
		if not builtin:
			print(resultat[:2])
		res.append(resultat)
		##
		prev_e = e
	######################
	return res, stderr

if __name__ == "__main__":
	idx = 1
	if idx == 0:
		filename = "imgproc/src/opencl/histogram.cl"
		cpp_args = " -DBINS=256 -DHISTS_COUNT=16 -DWGS=1"
		# cpp_args += " -Dbarrier(x)=__barrier__(x)"
		#cpp_args += " -include opencl-c.h"
		filename = r'c:/opencv-4.5.1/sources/modules/' + filename
	elif idx == 1:
		filename = "cl_parse_test1.cl"
		cpp_args = ""
	js, stderr = parse(filename, cpp_args, mode_fast=False)
	print(js[-1])