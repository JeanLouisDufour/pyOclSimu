#-------------------------------------------------------------------------------
# pycparser: using_gcc_E_libc.py
#
# Similar to the using_cpp_libc.py example, but uses 'gcc -E' instead
# of 'cpp'. The same can be achieved with Clang instead of gcc. If you have
# Clang installed, simply replace 'gcc' with 'clang' here.
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#-------------------------------------------------------------------------------
import sys

# This is not required if you've installed pycparser into
# your site-packages/ with setup.py
#
#sys.path.extend(['.', '..'])

from c_file import parse_file
from subprocess import check_output, run, PIPE

if __name__ == "__main__":
	if len(sys.argv) > 1:
		filename  = sys.argv[1]
	else:
		filename = r'c_files/simplemain.c'
		filename = r'c:/opencv-4.5.1/sources/modules/imgproc/src/opencl/clahe.cl'

ocv_k_l = [
# 	"core/src/opencl/arithm.cl -DDEPTH_dst=3 -DBINARY_OP -DOP_ADD -DdstT=int     -DrowsPerWI=3",
# 	"core/src/opencl/convert.cl -DNO_SCALE -DsrcT=int -DdstT=float               -DconvertToDT=convert_float",
# 	"core/src/opencl/copymakeborder.cl -Dcn=1 -DBORDER_CONSTANT -DST=int -DT=int      -DrowsPerWI=3",
# 	"core/src/opencl/copyset.cl -DdstT=int             -DdstT1=int -Dcn=1 -DrowsPerWI=3",
# #?	"core/src/opencl/cvtclr_dx.cl",
# 	"core/src/opencl/fft.cl -DCT=float2 -DFT=float      -DLOCAL_SIZE=36 -Dkercn=1 -DRADIX_PROCESS=",
# 	"core/src/opencl/flip.cl -Dkercn=1 -DT=uchar            -DPIX_PER_WI_Y=3",
# 	"core/src/opencl/gemm.cl -DT1=float -DT=float -DWT=float        -DLOCAL_SIZE=36",
# 	"core/src/opencl/halfconvert.cl -DsrcT=half -DdstT=float         -DrowsPerWI=17",
# 	"core/src/opencl/inrange.cl -Dcn=1 -DdstT=int -DsrcT1=int       -Dkercn=1 -DcolsPerWI=49",
# 	"core/src/opencl/intel_gemm.cl",
# 	"core/src/opencl/lut.cl -Dlcn=1 -Ddcn=1 -DdstT=int             -DsrcT=int",
# 	"core/src/opencl/meanstddev.cl -DsrcT=float -DdstT=float -DsqdstT=float"
# 		"        -DWGS2_ALIGNED=2 -DWGS=3"
# 		" -DconvertToDT=convert_float -DconvertToSDT=convert_float",
# 	"core/src/opencl/minmaxloc.cl -DdstT=int -DsrcT=int"
# 		"        -DMINMAX_STRUCT_ALIGNMENT=3 -Dkercn=1 -DWGS=3 -DsrcT1=int"
# 		" -DWGS2_ALIGNED=2 -DconvertToDT=convert_int",
# 	"core/src/opencl/mixchannels.cl -DT=int"
# 		" -DDECLARE_INPUT_MAT_N=DECLARE_INPUT_MAT(1)"
# 		" -DDECLARE_OUTPUT_MAT_N=DECLARE_OUTPUT_MAT(1)"
# 		" -DDECLARE_INDEX_N=DECLARE_INDEX(1)"
# 		" -DPROCESS_ELEM_N=PROCESS_ELEM(1)"
# 		"        -Dscn1=2 -Ddcn1=2",
# 	"core/src/opencl/mulspectrums.cl",
# 	"core/src/opencl/normalize.cl -DworkT=float -DsrcT=float -DdstT=float"
# 		"        -DrowsPerWI=3 -DconvertToWT=convert_float -DconvertToDT=convert_float",
# 	"core/src/opencl/reduce.cl -DsrcT=int -DOP_COUNT_NON_ZERO -DsrcT1=int"
# 		"        -DWGS2_ALIGNED=2 -DWGS=3",
# 	"core/src/opencl/reduce2.cl -DOCL_CV_REDUCE_SUM -DdstT0=int -DdstT=int -DsrcT=int"
# 		"        -Dcn=1 -DconvertToDT=convert_int -DconvertToDT0=convert_int",
# 	"core/src/opencl/repeat.cl -DT=int           -DrowsPerWI=3 -Dnx=4 -Dny=5",
# 	"core/src/opencl/set_identity.cl -DST=int -DT=int           -DrowsPerWI=3",
# 	"core/src/opencl/split_merge.cl -DOP_MERGE -DT=int"
# 		" -DDECLARE_SRC_PARAMS_N=DECLARE_SRC_PARAM(1)"
# 		" -DDECLARE_INDEX_N=DECLARE_INDEX(1)"
# 		" -DPROCESS_ELEMS_N=PROCESS_ELEM(1)"
# 		"         -Dcn=1 -Dscn1=3",
# 	"core/src/opencl/transpose.cl -DT=int         -DTILE_DIM=4 -DBLOCK_ROWS=2",
	
# 	"dnn/src/opencl/activations.cl -DT=float        -Dconvert_T=convert_float",
# 	"dnn/src/opencl/batchnorm.cl -DNUM=1 -DDtype=int      -Dconvert_T=convert_int",
# 	"dnn/src/opencl/col2im.cl -DT=int"
# 		"         -DKERNEL_H=2 -DKERNEL_W=2 -DPAD_H=3 -DPAD_W=3 -DSTRIDE_H=7 -DSTRIDE_W=7",
# 	"dnn/src/opencl/concat.cl -DDtype=int",
# 	"dnn/src/opencl/conv_layer_spatial.cl",
# 	"dnn/src/opencl/conv_spatial_helper.cl -DDtype=int",
# 	"dnn/src/opencl/detection_output.cl",
# 	"dnn/src/opencl/dummy.cl",
# 	"dnn/src/opencl/eltwise.cl -DDtype=float -DDtype4=float4",
# 	"dnn/src/opencl/gemm_buffer.cl -DDtype=float -DDtype8=float8 -DDtype4=float4",
# 	"dnn/src/opencl/gemm_image.cl",
# 	"dnn/src/opencl/im2col.cl -DT=int",
# 	"dnn/src/opencl/lrn.cl -DT=float",
# 	"dnn/src/opencl/math.cl -DDtype=int            -Dconvert_Dtype=convert_int",
# 	"dnn/src/opencl/matvec_mul.cl -DDtype=float -DDtype4=float4        -Dconvert_Dtype=convert_float",
# 	"dnn/src/opencl/mvn.cl -DNUM=1 -DKERNEL_MVN",
# 	"dnn/src/opencl/ocl4dnn_lrn.cl -DDtype=float",
# 	"dnn/src/opencl/ocl4dnn_pooling.cl",
# 	"dnn/src/opencl/permute.cl -DDtype=int",
# 	"dnn/src/opencl/pooling.cl -DT=float",
# 	"dnn/src/opencl/prior_box.cl -DDtype=float -DDtype4=float4      -Dconvert_T=convert_float4",
# 	"dnn/src/opencl/region.cl",
# 	"dnn/src/opencl/slice.cl -DWSZ=1 -DDIMS=1 -DELEMSIZE=1 -DSLICE_KERNEL_SUFFIX=0"
# 		"         -DBLOCK_SIZE=2 -DSRC_START_0=1 -DSRC_STEP_0=1"
# 		" -DBLOCK_COLS=10 -DBLOCK_ROWS=10 -DBLOCK_SRC_STRIDE=1",
# 	"dnn/src/opencl/softmax.cl -DT=float",
# 	"dnn/src/opencl/softmax_loss.cl -DDtype=float        -DDTYPE_MAX=FLT_MAX",
	
#  	"imgproc/src/opencl/accumulate.cl -DACCUMULATE -DsrcT1=int -DdstT1=int"
# 		 " -DrowsPerWI=36 -Dcn=1 -DconvertToDT=convert_int",
#  	"imgproc/src/opencl/bilateral.cl -Dcn=1 -Dfloat_t=float -Dint_t=int -Duchar_t=uchar"
# 		 " -Dradius=10 -Dconvert_int_t=convert_int -Dmaxk=10 -Dgauss_color_coeff=2"
# 		 " -Dconvert_float_t=convert_float -Dconvert_uchar_t=convert_uchar",
#  	"imgproc/src/opencl/blend_linear.cl -DT=int"
# 		 " -Dcn=1 -DconvertToT=convert_int",
#  	"imgproc/src/opencl/boxFilter.cl -DBORDER_CONSTANT -DWT=int -DST=int -DDT=int"
# 		 " -DconvertToWT=convert_int -DconvertToDT=convert_int"
# 		 " -DKERNEL_SIZE_X=3 -DLOCAL_SIZE_X=5 -DANCHOR_X=4"
# 		 " -DKERNEL_SIZE_Y=3 -DLOCAL_SIZE_Y=5 -DANCHOR_Y=4 -DBLOCK_SIZE_Y=6",
#  	"imgproc/src/opencl/boxFilter3x3.cl",
#  	"imgproc/src/opencl/calc_back_project.cl -Dhistdims=1"
# 		 " -Dscn=1",
#  	"imgproc/src/opencl/canny.cl",
#  	"imgproc/src/opencl/clahe.cl",
#  	"imgproc/src/opencl/color_hsv.cl"
# 		 " -DPIX_PER_WI_Y=2 -Dscn=3 -Ddcn=3 -Dbidx=0x2",
#  	"imgproc/src/opencl/color_lab.cl"
# 		 " -DPIX_PER_WI_Y=2 -Dscn=3 -Ddcn=3",
#  	"imgproc/src/opencl/color_rgb.cl -Ddepth=0"
# 		 " -DPIX_PER_WI_Y=2 -Dscn=3 -Ddcn=3 -Dbidx=0x2",
#  	"imgproc/src/opencl/color_yuv.cl"
# 		 " -DPIX_PER_WI_Y=2 -Dscn=3 -Ddcn=3 -Dbidx=0x2",
#  	"imgproc/src/opencl/corner.cl -DBORDER_CONSTANT -DCORNER_HARRIS"
# 		 " -DksX=1 -DanX=1 -DksY=1 -DanY=1",
#  	"imgproc/src/opencl/covardata.cl -DBORDER_CONSTANT -DSRCTYPE=int"
# 		 " -DBLK_X=1 -DBLK_Y=1",
#  	"imgproc/src/opencl/filter2D.cl -DBORDER_CONSTANT -DWT1=int -DWT=int -DdstT=int -DsrcT=int"
# 		 " -DANCHOR_X=1 -DANCHOR_Y=1 -DCOEFF=2 -DKERNEL_SIZE_X=3 -DKERNEL_SIZE_Y=4 -DLOCAL_SIZE=5"
# 		 " -DKERNEL_SIZE_Y2_ALIGNED=1 -DconvertToDstT=convert_int -DconvertToWT=convert_int",
# # filter2DSmall : l'expansion de LOOP(...) donne (({...})) NON SUPPORTE
	"imgproc/src/opencl/filter2DSmall.cl -DBORDER_CONSTANT -DWT1=float -DWT=float"
		" -DsrcT1=int -DsrcT=int -DPX_LOAD_VEC_SIZE=2 -DPX_LOAD_X_ITERATIONS=2 -DPX_LOAD_Y_ITERATIONS=2"
		"    "
		" -DconvertToWT=convert_float -DdstT=int -DconvertToDstT=convert_int"
		" -DPX_PER_WI_X=2 -DKERNEL_SIZE_X=2 -DCOEFF=1 -DPX_LOAD_NUM_PX=2"
		" -DPX_PER_WI_Y=2 -DKERNEL_SIZE_Y=2"
		" -DPRIV_DATA_WIDTH=6 -DANCHOR_X=1 -DANCHOR_Y=1",
	"imgproc/src/opencl/filterSep_singlePass.cl -DBORDER_CONSTANT -DWT1=int -DWT=int -DsrcT=int -DdstT=int"
		" -DconvertToWT=convert_int",
	"imgproc/src/opencl/filterSepCol.cl -DsrcT1=int -DsrcT=int -DdstT=int",
	"imgproc/src/opencl/filterSepRow.cl -DdstT1=int -DsrcT=int -DdstT=int",
# filterSmall : l'expansion de LOOP(...) donne (({...})) NON SUPPORTE
	"imgproc/src/opencl/filterSmall.cl -DBORDER_REPLICATE -DOP_ERODE -DWT1=int -DWT=int -DsrcT1=int -DsrcT=int -DPX_LOAD_VEC_SIZE=2 -DPX_LOAD_X_ITERATIONS=2 -DPX_LOAD_Y_ITERATIONS=2",
 	"imgproc/src/opencl/gaussianBlur3x3.cl",
 	"imgproc/src/opencl/gaussianBlur5x5.cl",
 	"imgproc/src/opencl/gftt.cl",
 	"imgproc/src/opencl/histogram.cl",
 	"imgproc/src/opencl/hough_lines.cl",
 	"imgproc/src/opencl/integral_sum.cl -DsumT=int",
 	"imgproc/src/opencl/laplacian3.cl",
 	"imgproc/src/opencl/laplacian5.cl -DBORDER_CONSTANT -DWT1=int -DWT=int -DsrcT=int -DdstT=int",
 	"imgproc/src/opencl/linearPolar.cl",
 	"imgproc/src/opencl/logPolar.cl",
 	"imgproc/src/opencl/match_template.cl -Dcn=1 -DWT=int",
 	"imgproc/src/opencl/medianFilter.cl -Dcn=1 -DT=int",
 	"imgproc/src/opencl/moments.cl -DTILE_SIZE=32",
 	"imgproc/src/opencl/morph.cl -DOP_ERODE -DT=int",
 	"imgproc/src/opencl/morph3x3.cl",
 	"imgproc/src/opencl/precornerdetect.cl",
 	"imgproc/src/opencl/pyr_down.cl -DBORDER_REPLICATE -DFT=float -Dkercn=1 -DT=float",
	"imgproc/src/opencl/pyr_up.cl -DFT=float -DT=float",
	"imgproc/src/opencl/pyramid_up.cl",
	"imgproc/src/opencl/remap.cl -DBORDER_CONSTANT",
	"imgproc/src/opencl/resize.cl",
	"imgproc/src/opencl/sepFilter3x3.cl",
	"imgproc/src/opencl/threshold.cl -DT1=int -DT=int",
	"imgproc/src/opencl/warp_affine.cl",
	"imgproc/src/opencl/warp_perspective.cl",
	"imgproc/src/opencl/warp_transform.cl -DST=int",
	
	]

gcc_path = r'c:/mingw64/bin/gcc.exe'
clang_path = r'c:/Program Files/LLVM/bin/clang.exe'
clang_options = '-cl-std=CL2.0 -c -Xclang -finclude-default-header -target spir64 -O0 -emit-llvm -o c:/Temp/tmp.bc'.split()

for s in ocv_k_l:
	if s.startswith(('core__','dnn__')) or '/gemm_INH' in s: continue
	print('*'*64)
	print(s)
	sp_idx = s.find(' ')
	if sp_idx < 0:
		filename = s
		cpp_args = ''
	else:
		filename = s[:sp_idx]
		cpp_args = s[sp_idx:]
	filename = r'c:/opencv-4.5.1/sources/modules/' + filename
	if False:
		ast = parse_file(filename, use_cpp=True,
            cpp_path=gcc_path,
            cpp_args=['-E', r'-Iutils/fake_libc_include'] + ['-x','c','-D__attribute__(x)='] + cpp_args.split())
		#	ast.show()
	else:
		path_list = [clang_path] + clang_options + cpp_args.split() + [filename]
		#text = check_output(path_list, universal_newlines=True)
		p = run(path_list, stdout=PIPE, stderr=PIPE, universal_newlines=True)
		if p.returncode or p.stdout or p.stderr:
			print('!!!!!!!! WARNING !!!!!!!!!')
			print('return code',p.returncode)
			print('stdout:\n',p.stdout)
			print('stderr:\n',p.stderr)
			print('!'*64)
			if p.returncode == 0:
				assert 'error:' not in p.stderr and p.stdout == ''
			else:
				assert False, filename

# clang -c -Xclang -finclude-default-header  -target spir64 -O0 -emit-llvm -o test.bc c:/opencv-4.5.1/sources/modules/imgproc/src/opencl/clahe.cl
# https://webdevdesigner.com/q/how-to-use-clang-with-mingw-w64-headers-on-windows-18121/
# for i in */*.c; do clang -Xclang -ast-dump=json -fsyntax-only -target x86_64-pc-windows-gnu $i > ${i%.c}.clang_ast.json; done

