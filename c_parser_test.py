from c_parser import CParser, Coord, ParseError

cpar = CParser(
                lex_optimize=False,
                yacc_debug=True,
                yacc_optimize=False,
                yacctab='yacctab')

#cpar._add_typedef_name('uchar', None)

#s = 'typedef unsigned char uchar;'
#t = cpar.parse(s)

s1 = 'uchar xxx;'
t1 = cpar.parse(s1)

s2 = 'inline int calc_lut(__local int* smem, int val, int tid) {uchar dummy;}'
t2 = cpar.parse(s2)


s3 = """# 1 "c:/opencv-4.5.1/sources/modules/imgproc/src/opencl/clahe.cl"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "c:/opencv-4.5.1/sources/modules/imgproc/src/opencl/clahe.cl"
# 46 "c:/opencv-4.5.1/sources/modules/imgproc/src/opencl/clahe.cl"
inline int calc_lut(__local int* smem, int val, int tid)
{
    smem[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0)
        for (int i = 1; i < 256; ++i)
            smem[i] += smem[i - 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    return smem[tid];
}

inline int reduce(__local volatile int* smem, int val, int tid)
{
    smem[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128)
        smem[tid] = val += smem[tid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64)
        smem[tid] = val += smem[tid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem[tid] += smem[tid + 32];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16)
    {
        smem[tid] += smem[tid + 16];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 8)
    {
        smem[tid] += smem[tid + 8];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 4)
    {
        smem[tid] += smem[tid + 4];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0)
    {
        smem[0] = (smem[0] + smem[1]) + (smem[2] + smem[3]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    val = smem[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

__kernel void calcLut(__global __const uchar * src, const int srcStep,
                      const int src_offset, __global uchar * lut,
                      const int dstStep, const int dst_offset,
                      const int2 tileSize, const int tilesX,
                      const int clipLimit, const float lutScale)
{
    __local int smem[512];

    int tx = get_group_id(0);
    int ty = get_group_id(1);
    int tid = get_local_id(1) * get_local_size(0)
                             + get_local_id(0);
    smem[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = get_local_id(1); i < tileSize.y; i += get_local_size(1))
    {
        __global const uchar* srcPtr = src + mad24(ty * tileSize.y + i, srcStep, tx * tileSize.x + src_offset);
        for (int j = get_local_id(0); j < tileSize.x; j += get_local_size(0))
        {
            const int data = srcPtr[j];
            atomic_inc(&smem[data]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int tHistVal = smem[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (clipLimit > 0)
    {

        int clipped = 0;
        if (tHistVal > clipLimit)
        {
            clipped = tHistVal - clipLimit;
            tHistVal = clipLimit;
        }


        clipped = reduce(smem, clipped, tid);


        int redistBatch = clipped / 256;
        tHistVal += redistBatch;

        int residual = clipped - redistBatch * 256;
        int rStep = 256 / residual;
        if (rStep < 1)
            rStep = 1;
        if (tid%rStep == 0 && (tid/rStep)<residual)
            ++tHistVal;
    }

    const int lutVal = calc_lut(smem, tHistVal, tid);
    uint ires = (uint)convert_int_rte(lutScale * lutVal);
    lut[(ty * tilesX + tx) * dstStep + tid + dst_offset] =
        convert_uchar(clamp(ires, (uint)0, (uint)255));
}

__kernel void transform(__global __const uchar * src, const int srcStep, const int src_offset,
                        __global uchar * dst, const int dstStep, const int dst_offset,
                        __global uchar * lut, const int lutStep, int lut_offset,
                        const int cols, const int rows,
                        const int2 tileSize,
                        const int tilesX, const int tilesY)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= cols || y >= rows)
        return;

    const float tyf = (convert_float(y) / tileSize.y) - 0.5f;
    int ty1 = convert_int_rtn(tyf);
    int ty2 = ty1 + 1;
    const float ya = tyf - ty1;
    ty1 = max(ty1, 0);
    ty2 = min(ty2, tilesY - 1);

    const float txf = (convert_float(x) / tileSize.x) - 0.5f;
    int tx1 = convert_int_rtn(txf);
    int tx2 = tx1 + 1;
    const float xa = txf - tx1;
    tx1 = max(tx1, 0);
    tx2 = min(tx2, tilesX - 1);

    const int srcVal = src[mad24(y, srcStep, x + src_offset)];

    float res = 0;

    res += lut[mad24(ty1 * tilesX + tx1, lutStep, srcVal + lut_offset)] * ((1.0f - xa) * (1.0f - ya));
    res += lut[mad24(ty1 * tilesX + tx2, lutStep, srcVal + lut_offset)] * ((xa) * (1.0f - ya));
    res += lut[mad24(ty2 * tilesX + tx1, lutStep, srcVal + lut_offset)] * ((1.0f - xa) * (ya));
    res += lut[mad24(ty2 * tilesX + tx2, lutStep, srcVal + lut_offset)] * ((xa) * (ya));

    uint ires = (uint)convert_int_rte(res);
    dst[mad24(y, dstStep, x + dst_offset)] = convert_uchar(clamp(ires, (uint)0, (uint)255));
}
"""
t3 = cpar.parse(s3)
