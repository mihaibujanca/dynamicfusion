#pragma once


#include "SolverUtil.h"
#include "SFSSolverState.h"
#include "../../shared/cudaUtil.h"

#define THREADS_PER_BLOCK 1024 // keep consistent with the CPU

#define DR_THREAD_SIZE1_X 32
#define DR_THREAD_SIZE1_Y 8

__inline__ __device__ bool IsValidPoint(float d)
{
    return ((d != MINF) && (d>0));
}

__inline__ __device__ float warpReduce(float val) {
    int offset = 32 >> 1;
    while (offset > 0) {
        val = val + __shfl_down(val, offset, 32);
        offset = offset >> 1;
    }
    return val;
}

extern __shared__ float bucket[];

__inline__ __device__ bool getGlobalNeighbourIdxFromLocalNeighourIdx(int centerIdx, int localNeighbourIdx, const SolverInput& input, int& outGlobalNeighbourIdx)
{
	int i;	int j; get2DIdx(centerIdx, input.width, input.height, i, j);

	const int d = ((int)(2*(localNeighbourIdx % 2)))-1;
	const int m = localNeighbourIdx/2;

	const int iNeigh = ((int)i) + m*d;
	const int jNeigh = ((int)j) + (1-m)*d;

	if(!isInsideImage(iNeigh, jNeigh, input.width, input.height)) return false;

	outGlobalNeighbourIdx = get1DIdx(iNeigh, jNeigh, input.width, input.height);
	return true;
}

__inline__ __device__ float3 point(float d, int posx, int posy, SolverInput& input) {
    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    return make_float3((((float)posx - ux) / fx)*d, (((float)posy - uy) / fy)*d, d);

}

__inline__ __device__ float sqMagnitude(float3 in) {
    return in.x*in.x + in.y*in.y + in.z*in.z;
}


__inline__ __device__ float4 calShading2depthGradHelper(const float d0, const float d1, const float d2, int posx, int posy, SolverInput &input)
{


    const int imgind = posy * input.width + posx;

    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;


    if ((IsValidPoint(d0)) && (IsValidPoint(d1)) && (IsValidPoint(d2)))
    {
        const float greyval = (input.d_targetIntensity[imgind] * 0.5f + input.d_targetIntensity[imgind - 1] * 0.25f + input.d_targetIntensity[imgind - input.width] * 0.25f);

        float ax = (posx - ux) / fx;
        float ay = (posy - uy) / fy;
        float an, an2;

        float px, py, pz;
        px = d2*(d1 - d0) / fy;
        py = d0*(d1 - d2) / fx;
        pz = -ax*px - ay*py - d2*d0 / (fx*fy);
        an2 = px*px + py*py + pz*pz;
        an = sqrt(an2);
        if (an == 0)
        {
            float4 retval;
            retval.x = 0.0f; retval.y = 0.0f; retval.z = 0.0f; retval.w = 0.0f;
            return retval;
        }

        px /= an;
        py /= an;
        pz /= an;


        float sh_callist0 = input.d_litcoeff[0];
        float sh_callist1 = py*input.d_litcoeff[1];
        float sh_callist2 = pz*input.d_litcoeff[2];
        float sh_callist3 = px*input.d_litcoeff[3];
        float sh_callist4 = px * py * input.d_litcoeff[4];
        float sh_callist5 = py * pz * input.d_litcoeff[5];
        float sh_callist6 = ((-px*px - py*py + 2 * pz*pz))*input.d_litcoeff[6];
        float sh_callist7 = pz * px * input.d_litcoeff[7];
        float sh_callist8 = (px * px - py * py)*input.d_litcoeff[8];

        //normal changes wrt depth
        register float gradx = 0, grady = 0, gradz = 0;


        gradx += -sh_callist1*px;
        gradx += -sh_callist2*px;
        gradx += (input.d_litcoeff[3] - sh_callist3*px);
        gradx += py*input.d_litcoeff[4] - sh_callist4 * 2 * px;
        gradx += -sh_callist5 * 2 * px;
        gradx += (-2 * px)*input.d_litcoeff[6] - sh_callist6 * 2 * px;
        gradx += pz*input.d_litcoeff[7] - sh_callist7 * 2 * px;
        gradx += 2 * px*input.d_litcoeff[8] - sh_callist8 * 2 * px;
        gradx /= an;


        grady += (input.d_litcoeff[1] - sh_callist1*py);
        grady += -sh_callist2*py;
        grady += -sh_callist3*py;
        grady += px*input.d_litcoeff[4] - sh_callist4 * 2 * py;
        grady += pz*input.d_litcoeff[5] - sh_callist5 * 2 * py;
        grady += (-2 * py)*input.d_litcoeff[6] - sh_callist6 * 2 * py;
        grady += -sh_callist7 * 2 * py;
        grady += (-2 * py)*input.d_litcoeff[8] - sh_callist8 * 2 * py;
        grady /= an;

        gradz += -sh_callist1*pz;
        gradz += (input.d_litcoeff[2] - sh_callist2*pz);
        gradz += -sh_callist3*pz;
        gradz += -sh_callist4 * 2 * pz;
        gradz += py*input.d_litcoeff[5] - sh_callist5 * 2 * pz;
        gradz += 4 * pz*input.d_litcoeff[6] - sh_callist6 * 2 * pz;
        gradz += px*input.d_litcoeff[7] - sh_callist7 * 2 * pz;
        gradz += -sh_callist8 * 2 * pz;
        gradz /= an;

        //shading value stored in sh_callist0
        sh_callist0 += sh_callist1;
        sh_callist0 += sh_callist2;
        sh_callist0 += sh_callist3;
        sh_callist0 += sh_callist4;
        sh_callist0 += sh_callist5;
        sh_callist0 += sh_callist6;
        sh_callist0 += sh_callist7;
        sh_callist0 += sh_callist8;
        sh_callist0 -= greyval;



        ///////////////////////////////////////////////////////
        //
        //               /|  2
        //             /  |
        //           /    |  
        //         0 -----|  1
        //
        ///////////////////////////////////////////////////////

        float3 grnds;

        grnds.x = -d2 / fy;
        grnds.y = (d1 - d2) / fx;
        grnds.z = -ax*grnds.x - ay*grnds.y - d2 / (fx*fy);
        sh_callist1 = (gradx*grnds.x + grady*grnds.y + gradz*grnds.z);

        grnds.x = d2 / fy;
        grnds.y = d0 / fx;
        grnds.z = -ax*grnds.x - ay*grnds.y;
        sh_callist2 = (gradx*grnds.x + grady*grnds.y + gradz*grnds.z);

        grnds.x = (d1 - d0) / fy;
        grnds.y = -d0 / fx;
        grnds.z = -ax*grnds.x - ay*grnds.y - d0 / (fx*fy);
        sh_callist3 = (gradx*grnds.x + grady*grnds.y + gradz*grnds.z);

        float4 retval;
        retval.w = sh_callist0;
        retval.x = sh_callist1;
        retval.y = sh_callist2;
        retval.z = sh_callist3;

        return retval;
    }
    else
    {
        float4 retval;
        retval.x = 0.0f; retval.y = 0.0f; retval.z = 0.0f; retval.w = 0.0f;
        return retval;
    }
}
