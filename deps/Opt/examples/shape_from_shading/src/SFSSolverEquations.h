#pragma once

#include "../../shared/cudaUtil.h"

#include "SFSSolverState.h"
#include "SFSSolverParameters.h"
#include "SFSSolverState.h"
#include "SFSSolverUtil.h"

#include "SFSSolverTerms.h"

#define DEPTH_DISCONTINUITY_THRE 0.01


__inline__ __device__ float readX(SolverState& state, int posy, int posx, int W) {
    return state.d_x[posy*W + posx];
}

__inline__ __device__ float readP(int posy, int posx, SolverState& state, int W) {
    return state.d_p[posy*W + posx];
}

__inline__ __device__ float rowMask(int posy, int posx, SolverInput &input) {
    return float(input.d_maskEdgeMapR[posy*input.width + posx]);
}

__inline__ __device__ float colMask(int posy, int posx, SolverInput &input) {
    return float(input.d_maskEdgeMapC[posy*input.width + posx]);
}



__inline__ __device__ float4 calShading2depthGradCompute(SolverState& state, int posx, int posy, SolverInput &input)
{
    const int W = input.width;
    const float d0 = readX(state, posy, posx - 1, W);
    const float d1 = readX(state, posy, posx, W);
    const float d2 = readX(state, posy - 1, posx, W);

    return calShading2depthGradHelper(d0, d1, d2, posx, posy, input);
}

__inline__ __device__ float4 calShading2depthGrad(SolverState& state, int x, int y, SolverInput &inp)
{
#if USE_PRECOMPUTE
    return make_float4(state.B_I_dx0[y*inp.width+x],state.B_I_dx1[y*inp.width+x],state.B_I_dx2[y*inp.width+x],state.B_I[y*inp.width+x]);
#else
    return calShading2depthGradCompute(state, x, y, inp);
#endif
}


/*
#if USE_PRECOMPUTE
    #define calShading2depthGrad(state, x, y, inp) make_float4(state.B_I_dx0[y*inp.width+x],state.B_I_dx1[y*inp.width+x],state.B_I_dx2[y*inp.width+x],state.B_I[y*inp.width+x])
#else
    #define calShading2depthGrad calShading2depthGradCompute
#endif
*/

////////////////////////////////////////
// evalF
////////////////////////////////////////
__inline__ __device__ float evalFDevice(int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
    int posy; int posx; get2DIdx(variableIdx, input.width, input.height, posy, posx);

    const int W = input.width;
    const int H = input.height;


    // Common stuff
    const float targetDepth = input.d_targetDepth[variableIdx]; const bool validTarget = IsValidPoint(targetDepth);
    const float XC = state.d_x[variableIdx];


    float cost = 0.0f;

    float3 E_s = make_float3(0.0f, 0.0f, 0.0f);
    float E_p = 0.0f;
    float E_r_h = 0.0f;
    float E_r_v = 0.0f;
    float E_r_d = 0.0f;
    float E_g_v = 0.0f;
    float E_g_h = 0.0f;

    if (validTarget)
    {
        if (posx>1 && posx<(W - 5) && posy>1 && posy<(H - 5)){

#           if USE_SHADING_CONSTRAINT
                float4 temp00 = calShading2depthGrad(state, posx, posy, input);
                float4 temp10 = calShading2depthGrad(state, posx+1, posy, input);
                float4 temp01 = calShading2depthGrad(state, posx, posy+1, input);

                

                E_g_h = (temp00.w - temp10.w);
                E_g_v = (temp00.w - temp01.w);
#               ifdef USE_MASK_REFINE
                    E_g_h *= input.d_maskEdgeMapR[variableIdx];
                    E_g_v *= input.d_maskEdgeMapC[variableIdx];
#               endif	
#           endif
            //////////////////////////////////////////////////////////////////
            //                   smoothness term
            /////////////////////////////////////////////////////////////////
#           if USE_REGULARIZATION

            float d  = readX(state, posy, posx, W);
            float d0 = readX(state, posy, posx - 1, W);
            float d1 = readX(state, posy, posx + 1, W);
            float d2 = readX(state, posy - 1, posx, W);
            float d3 = readX(state, posy + 1, posx, W);

            if (state.pguard[posy*W+posx])
            {
                E_s = 4.0f*point(d, posx, posy, input) - (point(d1, posx + 1, posy, input) + point(d0, posx - 1, posy, input) + point(d3, posx, posy + 1, input) + point(d2, posx, posy - 1, input));
            }
#           endif


#           if USE_DEPTH_CONSTRAINT						
            //position term 			
            E_p = XC - targetDepth;
#           endif



            //////////////////////////////////////////////////////////////////
            //                   piror term
            /////////////////////////////////////////////////////////////////
            //first: calculate the normal for PriorDepth
#           if USE_TEMPORAL_CONSTRAINT
            // TODO: implement
#           endif
            cost = (parameters.weightRegularizer   * sqMagnitude(E_s)) + // This is usually on the order of 1/10,000,000
                (parameters.weightFitting       * E_p*E_p) + // 
                (parameters.weightShading       * (E_g_h*E_g_h + E_g_v*E_g_v)) +
                (parameters.weightPrior * (E_r_h*E_r_h + E_r_v*E_r_v + E_r_d*E_r_d));
            //cost = input.d_maskEdgeMap[variableIdx+W*H];
            //cost = calShading2depthGrad(state, posx, posy, input).w;
        }
    }


    return cost;

}


__device__ inline float3 est_lap_init_3d_imp(SolverState& state, int posx, int posy, const float w0, const float w1, const float &ufx, const float &ufy, const int W, bool &b_valid)
{
    float3 retval;
    retval.x = 0.0f;
    retval.y = 0.0f;
    retval.z = 0.0f;

    if (state.pguard[posy*W+posx])
    {
        float d = readX(state, posy, posx, W);
        float d0 = readX(state, posy, posx - 1, W);
        float d1 = readX(state, posy, posx + 1, W);
        float d2 = readX(state, posy - 1, posx, W);
        float d3 = readX(state, posy + 1, posx, W);
        retval.x = d * w0 * 4;
        retval.y = d * w1 * 4;
        retval.z = d * 4;

        retval.x -= d0*(w0 - ufx);
        retval.y -= d0*w1;
        retval.z -= d0;

        retval.x -= d1*(w0 + ufx);
        retval.y -= d1*w1;
        retval.z -= d1;

        retval.x -= d2*w0;
        retval.y -= d2*(w1 - ufy);
        retval.z -= d2;

        retval.x -= d3*w0;
        retval.y -= d3*(w1 + ufy);
        retval.z -= d3;

    }
    else
        b_valid = false;

    return retval;
}


__device__ inline float3 est_lap_3d_bsp_imp_with_guard(SolverState& state, int posx, int posy, float w0, float w1, const float &ufx, const float &ufy, const int W)
{
    float3 retval = make_float3(0.0f, 0.0f, 0.0f);
      
    if (state.pguard[posy*W+posx]) {
        const float p = readP(posy, posx, state, W);
        const float p0 = readP(posy, posx - 1, state, W);
        const float p1 = readP(posy, posx + 1, state, W);
        const float p2 = readP(posy - 1, posx, state, W);
        const float p3 = readP(posy + 1, posx, state, W);
        
        retval.x = ( p * 4 * w0 - p0 * (w0 - ufx) - p1 * (w0 + ufx)   - p2 * w0 - p3 * w0);
        retval.y = ( p * 4 * w1 - p0 * w1 - p1 * w1 - p2 * (w1 - ufy) - p3 * (w1 + ufy));
        retval.z = ( p * 4 - p0 - p1 - p2 - p3);
    } //else { @b_valid = false }

    return retval;

}


__device__ inline float3 est_lap_3d_bsp_imp(SolverState& state, int posx, int posy, float w0, float w1, const float &ufx, const float &ufy, const int W)
{
    float3 retval;
    
    const float d = readP(posy, posx, state, W);
    const float d0 = readP(posy, posx - 1, state, W);
    const float d1 = readP(posy, posx + 1, state, W);
    const float d2 = readP(posy - 1, posx, state, W);
    const float d3 = readP(posy + 1, posx, state, W);

    retval.x = (d * 4 * w0 - d0 * (w0 - ufx) - d1 * (w0 + ufx) - d2 * w0 - d3 * w0);
    retval.y = (d * 4 * w1 - d0 * w1 - d1 * w1 - d2 * (w1 - ufy) - d3 * (w1 + ufy));
    retval.z = (d * 4 - d0 - d1 - d2 - d3);
    return retval;

}


////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float& pre)
{
    int posy; int posx; get2DIdx(variableIdx, input.width, input.height, posy, posx);

    const int W = input.width;
    const int H = input.height;

    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    const float ufx = 1.0f / input.calibparams.fx;
    const float ufy = 1.0f / input.calibparams.fy;


    float b = 0.0f; float p = 0.0f;

    // Common stuff
    const float targetDepth = input.d_targetDepth[variableIdx]; 
    const bool validTarget  = IsValidPoint(targetDepth);
    const float XC = state.d_x[variableIdx];
    if (validTarget)
    {
        if (posx>1 && posx<(W - 5) && posy>1 && posy<(H - 5)){

            float sum, tmpval;
            float val0, val1, val2;
            unsigned char maskval = 1;
#           if USE_SHADING_CONSTRAINT
            val0 = calShading2depthGrad(state, posx, posy, input).y;
            val1 = calShading2depthGrad(state, posx + 1, posy, input).x;
            val2 = calShading2depthGrad(state, posx, posy + 1, input).z;


#               ifdef USE_MASK_REFINE
            //calculating residue error
            //shading term
            //row edge constraint
            sum = 0.0f;
            tmpval = 0.0f;
            tmpval = -(calShading2depthGrad(state, posx - 1, posy, input).w - calShading2depthGrad(state, posx, posy, input).w);// edge 0				
            maskval = rowMask(posy, posx - 1, input);
            sum += tmpval*(-val0) * maskval;//(posy, posx-1)*val0,(posy,posx)*(-val0)			
            tmpval += val0*val0*maskval;

            tmpval = -(calShading2depthGrad(state, posx, posy, input).w - calShading2depthGrad(state, posx + 1, posy, input).w);//edge 2				
            maskval = rowMask(posy, posx, input);
            sum += tmpval*(val0 - val1) * maskval;// (posy,posx)*(val1-val0), (posy,posx+1)*(val0-val1)			
            tmpval += (val0 - val1)*(val0 - val1)* maskval;

            tmpval = -(calShading2depthGrad(state, posx + 1, posy, input).w - calShading2depthGrad(state, posx + 2, posy, input).w);//edge 4				
            maskval = rowMask(posy, posx + 1, input);
            sum += tmpval*(val1)* maskval;//(posy,posx+1)*(-val1), (posy,posx+2)*(val1)			
            tmpval += val1*val1* maskval;

            tmpval = -(calShading2depthGrad(state, posx - 1, posy + 1, input).w - calShading2depthGrad(state, posx, posy + 1, input).w);//edge 5				
            maskval = rowMask(posy + 1, posx - 1, input);
            sum += tmpval*(-val2) * maskval;//(posy+1,posx-1)*(val2),(posy+1,pox)*(-val2)			
            tmpval += val2*val2* maskval;

            tmpval = -(calShading2depthGrad(state, posx, posy + 1, input).w - calShading2depthGrad(state, posx + 1, posy + 1, input).w);//edge 7				
            maskval = rowMask(posy + 1, posx, input);
            sum += tmpval*val2 * maskval;//(posy+1,posx)*(-val2),(posy+1,posx+1)*(val2)
            tmpval += val2*val2 * maskval;

            //column edge constraint			
            tmpval = -(calShading2depthGrad(state, posx, posy - 1, input).w - calShading2depthGrad(state, posx, posy, input).w);//edge 1
            maskval = colMask(posy - 1, posx, input);
            sum += tmpval*(-val0) * maskval;//(posy-1,posx)*(val0),(posy,posx)*(-val0)			
            tmpval += val0*val0* maskval;

            tmpval = -(calShading2depthGrad(state, posx + 1, posy - 1, input).w - calShading2depthGrad(state, posx + 1, posy, input).w);//edge 3
            maskval = colMask(posy - 1, posx + 1, input);
            sum += tmpval*(-val1) * maskval;//(posy-1,posx+1)*(val1),(posy,posx+1)*(-val1)			
            tmpval += val1*val1* maskval;

            tmpval = -(calShading2depthGrad(state, posx, posy, input).w - calShading2depthGrad(state, posx, posy + 1, input).w);//edge 6
            maskval = colMask(posy, posx, input);
            sum += tmpval*(val0 - val2) * maskval;//(posy,posx)*(val2-val0),(posy+1,posx)*(val0-val2)			
            tmpval += (val0 - val2)*(val0 - val2)* maskval;

            tmpval = -(calShading2depthGrad(state, posx + 1, posy, input).w - calShading2depthGrad(state, posx + 1, posy + 1, input).w);//edge 8
            maskval = colMask(posy, posx + 1, input);
            sum += tmpval*val1 * maskval;//(posy,posx+1)*(-val1),(posy+1,posx+1)*(val1)
            tmpval += val1*val1* maskval;

            tmpval = -(calShading2depthGrad(state, posx, posy + 1, input).w - calShading2depthGrad(state, posx, posy + 2, input).w);//edge 9
            maskval = colMask(posy + 1, posx, input);
            sum += tmpval*val2 * maskval;//(posy+1,posx)*(-val2),(posy+2,posx)*(val2)
            tmpval += val2*val2* maskval;

            b += sum * parameters.weightShading;
            p += tmpval * parameters.weightShading;//shading constraint
#               else
                    tmpval = 0.0f;
                    tmpval = val0 * val0 * 2;
                    tmpval += (val0 - val1) * (val0 - val1);
                    tmpval += (val0 - val2) * (val0 - val2);
                    tmpval += val1 * val1 * 3;
                    tmpval += val2 * val2 * 3;
                    p += tmpval * parameters.weightShading;//shading constraint


                    sum = 0.0f;
                    sum += val0*calShading2depthGrad(state, posx, posy - 1, input).w;
                    sum += val1 * calShading2depthGrad(state, posx + 1, posy - 1, input).w;
                    sum += val0 * calShading2depthGrad(state, posx - 1, posy, input).w;
                    sum += (-val0 + val1 - val0 - val0 + val2 - val0) * calShading2depthGrad(state, posx, posy, input).w;
                    sum += (val0 - val1 - val1 - val1 - val1) * calShading2depthGrad(state, posx + 1, posy, input).w;
                    sum += val1 * calShading2depthGrad(state, posx + 2, posy, input).w;
                    sum += val2 * calShading2depthGrad(state, posx - 1, posy + 1, input).w;
                    sum += (-val2 - val2 + val0 - val2 - val2) * calShading2depthGrad(state, posx, posy + 1, input).w;
                    sum += (val2 + val1) * calShading2depthGrad(state, posx + 1, posy + 1, input).w;
                    sum += val2  * calShading2depthGrad(state, posx, posy + 2, input).w;

                    b += sum * parameters.weightShading;
#               endif	
#           endif


            //////////////////////////////////////////////////////////////////
            //                   smoothness term
            /////////////////////////////////////////////////////////////////
#           if USE_REGULARIZATION
            bool b_valid = true;

            val0 = (posx - ux) / fx;
            val1 = (posy - uy) / fy;
            
            //smoothness term							
            float3 lapval = est_lap_init_3d_imp(state, posx, posy, val0, val1, ufx, ufy, W, b_valid);
            sum = 0.0f;
            sum += lapval.x*val0*(-4.0f);
            sum += lapval.y*val1*(-4.0f);
            sum += lapval.z*(-4.0f);

            lapval = est_lap_init_3d_imp(state, posx - 1, posy, val0 - ufx, val1, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;

            lapval = est_lap_init_3d_imp(state, posx + 1, posy, val0 + ufx, val1, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;

            lapval = est_lap_init_3d_imp(state, posx, posy - 1, val0, val1 - ufy, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;

            lapval = est_lap_init_3d_imp(state, posx, posy + 1, val0, val1 + ufy, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;
#if (USE_PROPER_REGULARIZATION_BOUNDARY == 0)
            if (b_valid)
#endif
            {

                b += sum*parameters.weightRegularizer;
                tmpval = (val0 * val0 + val1 * val1 + 1)*(16 + 4);
                p += tmpval *parameters.weightRegularizer;//smoothness
            }
#           endif


#           if USE_DEPTH_CONSTRAINT						
            //position term 			
            p += parameters.weightFitting;//position constraint			
            b += -(XC - targetDepth) * parameters.weightFitting;
#           endif



            //////////////////////////////////////////////////////////////////
            //                   piror term
            /////////////////////////////////////////////////////////////////
            //first: calculate the normal for PriorDepth
#           if USE_TEMPORAL_CONSTRAINT
            float d;
            sum = 0.0f;
            float ax = (posx - ux) / fx;
            float ay = (posy - uy) / fy;

            tmpval = normal0.x * ax + normal0.y * ay + normal0.z;// derative of prior energy wrt depth			
            p += tmpval * tmpval * 2 * parameters.weightPrior;
            d = readValueFromCache2D_SFS(inX, tidy, tidx - 1);
            if (IsValidPoint(d))
                sum -= tmpval * (tmpval * readValueFromCache2D_SFS(inX, tidy, tidx) + (-tmpval + normal0.x / fx) * d);
            d = readValueFromCache2D_SFS(inX, tidy - 1, tidx);
            if (IsValidPoint(d))
                sum -= tmpval * (tmpval * readValueFromCache2D_SFS(inX, tidy, tidx) + (-tmpval + normal0.y / fy) * d);

            tmpval = normal1.x * ax + normal1.y * ay + normal1.z;// derative of prior energy wrt depth			
            p += tmpval * tmpval * parameters.weightPrior;
            d = readValueFromCache2D_SFS(inX, tidy, tidx + 1);
            if (IsValidPoint(d))
                sum -= -tmpval * ((tmpval + normal1.x / fx) * d - tmpval * readValueFromCache2D_SFS(inX, tidy, tidx));

            tmpval = normal2.x * ax + normal2.y * ay + normal2.z;// derative of prior energy wrt depth
            p += tmpval * tmpval * parameters.weightPrior;
            d = readValueFromCache2D_SFS(inX, tidy + 1, tidx);
            if (IsValidPoint(d))
                sum -= -tmpval * ((tmpval + normal2.y / fy) * d - tmpval * readValueFromCache2D_SFS(inX, tidy, tidx));

            b += sum  * parameters.weightPrior;
#           endif

        }
    }


    if (p > FLOAT_EPSILON) pre = 1.0f / p;
    else			      pre = 1.0f;
#if USE_PRECONDITIONER == 0
    pre = 1.0f;
#endif
    return b;
}


__device__ inline float add_mul_inp_grad_ls_bsp(SolverState& state, SolverInput& input, int posx, int posy, const int W)
{
    float4 temp = calShading2depthGrad(state, posx, posy, input);
    return (readP(posy, posx - 1, state, W)	* temp.x
        + readP(posy, posx, state, W)	    * temp.y
        + readP(posy - 1, posx, state, W)	* temp.z);
}



////////////////////////////////////////
// applyJTJ : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
    int posy; int posx; get2DIdx(variableIdx, input.width, input.height, posy, posx);

    const int W = input.width;
    const int H = input.height;

    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    const float ufx = 1.0f / input.calibparams.fx;
    const float ufy = 1.0f / input.calibparams.fy;

    float b = 0.0f;

    const float targetDepth = input.d_targetDepth[posy*W + posx]; const bool validTarget = IsValidPoint(targetDepth);
    const float PC = state.d_p[variableIdx];

    if (validTarget)
    {
        if ((posx>1) && (posx<(W - 5)) && (posy>1) && (posy<(H - 5)))
        {

            float sum = 0.0f;
            float tmpval = 0.0f;

            float val0, val1, val2;
#           if USE_SHADING_CONSTRAINT
            val0 = calShading2depthGrad(state, posx, posy, input).y;
            val1 = calShading2depthGrad(state, posx + 1, posy, input).x;
            val2 = calShading2depthGrad(state, posx, posy + 1, input).z;
            float4 grad_0_m1 = calShading2depthGrad(state,posx+0, posy-1, input);				
            float4 grad_1_m1 = calShading2depthGrad(state,posx+1, posy-1, input);
            float4 grad_m1_0 = calShading2depthGrad(state,posx-1, posy-0, input);
            float4 grad_0_0  = calShading2depthGrad(state,posx  , posy  , input);
            float4 grad_1_0  = calShading2depthGrad(state,posx+1, posy  , input);
            float4 grad_2_0  = calShading2depthGrad(state,posx+2, posy  , input);
            float4 grad_m1_1 = calShading2depthGrad(state,posx-1, posy+1, input);
            float4 grad_0_1  = calShading2depthGrad(state,posx  , posy+1, input);
            float4 grad_1_1  = calShading2depthGrad(state,posx+1, posy+1, input);
            float4 grad_0_2  = calShading2depthGrad(state,posx  , posy+2, input);

#               ifdef USE_MASK_REFINE
                    //readP(posy, posx, state, W)	* calShading2depthGrad(state, posx, posy, input).x; // Doesn't do anything?!

                    //the following is the adding of the relative edge constraints to the sum
                    //-val0, edge 0			
                    tmpval = readP(posy, posx - 2, state, W) *  grad_m1_0.x;
                    tmpval += readP(posy, posx - 1, state, W) * (grad_m1_0.y - grad_0_0.x);
                    tmpval += readP(posy - 1, posx - 1, state, W) *  grad_m1_0.z;
                    tmpval -= readP(posy, posx, state, W) *  grad_0_0.y;
                    tmpval -= readP(posy - 1, posx, state, W) *  grad_0_0.z;
                    sum += (-val0) * tmpval  * rowMask(posy, posx - 1, input);

                    //-val0, edge 1
                    tmpval = readP(posy - 1, posx - 1, state, W) *  grad_0_m1.x;
                    tmpval += readP(posy - 1, posx, state, W) * (grad_0_m1.y - grad_0_0.z);
                    tmpval += readP(posy - 2, posx, state, W) *  grad_0_m1.z;
                    tmpval -= readP(posy, posx - 1, state, W) *  grad_0_0.x;
                    tmpval -= readP(posy, posx, state, W) *  grad_0_0.y;
                    sum += (-val0) * tmpval  * colMask(posy - 1, posx, input);

                    //val0-val1, edge 2
                    tmpval = readP(posy, posx - 1, state, W) *  grad_0_0.x;
                    tmpval += readP(posy, posx, state, W) * (grad_0_0.y - grad_1_0.x);
                    tmpval += readP(posy - 1, posx, state, W) *  grad_0_0.z;
                    tmpval -= readP(posy, posx + 1, state, W) *  grad_1_0.y;
                    tmpval -= readP(posy - 1, posx + 1, state, W) *  grad_1_0.z;
                    sum += (val0 - val1) * tmpval * rowMask(posy, posx, input);

                    //-val1, edge 3			
                    tmpval = readP(posy - 1, posx, state, W) *  grad_1_m1.x;
                    tmpval += readP(posy - 1, posx + 1, state, W) * (grad_1_m1.y - grad_1_0.z);
                    tmpval += readP(posy - 2, posx + 1, state, W) *  grad_1_m1.z;
                    tmpval -= readP(posy, posx, state, W) *  grad_1_0.x;
                    tmpval -= readP(posy, posx + 1, state, W) *  grad_1_0.y;
                    sum += (-val1) * tmpval	* colMask(posy - 1, posx + 1, input);

                    //val1, edge 4
                    tmpval = readP(posy, posx, state, W) *  grad_1_0.x;
                    tmpval += readP(posy, posx + 1, state, W) * (grad_1_0.y - grad_2_0.x);
                    tmpval += readP(posy - 1, posx + 1, state, W) *  grad_1_0.z;
                    tmpval -= readP(posy, posx + 2, state, W) *  grad_2_0.y;
                    tmpval -= readP(posy - 1, posx + 2, state, W) *  grad_2_0.z;
                    sum += (val1)* tmpval * rowMask(posy, posx + 1, input);

                    //-val2, edge 5			
                    tmpval = readP(posy + 1, posx - 2, state, W) *  grad_m1_1.x;
                    tmpval += readP(posy + 1, posx - 1, state, W) * (grad_m1_1.y - grad_0_1.x);
                    tmpval += readP(posy, posx - 1, state, W) *  grad_m1_1.z;
                    tmpval -= readP(posy + 1, posx, state, W) *  grad_0_1.y;
                    tmpval -= readP(posy, posx, state, W) *  grad_0_1.z;
                    sum += (-val2) * tmpval * rowMask(posy + 1, posx - 1, input);

                    //val0-val2, edge 6
                    tmpval = readP(posy, posx - 1, state, W) *  grad_0_0.x;
                    tmpval += readP(posy, posx, state, W) * (grad_0_0.y - grad_0_1.z);
                    tmpval += readP(posy - 1, posx, state, W) *  grad_0_0.z;
                    tmpval -= readP(posy + 1, posx - 1, state, W) *  grad_0_1.x;
                    tmpval -= readP(posy + 1, posx, state, W) *  grad_0_1.y;
                    sum += (val0 - val2) * tmpval * colMask(posy, posx, input);

                    //val2, edge 7
                    tmpval = readP(posy + 1, posx - 1, state, W) *  grad_0_1.x;
                    tmpval += readP(posy + 1, posx, state, W) * (grad_0_1.y - grad_1_1.x);
                    tmpval += readP(posy, posx, state, W) *  grad_0_1.z;
                    tmpval -= readP(posy + 1, posx + 1, state, W) *  grad_1_1.y;
                    tmpval -= readP(posy, posx + 1, state, W) *  grad_1_1.z;
                    sum += val2 * tmpval * rowMask(posy + 1, posx, input);

                    //val1, edge 8
                    tmpval = readP(posy, posx, state, W) *  grad_1_0.x;
                    tmpval += readP(posy, posx + 1, state, W) * (grad_1_0.y - grad_1_1.z);
                    tmpval += readP(posy - 1, posx + 1, state, W) *  grad_1_0.z;
                    tmpval -= readP(posy + 1, posx, state, W) *  grad_1_1.x;
                    tmpval -= readP(posy + 1, posx + 1, state, W) *  grad_1_1.y;
                    sum += val1 * tmpval * colMask(posy, posx + 1, input);

                    //val2, edge 9
                    tmpval = readP(posy + 1, posx - 1, state, W) *  grad_0_1.x;
                    tmpval += readP(posy + 1, posx, state, W) * (grad_0_1.y - grad_0_2.z);
                    tmpval += readP(posy, posx, state, W) *  grad_0_1.z;
                    tmpval -= readP(posy + 2, posx - 1, state, W) *  grad_0_2.x;
                    tmpval -= readP(posy + 2, posx, state, W) *  grad_0_2.y;
                    sum += val2 * tmpval * colMask(posy + 1, posx, input);

                    b += sum * parameters.weightShading;

#               else											
                    sum += (val1*4.0f - val0) * add_mul_inp_grad_ls_bsp(state, input, posx + 1, posy, W);//mulitplication of grad with inP needs to consid			
                    sum += (val2*4.0f - val0) * add_mul_inp_grad_ls_bsp(state, input, posx, posy + 1, W);
                    sum += (val0*4.0f - val1 - val2) * add_mul_inp_grad_ls_bsp(state, input, posx, posy, W);
                    sum += (-val2 - val1) * add_mul_inp_grad_ls_bsp(state, input, posx + 1, posy + 1, W);
                    sum += (-val0) * add_mul_inp_grad_ls_bsp(state, input, posx - 1, posy, W);
                    sum += (-val1) * add_mul_inp_grad_ls_bsp(state, input, posx + 2, posy, W);
                    sum += (-val0) * add_mul_inp_grad_ls_bsp(state, input, posx, posy - 1, W);
                    sum += (-val1) *  add_mul_inp_grad_ls_bsp(state, input, posx + 1, posy - 1, W);
                    sum += (-val2) *  add_mul_inp_grad_ls_bsp(state, input, posx - 1, posy + 1, W);
                    sum += (-val2) *  add_mul_inp_grad_ls_bsp(state, input, posx, posy + 2, W);
                    b += sum * parameters.weightShading;
#               endif
#           endif


            //////////////////////////////////////////////////////////////////
            //                  Smoothness Term
            /////////////////////////////////////////////////////////////////
#           if USE_REGULARIZATION	

            sum = 0;
            val0 = (posx - ux) / fx;
            val1 = (posy - uy) / fy;
#           if USE_PROPER_REGULARIZATION_BOUNDARY
                float3 lapval = est_lap_3d_bsp_imp_with_guard(state, posx, posy, val0, val1, ufx, ufy, W);
                sum += lapval.x*val0*(4.0f);
                sum += lapval.y*val1*(4.0f);
                sum += lapval.z*(4.0f);
                
                lapval = est_lap_3d_bsp_imp_with_guard(state, posx - 1, posy, val0 - ufx, val1, ufx, ufy, W);
                sum -= lapval.x*val0;
                sum -= lapval.y*val1;
                sum -= lapval.z;

                lapval = est_lap_3d_bsp_imp_with_guard(state, posx + 1, posy, val0 + ufx, val1, ufx, ufy, W);
                sum -= lapval.x*val0;
                sum -= lapval.y*val1;
                sum -= lapval.z;

                lapval = est_lap_3d_bsp_imp_with_guard(state, posx, posy - 1, val0, val1 - ufy, ufx, ufy, W);
                sum -= lapval.x*val0;
                sum -= lapval.y*val1;
                sum -= lapval.z;

                lapval = est_lap_3d_bsp_imp_with_guard(state, posx, posy + 1, val0, val1 + ufy, ufx, ufy, W);
                sum -= lapval.x*val0;
                sum -= lapval.y*val1;
                sum -= lapval.z;
#           else
                float3 lapval = est_lap_3d_bsp_imp(state, posx, posy, val0, val1, ufx, ufy, W);
                sum += lapval.x*val0*(4.0f);
                sum += lapval.y*val1*(4.0f);
                sum += lapval.z*(4.0f);
                
                lapval = est_lap_3d_bsp_imp(state, posx - 1, posy, val0 - ufx, val1, ufx, ufy, W);
                sum -= lapval.x*val0;
                sum -= lapval.y*val1;
                sum -= lapval.z;

                lapval = est_lap_3d_bsp_imp(state, posx + 1, posy, val0 + ufx, val1, ufx, ufy, W);
                sum -= lapval.x*val0;
                sum -= lapval.y*val1;
                sum -= lapval.z;

                lapval = est_lap_3d_bsp_imp(state, posx, posy - 1, val0, val1 - ufy, ufx, ufy, W);
                sum -= lapval.x*val0;
                sum -= lapval.y*val1;
                sum -= lapval.z;

                lapval = est_lap_3d_bsp_imp(state, posx, posy + 1, val0, val1 + ufy, ufx, ufy, W);
                sum -= lapval.x*val0;
                sum -= lapval.y*val1;
                sum -= lapval.z;
#           endif

            //sum = readP(posy + 1, posx + 2, state, W);
            b += sum*parameters.weightRegularizer;
#           endif

            //////////////////////////////////////////////////////////////////
            //                  Position Term
            /////////////////////////////////////////////////////////////////		

#           if USE_DEPTH_CONSTRAINT
            b += PC*parameters.weightFitting;
#           endif


            //////////////////////////////////////////////////////////////////
            //                   piror term
            /////////////////////////////////////////////////////////////////			
#           if USE_TEMPORAL_CONSTRAINT
            sum = 0.0f;
            float ax = (posx - ux) / fx;
            float ay = (posy - uy) / fy;
            tmpval = normal0.x * ax + normal0.y * ay + normal0.z;// derative of prior energy wrt depth			
            sum += tmpval * (tmpval * readP(posy, posx, state, W) + (-tmpval + normal0.x / fx) * readP(posy, posx - 1, state, W));
            sum += tmpval * (tmpval * readP(posy, posx, state, W) + (-tmpval + normal0.y / fy) * readP(posy - 1, posx, state, W));

            tmpval = normal1.x * ax + normal1.y * ay + normal1.z;// derative of prior energy wrt depth			
            sum += -tmpval * ((tmpval + normal1.x / fx) * readP(posy, posx + 1, state, W) - tmpval * readP(posy, posx, state, W));

            tmpval = normal2.x * ax + normal2.y * ay + normal2.z;// derative of prior energy wrt depth			
            sum += -tmpval * ((tmpval + normal2.y / fy) * readP(posy + 1, posx, state, W) - tmpval * readP(posy, posx, state, W));

            b += sum  * parameters.weightPrior;
#           endif
        }
    }


    return b;
}

