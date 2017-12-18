local S = require("std")
local util = require("util")
require("precision")

local ffi = require("ffi")

local C = util.C
local Timer = util.Timer

local getValidUnknown = util.getValidUnknown

local GuardedInvertType = { CERES = {}, MODIFIED_CERES = {}, EPSILON_ADD = {} }

-- CERES default, ONCE_PER_SOLVE
local JacobiScalingType = { NONE = {}, ONCE_PER_SOLVE = {}, EVERY_ITERATION = {}}

local guardDivisionByZero = true

local initialization_parameters = {
    use_cusparse = false,
    use_fused_jtj = false,
    guardedInvertType = GuardedInvertType.CERES,
    jacobiScaling = JacobiScalingType.ONCE_PER_SOLVE
}

local solver_parameter_defaults = {
    residual_reset_period = 10,
    min_relative_decrease = 1e-3,
    min_trust_region_radius = 1e-32,
    max_trust_region_radius = 1e16,
    q_tolerance = 0.0001,
    function_tolerance = 0.000001,
    trust_region_radius = 1e4,
    radius_decrease_factor = 2.0,
    min_lm_diagonal = 1e-6,
    max_lm_diagonal = 1e32,
    nIterations = 10,
    lIterations = 10
}


local logDebugCudaOptFloat
if _opt_verbosity > 2 then 
    logDebugCudaOptFloat = macro(function(name,val)
        return quote
            var h_val : opt_float
            C.cudaMemcpy(&h_val, val, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
            C.printf("%s: %f\n", name, h_val)
        end
    end)
else
    logDebugCudaOptFloat = macro(function(name,val)
        return 0
    end)
end

local multistep_alphaDenominator_compute = initialization_parameters.use_cusparse

local cd = macro(function(apicall) 
    local apicallstr = tostring(apicall)
    local filename = debug.getinfo(1,'S').source
    return quote
        var str = [apicallstr]
        var r = apicall
        if r ~= 0 then  
            C.printf("Cuda reported error %d: %s\n",r, C.cudaGetErrorString(r))
            C.printf("In call: %s", str)
            C.printf("In file: %s\n", filename)
            C.exit(r)
        end
    in
        r
    end end)
if initialization_parameters.use_cusparse then
    local cusparsepath = "/usr/local/cuda"
    local cusparselibpath = "/lib64/libcusparse.dylib"
    if ffi.os == "Windows" then
        cusparsepath = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v7.5"
        cusparselibpath = "\\bin\\cusparse64_75.dll"
    end
    if ffi.os == "Linux" then
        local cusparselibpath = "/lib/libcusparse.so"
    end
    terralib.linklibrary(cusparsepath..cusparselibpath)
    terralib.includepath = terralib.includepath..";"..
                           cusparsepath.."/include/"
    CUsp = terralib.includecstring [[
        #include <cusparse_v2.h>
    ]]
end


local gpuMath = util.gpuMath

local FLOAT_EPSILON = `[opt_float](0.00000001f) 
-- GAUSS NEWTON (or LEVENBERG-MARQUADT)
return function(problemSpec)
    local UnknownType = problemSpec:UnknownType()
    local TUnknownType = UnknownType:terratype()	
    -- start of the unknowns that correspond to this image
    -- for each entry there are a constant number of unknowns
    -- corresponds to the col dim of the J matrix
    local imagename_to_unknown_offset = {}
    
    -- start of the rows of residuals for this energy spec
    -- corresponds to the row dim of the J matrix
    local energyspec_to_residual_offset_exp = {}
    
    -- start of the block of non-zero entries that correspond to this energy spec
    -- the total dimension here adds up to the number of non-zeros
    local energyspec_to_rowidx_offset_exp = {}
    
    local nUnknowns,nResidualsExp,nnzExp = 0,`0,`0
    local parametersSym = symbol(&problemSpec:ParameterType(),"parameters")
    local function numberofelements(ES)
        if ES.kind.kind == "CenteredFunction" then
            return ES.kind.ispace:cardinality()
        else
            return `parametersSym.[ES.kind.graphname].N
        end
    end
    if problemSpec.energyspecs then
        for i,image in ipairs(UnknownType.images) do
            imagename_to_unknown_offset[image.name] = nUnknowns
            --print(("image %s has offset %d"):format(image.name,nUnknowns))
            nUnknowns = nUnknowns + image.imagetype.ispace:cardinality()*image.imagetype.channelcount
        end
        for i,es in ipairs(problemSpec.energyspecs) do
            --print("ES",i,nResidualsExp,nnzExp)
            energyspec_to_residual_offset_exp[es] = nResidualsExp
            energyspec_to_rowidx_offset_exp[es] = nnzExp
            
            local residuals_per_element = #es.residuals
            nResidualsExp = `nResidualsExp + [numberofelements(es)]*residuals_per_element
            local nentries = 0
            for i,r in ipairs(es.residuals) do
                nentries = nentries + #r.unknowns
            end
            nnzExp = `nnzExp + [numberofelements(es)]*nentries
        end
        print("nUnknowns = ",nUnknowns)
        print("nResiduals = ",nResidualsExp)
        print("nnz = ",nnzExp)
    end
    
    local isGraph = problemSpec:UsesGraphs() 
    
    local struct SolverParameters {
        min_relative_decrease : float
        min_trust_region_radius : float
        max_trust_region_radius : float
        q_tolerance : float
        function_tolerance : float
        trust_region_radius : float
        radius_decrease_factor : float
        min_lm_diagonal : float
        max_lm_diagonal : float

        residual_reset_period : int
        nIter : int             --current non-linear iter counter
        nIterations : int       --non-linear iterations
        lIterations : int       --linear iterations
    }

    

    local struct PlanData {
        plan : opt.Plan
        parameters : problemSpec:ParameterType()
        solverparameters : SolverParameters
        scratch : &opt_float

        delta : TUnknownType	--current linear update to be computed -> num vars
        r : TUnknownType		--residuals -> num vars	--TODO this needs to be a 'residual type'
        b : TUnknownType        --J^TF. Constant during inner iterations, only used to recompute r to counteract drift -> num vars --TODO this needs to be a 'residual type'
        Adelta : TUnknownType       -- (A'A+D'D)delta TODO this needs to be a 'residual type'
        z : TUnknownType		--preconditioned residuals -> num vars	--TODO this needs to be a 'residual type'
        p : TUnknownType		--descent direction -> num vars
        Ap_X : TUnknownType	--cache values for next kernel call after A = J^T x J x p -> num vars
        CtC : TUnknownType -- The diagonal matrix C'C for the inner linear solve (J'J+C'C)x = J'F Used only by LM
        preconditioner : TUnknownType --pre-conditioner for linear system -> num vars
        SSq : TUnknownType -- Square of jacobi scaling diagonal
        g : TUnknownType		--gradient of F(x): g = -2J'F -> num vars

        prevX : TUnknownType -- Place to copy unknowns to before speculatively updating. Avoids hassle when (X + delta) - delta != X 

        scanAlphaNumerator : &opt_float
        scanAlphaDenominator : &opt_float
        scanBetaNumerator : &opt_float

        modelCost : &opt_float    -- modelCost = L(delta) where L(h) = F' F + 2 h' J' F + h' J' J h
        q : &opt_float -- Q value for zeta calculation (see CERES)

        timer : Timer
        endSolver : util.TimerEvent

        prevCost : opt_float
        
        J_csrValA : &opt_float
        J_csrColIndA : &int
        J_csrRowPtrA : &int
        
        JT_csrValA : &float
        JT_csrRowPtrA : &int
        JT_csrColIndA : &int

        JTJ_csrValA : &float
        JTJ_csrRowPtrA : &int
        JTJ_csrColIndA : &int

        JTJ_nnz : int
        
        Jp : &float
    }
	if initialization_parameters.use_cusparse then
	    PlanData.entries:insert {"handle", CUsp.cusparseHandle_t }
	    PlanData.entries:insert {"desc", CUsp.cusparseMatDescr_t }
	end
	S.Object(PlanData)
	local terra swapCol(pd : &PlanData, a : int, b : int)
	    pd.J_csrValA[a],pd.J_csrColIndA[a],pd.J_csrValA[b],pd.J_csrColIndA[b] =
	        pd.J_csrValA[b],pd.J_csrColIndA[b],pd.J_csrValA[a],pd.J_csrColIndA[a]
	end
	local terra sortCol(pd : &PlanData, s : int, e : int)
	    for i = s,e do
            var minidx = i
            var min = pd.J_csrColIndA[i]
            for j = i+1,e do
                if pd.J_csrColIndA[j] < min then
                    min = pd.J_csrColIndA[j]
                    minidx = j
                end
            end
            swapCol(pd,i,minidx)
        end
    end
    local terra wrap(c : int, v : float)
        if c < 0 then
            if v ~= 0.f then
                printf("wrap a non-zero? %d %f\n",c,v)
            end
            c = c + nUnknowns
        end
        if c >= nUnknowns then
            if v ~= 0.f then
                printf("wrap a non-zero? %d %f\n",c,v)
            end
            c = c - nUnknowns
        end
        return c
    end
    local function generateDumpJ(ES,dumpJ,idx,pd)
        local nnz_per_entry = 0
        for i,r in ipairs(ES.residuals) do
            nnz_per_entry = nnz_per_entry + #r.unknowns
        end
        local base_rowidx = energyspec_to_rowidx_offset_exp[ES]
        local base_residual = energyspec_to_residual_offset_exp[ES]
        local idx_offset
        if idx.type == int then
            idx_offset = idx
        else    
            idx_offset = `idx:tooffset()
        end
        local local_rowidx = `base_rowidx + idx_offset*nnz_per_entry
        local local_residual = `base_residual + idx_offset*[#ES.residuals]
        local function GetOffset(idx,index)
            if index.kind == "Offset" then
                return `idx([{unpack(index.data)}]):tooffset()
            else
                return `parametersSym.[index.graph.name].[index.element][idx]:tooffset()
            end
        end
        return quote
            var rhs = dumpJ(idx,pd.parameters)
            --[[escape                
                local nnz = 0
                local residual = 0
                for i,r in ipairs(ES.residuals) do
                    emit quote
                        pd.J_csrRowPtrA[local_residual+residual] = local_rowidx + nnz
                    end
                    local begincolumns = nnz
                    for i,u in ipairs(r.unknowns) do
                        local image_offset = imagename_to_unknown_offset[u.image.name]
                        local nchannels = u.image.type.channelcount
                        local uidx = GetOffset(idx,u.index)
                        local unknown_index = `image_offset + nchannels*uidx + u.channel

                        emit quote
                            pd.J_csrValA[local_rowidx + nnz] = opt_float(rhs.["_"..tostring(nnz)])
                            pd.J_csrColIndA[local_rowidx + nnz] = wrap(unknown_index,opt_float(rhs.["_"..tostring(nnz)]))
                        end
                        nnz = nnz + 1
                    end
                    -- sort the columns
                    emit quote
                        sortCol(&pd, local_rowidx + begincolumns, local_rowidx + nnz)
                    end
                    residual = residual + 1
                end
            end--]]
        end
	end
	
	local delegate = {}
	function delegate.CenterFunctions(UnknownIndexSpace,fmap)
	    local kernels = {}
	    local unknownElement = UnknownType:VectorTypeForIndexSpace(UnknownIndexSpace)
	    local Index = UnknownIndexSpace:indextype()

        local unknownWideReduction = macro(function(idx,val,reductionTarget) return quote
            val = util.warpReduce(val)
            if (util.laneid() == 0) then                
                util.atomicAdd(reductionTarget, val)
            end
        end end)

        local terra square(x : opt_float) : opt_float
            return x*x
        end

        local terra guardedInvert(p : unknownElement)
            escape 
                if initialization_parameters.guardedInvertType == GuardedInvertType.CERES then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            invp(i) = [opt_float](1.f) / square(opt_float(1.f) + util.gpuMath.sqrt(invp(i)))
                        end
                        return invp
                    end
                elseif initialization_parameters.guardedInvertType == GuardedInvertType.MODIFIED_CERES then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                             invp(i) = [opt_float](1.f) / (opt_float(1.f) + invp(i))
                        end
                        return invp
                    end
                elseif initialization_parameters.guardedInvertType == GuardedInvertType.EPSILON_ADD then
                    emit quote
                        var invp = p
                        for i = 0, invp:size() do
                            invp(i) = [opt_float](1.f) / (FLOAT_EPSILON + invp(i))
                        end
                        return invp
                    end
                end
            end
        end

        local terra clamp(x : unknownElement, minVal : unknownElement, maxVal : unknownElement) : unknownElement
            var result = x
            for i = 0, result:size() do
                result(i) = util.gpuMath.fmin(util.gpuMath.fmax(x(i), minVal(i)), maxVal(i))
            end
            return result
        end

        terra kernels.PCGInit1(pd : PlanData)
            var d : opt_float = opt_float(0.0f) -- init for out of bounds lanes
        
            var idx : Index
            if idx:initFromCUDAParams() then
        
                -- residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0                            
                var residuum : unknownElement = 0.0f
                var pre : unknownElement = 0.0f	
            
                if not fmap.exclude(idx,pd.parameters) then 
                
                    pd.delta(idx) = opt_float(0.0f)   
                
                    residuum, pre = fmap.evalJTF(idx, pd.parameters)
                    residuum = -residuum
                    pd.r(idx) = residuum
                
                    if not problemSpec.usepreconditioner then
                        pre = opt_float(1.0f)
                    end
                end        
            
                if (not fmap.exclude(idx,pd.parameters)) and (not isGraph) then		
                    pre = guardedInvert(pre)
                    var p = pre*residuum	-- apply pre-conditioner M^-1			   
                    pd.p(idx) = p
                
                    d = residuum:dot(p) 
                end
            
                pd.preconditioner(idx) = pre
            end 
            if not isGraph then
                unknownWideReduction(idx,d,pd.scanAlphaNumerator)
            end
        end
    
        terra kernels.PCGInit1_Finish(pd : PlanData)	--only called for graphs
            var d : opt_float = opt_float(0.0f) -- init for out of bounds lanes
            var idx : Index
            if idx:initFromCUDAParams() then
                var residuum = pd.r(idx)			
                var pre = pd.preconditioner(idx)
            
                pre = guardedInvert(pre)
            
                if not problemSpec.usepreconditioner then
                    pre = opt_float(1.0f)
                end
            
                var p = pre*residuum	-- apply pre-conditioner M^-1
                pd.preconditioner(idx) = pre
                pd.p(idx) = p
                d = residuum:dot(p)
            end

            unknownWideReduction(idx,d,pd.scanAlphaNumerator)
        end

        terra kernels.PCGStep1(pd : PlanData)
            var d : opt_float = opt_float(0.0f)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var tmp : unknownElement = 0.0f
                 -- A x p_k  => J^T x J x p_k 
                tmp = fmap.applyJTJ(idx, pd.parameters, pd.p, pd.CtC)
                pd.Ap_X(idx) = tmp					 -- store for next kernel call
                d = pd.p(idx):dot(tmp)			 -- x-th term of denominator of alpha
            end
            if not [multistep_alphaDenominator_compute] then
                unknownWideReduction(idx,d,pd.scanAlphaDenominator)
            end
        end
        if multistep_alphaDenominator_compute then
            terra kernels.PCGStep1_Finish(pd : PlanData)
                var d : opt_float = opt_float(0.0f)
                var idx : Index
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                    d = pd.p(idx):dot(pd.Ap_X(idx))           -- x-th term of denominator of alpha
                end
                unknownWideReduction(idx,d,pd.scanAlphaDenominator)
            end
        end

        terra kernels.PCGStep2(pd : PlanData)
            var betaNum = opt_float(0.0f) 
            var q = opt_float(0.0f) -- Only used if LM
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                -- sum over block results to compute denominator of alpha
                var alphaDenominator : opt_float = pd.scanAlphaDenominator[0]
                var alphaNumerator : opt_float = pd.scanAlphaNumerator[0]

                -- update step size alpha
                var alpha = opt_float(0.0f)
                if (not [guardDivisionByZero]) or alphaDenominator > opt_float(0.0f) then
                    alpha = alphaNumerator/alphaDenominator 
                end
    
                var delta = pd.delta(idx)+alpha*pd.p(idx)       -- do a descent step
                pd.delta(idx) = delta

                var r = pd.r(idx)-alpha*pd.Ap_X(idx)				-- update residuum
                pd.r(idx) = r										-- store for next kernel call

                var pre = pd.preconditioner(idx)
                if not problemSpec.usepreconditioner then
                    pre = opt_float(1.0f)
                end
        
                var z = pre*r										-- apply pre-conditioner M^-1
                pd.z(idx) = z;										-- save for next kernel call

                betaNum = z:dot(r)									-- compute x-th term of the numerator of beta

                if [problemSpec:UsesLambda()] then
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = 0.5*(delta:dot(r + pd.b(idx))) 
                end
            end
            
            unknownWideReduction(idx,betaNum,pd.scanBetaNumerator)
            if [problemSpec:UsesLambda()] then
                unknownWideReduction(idx,q,pd.q)
            end
        end

        terra kernels.PCGStep2_1stHalf(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var alphaDenominator : opt_float = pd.scanAlphaDenominator[0]
                var alphaNumerator : opt_float = pd.scanAlphaNumerator[0]
                -- update step size alpha
                var alpha = opt_float(0.0f)
                if (not [guardDivisionByZero]) or alphaDenominator > opt_float(0.0f) then
                    alpha = alphaNumerator/alphaDenominator 
                end
                pd.delta(idx) = pd.delta(idx)+alpha*pd.p(idx)       -- do a descent step
            end
        end

        terra kernels.PCGStep2_2ndHalf(pd : PlanData)
            var betaNum = opt_float(0.0f) 
            var q = opt_float(0.0f) 
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                -- Recompute residual
                var Ax = pd.Adelta(idx)
                var b = pd.b(idx)
                var r = b - Ax
                pd.r(idx) = r

                var pre = pd.preconditioner(idx)
                if not problemSpec.usepreconditioner then
                    pre = opt_float(1.0f)
                end
                var z = pre*r       -- apply pre-conditioner M^-1
                pd.z(idx) = z;      -- save for next kernel call
                betaNum = z:dot(r)        -- compute x-th term of the numerator of beta
                if [problemSpec:UsesLambda()] then
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = 0.5*(pd.delta(idx):dot(r + b)) 
                end
            end
            unknownWideReduction(idx,betaNum,pd.scanBetaNumerator) 
            if [problemSpec:UsesLambda()] then
                unknownWideReduction(idx,q,pd.q)
            end
        end


        terra kernels.PCGStep3(pd : PlanData)			
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
            
                var rDotzNew : opt_float = pd.scanBetaNumerator[0]	-- get new numerator
                var rDotzOld : opt_float = pd.scanAlphaNumerator[0]	-- get old denominator

                var beta : opt_float = opt_float(0.0f)
                if (not [guardDivisionByZero]) or rDotzOld > opt_float(0.0f) then
                    beta = rDotzNew/rDotzOld
                end
                pd.p(idx) = pd.z(idx)+beta*pd.p(idx)			    -- update decent direction
            end
        end
    
        terra kernels.PCGLinearUpdate(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.parameters.X(idx) + pd.delta(idx)
            end
        end	
        
        terra kernels.revertUpdate(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.parameters.X(idx) = pd.prevX(idx)
            end
        end	

        terra kernels.computeAdelta(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.Adelta(idx) = fmap.applyJTJ(idx, pd.parameters, pd.delta, pd.CtC)
            end
        end

        terra kernels.savePreviousUnknowns(pd : PlanData)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                pd.prevX(idx) = pd.parameters.X(idx)
            end
        end 

        terra kernels.computeCost(pd : PlanData)
            var cost : opt_float = opt_float(0.0f)
            var idx : Index
            if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                var params = pd.parameters
                cost = fmap.cost(idx, params)
            end

            cost = util.warpReduce(cost)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scratch, cost)
            end
        end
        if not fmap.dumpJ then
            terra kernels.saveJToCRS(pd : PlanData)
            end
        else
            terra kernels.saveJToCRS(pd : PlanData)
                var idx : Index
                var [parametersSym] = &pd.parameters
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                    [generateDumpJ(fmap.derivedfrom,fmap.dumpJ,idx,pd)]
                end
            end
        end
    

        if fmap.precompute then
            terra kernels.precompute(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() then
                   fmap.precompute(idx,pd.parameters)
                end
            end
        end
        if problemSpec:UsesLambda() then
            terra kernels.PCGComputeCtC(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then 
                    var CtC = fmap.computeCtC(idx, pd.parameters)
                    pd.CtC(idx) = CtC    
                end 
            end

            terra kernels.PCGSaveSSq(pd : PlanData)
                var idx : Index
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then 
                    pd.SSq(idx) = pd.preconditioner(idx)       
                end 
            end

            terra kernels.PCGFinalizeDiagonal(pd : PlanData)
                var idx : Index
                var d = opt_float(0.0f)
                var q = opt_float(0.0f)
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then 
                    var unclampedCtC = pd.CtC(idx)
                    var invS_iiSq : unknownElement = opt_float(1.0f)
                    if [initialization_parameters.jacobiScaling == JacobiScalingType.ONCE_PER_SOLVE] then
                        invS_iiSq = opt_float(1.0f) / pd.SSq(idx)
                    elseif [initialization_parameters.jacobiScaling == JacobiScalingType.EVERY_ITERATION] then 
                        invS_iiSq = opt_float(1.0f) / pd.preconditioner(idx)
                    end -- else if  [initialization_parameters.jacobiScaling == JacobiScalingType.NONE] then invS_iiSq == 1
                    var clampMultiplier = invS_iiSq / pd.parameters.trust_region_radius
                    var minVal = pd.parameters.min_lm_diagonal * clampMultiplier
                    var maxVal = pd.parameters.max_lm_diagonal * clampMultiplier
                    var CtC = clamp(unclampedCtC, minVal, maxVal)
                    pd.CtC(idx) = CtC
                    
                    -- Calculate true preconditioner, taking into account the diagonal
                    var pre = opt_float(1.0f) / (CtC+pd.parameters.trust_region_radius*unclampedCtC) 
                    pd.preconditioner(idx) = pre
                    var residuum = pd.r(idx)
                    pd.b(idx) = residuum -- copy over to b
                    var p = pre*residuum    -- apply pre-conditioner M^-1
                    pd.p(idx) = p
                    d = residuum:dot(p)
                    -- computeQ    
                    -- Right side is -2 of CERES versions, left is just negative version, 
                    --  so after the dot product, just need to multiply by 2 to recover value identical to CERES  
                    q = 0.5*(pd.delta(idx):dot(residuum + residuum)) 
                end    
                unknownWideReduction(idx,q,pd.q)
                unknownWideReduction(idx,d,pd.scanAlphaNumerator)
            end

            terra kernels.computeModelCost(pd : PlanData)            
                var cost : opt_float = opt_float(0.0f)
                var idx : Index
                if idx:initFromCUDAParams() and not fmap.exclude(idx,pd.parameters) then
                    var params = pd.parameters              
                    cost = fmap.modelcost(idx, params, pd.delta)
                end

                cost = util.warpReduce(cost)
                if (util.laneid() == 0) then
                    util.atomicAdd(pd.modelCost, cost)
                end
            end

        end -- :UsesLambda()
	    return kernels
	end
	
	function delegate.GraphFunctions(graphname,fmap,ES)
	    --print("ES-graph",fmap.derivedfrom)
	    local kernels = {}
        terra kernels.PCGInit1_Graph(pd : PlanData)
            var tIdx = 0
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                fmap.evalJTF(tIdx, pd.parameters, pd.r, pd.preconditioner)
            end
        end    
        
    	terra kernels.PCGStep1_Graph(pd : PlanData)
            var d = opt_float(0.0f)
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
               d = d + fmap.applyJTJ(tIdx, pd.parameters, pd.p, pd.Ap_X)
            end 
            if not [multistep_alphaDenominator_compute] then
                d = util.warpReduce(d)
                if (util.laneid() == 0) then
                    util.atomicAdd(pd.scanAlphaDenominator, d)
                end
            end
        end

        terra kernels.computeAdelta_Graph(pd : PlanData)
            var tIdx = 0 
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                fmap.applyJTJ(tIdx, pd.parameters, pd.delta, pd.Adelta)
            end
        end

        terra kernels.computeCost_Graph(pd : PlanData)
            var cost : opt_float = opt_float(0.0f)
            var tIdx = 0
            if util.getValidGraphElement(pd,[graphname],&tIdx) then
                cost = fmap.cost(tIdx, pd.parameters)
            end 
            cost = util.warpReduce(cost)
            if (util.laneid() == 0) then
                util.atomicAdd(pd.scratch, cost)
            end
        end
        if not fmap.dumpJ then
            terra kernels.saveJToCRS_Graph(pd : PlanData)
            end
        else
            terra kernels.saveJToCRS_Graph(pd : PlanData)
                var tIdx = 0
                var [parametersSym] = &pd.parameters
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    [generateDumpJ(fmap.derivedfrom,fmap.dumpJ,tIdx,pd)]
                end
            end
        end
        if problemSpec:UsesLambda() then
            terra kernels.PCGComputeCtC_Graph(pd : PlanData)
                var tIdx = 0
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    fmap.computeCtC(tIdx, pd.parameters, pd.CtC)
                end
            end    

            terra kernels.computeModelCost_Graph(pd : PlanData)          
                var cost : opt_float = opt_float(0.0f)
                var tIdx = 0
                if util.getValidGraphElement(pd,[graphname],&tIdx) then
                    cost = fmap.modelcost(tIdx, pd.parameters, pd.delta)
                end 
                cost = util.warpReduce(cost)
                if (util.laneid() == 0) then
                    util.atomicAdd(pd.modelCost, cost)
                end
            end
        end

	    return kernels
	end
	
	local gpu = util.makeGPUFunctions(problemSpec, PlanData, delegate, {"PCGInit1",
                                                                        "PCGInit1_Finish",
                                                                        "PCGComputeCtC",
                                                                        "PCGFinalizeDiagonal",
                                                                        "PCGStep1",
                                                                        "PCGStep1_Finish",
                                                                        "PCGStep2",
                                                                        "PCGStep2_1stHalf",
                                                                        "PCGStep2_2ndHalf",
                                                                        "PCGStep3",
                                                                        "PCGLinearUpdate",
                                                                        "revertUpdate",
                                                                        "savePreviousUnknowns",
                                                                        "computeCost",
                                                                        "PCGSaveSSq",
                                                                        "precompute",
                                                                        "computeAdelta",
                                                                        "computeAdelta_Graph",
                                                                        "PCGInit1_Graph",
                                                                        "PCGComputeCtC_Graph",
                                                                        "PCGStep1_Graph",
                                                                        "computeCost_Graph",
                                                                        "computeModelCost",
                                                                        "computeModelCost_Graph",
                                                                        "saveJToCRS",
                                                                        "saveJToCRS_Graph"
                                                                        })

    local terra computeCost(pd : &PlanData) : opt_float
        C.cudaMemset(pd.scratch, 0, sizeof(opt_float))
        gpu.computeCost(pd)
        gpu.computeCost_Graph(pd)
        var f : opt_float
        C.cudaMemcpy(&f, pd.scratch, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local terra computeModelCost(pd : &PlanData) : opt_float
        C.cudaMemset(pd.modelCost, 0, sizeof(opt_float))
        gpu.computeModelCost(pd)
        gpu.computeModelCost_Graph(pd)
        var f : opt_float
        C.cudaMemcpy(&f, pd.modelCost, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local sqrtf = util.cpuMath.sqrt

    local terra fetchQ(pd : &PlanData) : opt_float
        var f : opt_float
        C.cudaMemcpy(&f, pd.q, sizeof(opt_float), C.cudaMemcpyDeviceToHost)
        return f
    end

    local computeModelCostChange
    
    if problemSpec:UsesLambda() then
        terra computeModelCostChange(pd : &PlanData) : opt_float
            var cost = pd.prevCost
            var model_cost = computeModelCost(pd)
            logSolver(" cost=%f \n",cost)
            logSolver(" model_cost=%f \n",model_cost)
            var model_cost_change = cost - model_cost
            logSolver(" model_cost_change=%f \n",model_cost_change)
            return model_cost_change
        end
    end

    local terra GetToHost(ptr : &opaque, N : int) : &int
        var r = [&int](C.malloc(sizeof(int)*N))
        C.cudaMemcpy(r,ptr,N*sizeof(int),C.cudaMemcpyDeviceToHost)
        return r
    end
    local cusparseInner,cusparseOuter
    if initialization_parameters.use_cusparse then
        terra cusparseOuter(pd : &PlanData)
            var [parametersSym] = &pd.parameters
            --logSolver("saving J...\n")
            gpu.saveJToCRS(pd)
            if isGraph then
                gpu.saveJToCRS_Graph(pd)
            end
            --logSolver("... done\n")
            if pd.JTJ_csrRowPtrA == nil then
                --allocate row
                --C.printf("alloc JTJ\n")
                var numrows = nUnknowns + 1
                cd(C.cudaMalloc([&&opaque](&pd.JTJ_csrRowPtrA),sizeof(int)*numrows))
                var endJTJalloc : util.TimerEvent
                pd.timer:startEvent("J^TJ alloc",nil,&endJTJalloc)
                cd(CUsp.cusparseXcsrgemmNnz(pd.handle, CUsp.CUSPARSE_OPERATION_TRANSPOSE, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      nUnknowns,nUnknowns,nResidualsExp,
                                      pd.desc,nnzExp,pd.J_csrRowPtrA,pd.J_csrColIndA,
                                      pd.desc,nnzExp,pd.J_csrRowPtrA,pd.J_csrColIndA,
                                      pd.desc,pd.JTJ_csrRowPtrA, &pd.JTJ_nnz))
                pd.timer:endEvent(nil,endJTJalloc)
                
                cd(C.cudaMalloc([&&opaque](&pd.JTJ_csrColIndA), sizeof(int)*pd.JTJ_nnz))
                cd(C.cudaMalloc([&&opaque](&pd.JTJ_csrValA), sizeof(float)*pd.JTJ_nnz))
                cd(C.cudaThreadSynchronize())
            end
            
            var endJTJmm : util.TimerEvent
            pd.timer:startEvent("JTJ multiply",nil,&endJTJmm)
    
            cd(CUsp.cusparseScsrgemm(pd.handle, CUsp.CUSPARSE_OPERATION_TRANSPOSE, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
           nUnknowns,nUnknowns,nResidualsExp,
           pd.desc,nnzExp,pd.J_csrValA,pd.J_csrRowPtrA,pd.J_csrColIndA,
           pd.desc,nnzExp,pd.J_csrValA,pd.J_csrRowPtrA,pd.J_csrColIndA,
           pd.desc, pd.JTJ_csrValA, pd.JTJ_csrRowPtrA,pd.JTJ_csrColIndA ))
           pd.timer:endEvent(nil,endJTJmm)
           
            var endJtranspose : util.TimerEvent
            pd.timer:startEvent("J_transpose",nil,&endJtranspose)
            cd(CUsp.cusparseScsr2csc(pd.handle,nResidualsExp, nUnknowns,nnzExp,
                                 pd.J_csrValA,pd.J_csrRowPtrA,pd.J_csrColIndA,
                                 pd.JT_csrValA,pd.JT_csrColIndA,pd.JT_csrRowPtrA,
                                 CUsp.CUSPARSE_ACTION_NUMERIC,CUsp.CUSPARSE_INDEX_BASE_ZERO))
            pd.timer:endEvent(nil,endJtranspose)
        end
        terra cusparseInner(pd : &PlanData)
            var [parametersSym] = &pd.parameters
            
            if false then
                C.printf("begin debug dump\n")
                var J_csrColIndA = GetToHost(pd.J_csrColIndA,nnzExp)
                var J_csrRowPtrA = GetToHost(pd.J_csrRowPtrA,nResidualsExp + 1)
                for i = 0,nResidualsExp do
                    var b,e = J_csrRowPtrA[i],J_csrRowPtrA[i+1]
                    if b >= e or b < 0 or b >= nnzExp or e < 0 or e > nnzExp then
                        C.printf("ERROR: %d %d %d (total = %d)\n",i,b,e,nResidualsExp)
                    end
                    --C.printf("residual %d -> {%d,%d}\n",i,b,e)
                    for j = b,e do
                        if J_csrColIndA[j] < 0 or J_csrColIndA[j] >= nnzExp then
                            C.printf("ERROR: j %d (total = %d)\n",j,J_csrColIndA[j])
                        end
                        if j ~= b and J_csrColIndA[j-1] >= J_csrColIndA[j] then
                            C.printf("ERROR: sort j[%d] = %d, j[%d] = %d\n",j-1,J_csrColIndA[j-1],j,J_csrColIndA[j])
                        end
                        --C.printf("colindex: %d\n",J_csrColIndA[j])
                    end
                end
                C.printf("end debug dump\n")
            end
            
            var consts = array(0.f,1.f,2.f)
            cd(C.cudaMemset(pd.Ap_X._contiguousallocation, -1, sizeof(float)*nUnknowns))
            
            if initialization_parameters.use_fused_jtj then
                var endJTJp : util.TimerEvent
                pd.timer:startEvent("J^TJp",nil,&endJTJp)
                cd(CUsp.cusparseScsrmv(
                            pd.handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                            nUnknowns, nUnknowns,pd.JTJ_nnz,
                            &consts[1], pd.desc,
                            pd.JTJ_csrValA, 
                            pd.JTJ_csrRowPtrA, pd.JTJ_csrColIndA,
                            [&float](pd.p._contiguousallocation),
                            &consts[0], [&float](pd.Ap_X._contiguousallocation)
                        ))
                pd.timer:endEvent(nil,endJTJp)
            else
                var endJp : util.TimerEvent
                pd.timer:startEvent("Jp",nil,&endJp)
                cd(CUsp.cusparseScsrmv(
                            pd.handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                            nResidualsExp, nUnknowns,nnzExp,
                            &consts[1], pd.desc,
                            pd.J_csrValA, 
                            pd.J_csrRowPtrA, pd.J_csrColIndA,
                            [&float](pd.p._contiguousallocation),
                            &consts[0], pd.Jp
                        ))
                pd.timer:endEvent(nil,endJp)
                var endJT : util.TimerEvent
                pd.timer:startEvent("J^T",nil,&endJT)
                cd(CUsp.cusparseScsrmv(
                            pd.handle, CUsp.CUSPARSE_OPERATION_NON_TRANSPOSE,
                            nUnknowns, nResidualsExp, nnzExp,
                            &consts[1], pd.desc,
                            pd.JT_csrValA, 
                            pd.JT_csrRowPtrA, pd.JT_csrColIndA,
                            pd.Jp,
                            &consts[0],[&float](pd.Ap_X._contiguousallocation) 
                        ))
                pd.timer:endEvent(nil,endJT)
            end
        end
    else
        terra cusparseInner(pd : &PlanData) end
        terra cusparseOuter(pd : &PlanData) end
    end

	local terra init(data_ : &opaque, params_ : &&opaque)
	   var pd = [&PlanData](data_)
	   pd.timer:init()
	   pd.timer:startEvent("overall",nil,&pd.endSolver)
       [util.initParameters(`pd.parameters,problemSpec,params_,true)]
       var [parametersSym] = &pd.parameters
        escape if initialization_parameters.use_cusparse then emit quote
            if pd.J_csrValA == nil then
                cd(CUsp.cusparseCreateMatDescr( &pd.desc ))
                cd(CUsp.cusparseSetMatType( pd.desc,CUsp.CUSPARSE_MATRIX_TYPE_GENERAL ))
                cd(CUsp.cusparseSetMatIndexBase( pd.desc,CUsp.CUSPARSE_INDEX_BASE_ZERO ))
                cd(CUsp.cusparseCreate( &pd.handle ))
                
                logSolver("nnz = %s\n",[tostring(nnzExp)])
                logSolver("nResiduals = %s\n",[tostring(nResidualsExp)])
                logSolver("nnz = %d, nResiduals = %d\n",int(nnzExp),int(nResidualsExp))
                
                -- J alloc
                C.cudaMalloc([&&opaque](&(pd.J_csrValA)), sizeof(opt_float)*nnzExp)
                C.cudaMalloc([&&opaque](&(pd.J_csrColIndA)), sizeof(int)*nnzExp)
                C.cudaMemset(pd.J_csrColIndA,-1,sizeof(int)*nnzExp)
                C.cudaMalloc([&&opaque](&(pd.J_csrRowPtrA)), sizeof(int)*(nResidualsExp+1))
                
                -- J^T alloc
                C.cudaMalloc([&&opaque](&pd.JT_csrValA), nnzExp*sizeof(float))
                C.cudaMalloc([&&opaque](&pd.JT_csrColIndA), nnzExp*sizeof(int))
                C.cudaMalloc([&&opaque](&pd.JT_csrRowPtrA), (nUnknowns + 1) *sizeof(int))
                
                -- Jp alloc
                cd(C.cudaMalloc([&&opaque](&pd.Jp), nResidualsExp*sizeof(float)))
                
                -- write J_csrRowPtrA end
                var nnz = nnzExp
                C.printf("setting rowptr[%d] = %d\n",nResidualsExp,nnz)
                cd(C.cudaMemcpy(&pd.J_csrRowPtrA[nResidualsExp],&nnz,sizeof(int),C.cudaMemcpyHostToDevice))
            end
        end end end

	   pd.solverparameters.nIter = 0
       escape 
            if problemSpec:UsesLambda() then
              emit quote 
                pd.parameters.trust_region_radius       = pd.solverparameters.trust_region_radius
                pd.parameters.radius_decrease_factor    = pd.solverparameters.radius_decrease_factor
                pd.parameters.min_lm_diagonal           = pd.solverparameters.min_lm_diagonal
                pd.parameters.max_lm_diagonal           = pd.solverparameters.max_lm_diagonal
              end
	        end 
       end
	   gpu.precompute(pd)
	   pd.prevCost = computeCost(pd)
	end

	local terra cleanup(pd : &PlanData)
        logSolver("final cost=%f\n", pd.prevCost)
        pd.timer:endEvent(nil,pd.endSolver)
        pd.timer:evaluate()
        pd.timer:cleanup()
    end

	local terra step(data_ : &opaque, params_ : &&opaque)
        var pd = [&PlanData](data_)
        var residual_reset_period : int         = pd.solverparameters.residual_reset_period
        var min_relative_decrease : opt_float   = pd.solverparameters.min_relative_decrease
        var min_trust_region_radius : opt_float = pd.solverparameters.min_trust_region_radius
        var max_trust_region_radius : opt_float = pd.solverparameters.max_trust_region_radius
        var q_tolerance : opt_float             = pd.solverparameters.q_tolerance
        var function_tolerance : opt_float      = pd.solverparameters.function_tolerance
        var Q0 : opt_float
        var Q1 : opt_float
		[util.initParameters(`pd.parameters,problemSpec, params_,false)]
		if pd.solverparameters.nIter < pd.solverparameters.nIterations then
			C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset
			C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))	--scan in PCGInit1 requires reset

			gpu.PCGInit1(pd)
			if isGraph then
				gpu.PCGInit1_Graph(pd)	
				gpu.PCGInit1_Finish(pd)	
			end

            escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        C.cudaMemset(pd.scanAlphaNumerator, 0, sizeof(opt_float))
                        C.cudaMemset(pd.q, 0, sizeof(opt_float))
                        if [initialization_parameters.jacobiScaling == JacobiScalingType.ONCE_PER_SOLVE] and pd.solverparameters.nIter == 0 then
                            gpu.PCGSaveSSq(pd)
                        end
                        gpu.PCGComputeCtC(pd)
                        gpu.PCGComputeCtC_Graph(pd)
                        -- This also computes Q
                        gpu.PCGFinalizeDiagonal(pd)
                        Q0 = fetchQ(pd)
                    end
                end
            end
            logDebugCudaOptFloat("init scanAlphaNumerator", pd.scanAlphaNumerator)
            cusparseOuter(pd)
            for lIter = 0, pd.solverparameters.lIterations do				

                C.cudaMemset(pd.scanAlphaDenominator, 0, sizeof(opt_float))
                C.cudaMemset(pd.q, 0, sizeof(opt_float))

                if not initialization_parameters.use_cusparse then
    				gpu.PCGStep1(pd)
    				if isGraph then
    					gpu.PCGStep1_Graph(pd)
    				end
                end

				-- only does anything if initialization_parameters.use_cusparse is true
                cusparseInner(pd)

                if multistep_alphaDenominator_compute then
                    gpu.PCGStep1_Finish(pd)
                end
				logDebugCudaOptFloat("scanAlphaDenominator", pd.scanAlphaDenominator)
				C.cudaMemset(pd.scanBetaNumerator, 0, sizeof(opt_float))
				
				if [problemSpec:UsesLambda()] and ((lIter + 1) % residual_reset_period) == 0 then
                    gpu.PCGStep2_1stHalf(pd)
                    gpu.computeAdelta(pd)
                    if isGraph then
                        gpu.computeAdelta_Graph(pd)
                    end
                    gpu.PCGStep2_2ndHalf(pd)
                else
                    gpu.PCGStep2(pd)
                end
                logDebugCudaOptFloat("scanBetaNumerator", pd.scanBetaNumerator)
                gpu.PCGStep3(pd)

				-- save new rDotz for next iteration
				C.cudaMemcpy(pd.scanAlphaNumerator, pd.scanBetaNumerator, sizeof(opt_float), C.cudaMemcpyDeviceToDevice)	
				
				if [problemSpec:UsesLambda()] then
	                Q1 = fetchQ(pd)
	                var zeta = [opt_float](lIter+1)*(Q1 - Q0) / Q1 
                    --logSolver("%d: Q0(%g) Q1(%g), zeta(%g)\n", lIter, Q0, Q1, zeta)
	                if zeta < q_tolerance then
                        logSolver("zeta=%.18g, breaking at iteration: %d\n", zeta, (lIter+1))
	                    break
	                end
	                Q0 = Q1
				end
			end
			

            var model_cost_change : opt_float

            escape if problemSpec:UsesLambda() then
                emit quote 
                    model_cost_change = computeModelCostChange(pd)
                    gpu.savePreviousUnknowns(pd)
                end
            end end

			gpu.PCGLinearUpdate(pd)    
			gpu.precompute(pd)
			var newCost = computeCost(pd)

			escape 
                if problemSpec:UsesLambda() then
                    emit quote
                        var cost_change = pd.prevCost - newCost
                        
                        
                        -- See CERES's TrustRegionStepEvaluator::StepAccepted() for a more complicated version of this
                        var relative_decrease = cost_change / model_cost_change
                        if cost_change >= 0 and relative_decrease > min_relative_decrease then
                            var absolute_function_tolerance = pd.prevCost * function_tolerance
                            if cost_change <= absolute_function_tolerance then
                                logSolver("\nFunction tolerance reached, exiting\n")
                                cleanup(pd)
                                return 0
                            end

                            var step_quality = relative_decrease
                            var min_factor = 1.0/3.0
                            var tmp_factor = 1.0 - util.cpuMath.pow(2.0 * step_quality - 1.0, 3.0)
                            pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / util.cpuMath.fmax(min_factor, tmp_factor)
                            pd.parameters.trust_region_radius = util.cpuMath.fmin(pd.parameters.trust_region_radius, max_trust_region_radius)
                            pd.parameters.radius_decrease_factor = 2.0

                            pd.prevCost = newCost
                        else 
                            gpu.revertUpdate(pd)

                            pd.parameters.trust_region_radius = pd.parameters.trust_region_radius / pd.parameters.radius_decrease_factor
                            logSolver(" trust_region_radius=%f \n", pd.parameters.trust_region_radius)
                            pd.parameters.radius_decrease_factor = 2.0 * pd.parameters.radius_decrease_factor
                            if pd.parameters.trust_region_radius <= min_trust_region_radius then
                                logSolver("\nTrust_region_radius is less than the min, exiting\n")
                                cleanup(pd)
                                return 0
                            end
                            logSolver("REVERT\n")
                            gpu.precompute(pd)
                        end
                    end
                else
                    emit quote
                        logSolver("cost: %f -> %f\n", pd.prevCost, newCost)
                        pd.prevCost = newCost 
                    end
                end 
            end

            --[[ 
            To match CERES we would check for termination:
            iteration_summary_.gradient_max_norm <= options_.gradient_tolerance
            ]]

            pd.solverparameters.nIter = pd.solverparameters.nIter + 1
            return 1
        else
            cleanup(pd)
            return 0
        end
    end

    local terra cost(data_ : &opaque) : double
        var pd = [&PlanData](data_)
        return [double](pd.prevCost)
    end

    local terra initializeSolverParameters(params : &SolverParameters)
        escape
            -- for each value in solver_parameter_defaults, assign to params
            for name,value in pairs(solver_parameter_defaults) do
                local foundVal = false
                -- TODO, more elegant solution to this
                for _,entry in ipairs(SolverParameters.entries) do
                    if entry.field == name then
                        foundVal = true
                        emit quote params.[name] = [entry.type]([value])
                        end
                        break
                    end
                end
                if not foundVal then
                    print("Tried to initialize "..name.." but not found")
                end
            end
        end
    end

    local terra setSolverParameter(data_ : &opaque, name : rawstring, value : &opaque) 
        var pd = [&PlanData](data_)
        var success = false
        escape
            -- Instead of building a table datastructure, 
            -- explicitly emit an if-statement chain for setting the parameter
            for _,entry in ipairs(SolverParameters.entries) do
                emit quote
                    if C.strcmp([entry.field],name)==0 then
                        pd.solverparameters.[entry.field] = @[&entry.type]([value])
                        return
                    end
                end
            end
        end
        logSolver("Warning: tried to set nonexistent solver parameter %s\n", name)
    end

    local terra free(data_ : &opaque)
        var pd = [&PlanData](data_)
        pd.delta:freeData()
        pd.r:freeData()
        pd.b:freeData()
        pd.Adelta:freeData()
        pd.z:freeData()
        pd.p:freeData()
        pd.Ap_X:freeData()
        pd.CtC:freeData()
        pd.SSq:freeData()
        pd.preconditioner:freeData()
        pd.g:freeData()
        pd.prevX:freeData()

        [util.freePrecomputedImages(`pd.parameters,problemSpec)]

        cd(C.cudaFree([&opaque](pd.scanAlphaNumerator)))
        cd(C.cudaFree([&opaque](pd.scanBetaNumerator)))
        cd(C.cudaFree([&opaque](pd.scanAlphaDenominator)))
        cd(C.cudaFree([&opaque](pd.modelCost)))

        cd(C.cudaFree([&opaque](pd.scratch)))
        cd(C.cudaFree([&opaque](pd.q)))
        
        -- TODO: correctly deallocate when using cusparse
        pd.J_csrValA = nil
        pd.JTJ_csrRowPtrA = nil
        -- TODO: Rearchitect to enable deleting of the plan data
    end

	local terra makePlan() : &opt.Plan
		var pd = PlanData.alloc()
		pd.plan.data = pd
		pd.plan.init,pd.plan.step,pd.plan.cost,pd.plan.setsolverparameter,pd.plan.free = init,step,cost,setSolverParameter,free
		pd.delta:initGPU()
		pd.r:initGPU()
        pd.b:initGPU()
        pd.Adelta:initGPU()
		pd.z:initGPU()
		pd.p:initGPU()
		pd.Ap_X:initGPU()
        pd.CtC:initGPU()
        pd.SSq:initGPU()
		pd.preconditioner:initGPU()
		pd.g:initGPU()
        pd.prevX:initGPU()

        initializeSolverParameters(&pd.solverparameters)
		
		[util.initPrecomputedImages(`pd.parameters,problemSpec)]	
		C.cudaMalloc([&&opaque](&(pd.scanAlphaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanBetaNumerator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.scanAlphaDenominator)), sizeof(opt_float))
		C.cudaMalloc([&&opaque](&(pd.modelCost)), sizeof(opt_float))
		
		C.cudaMalloc([&&opaque](&(pd.scratch)), sizeof(opt_float))
        C.cudaMalloc([&&opaque](&(pd.q)), sizeof(opt_float))
		pd.J_csrValA = nil
		pd.JTJ_csrRowPtrA = nil
		return &pd.plan
	end

	return makePlan
end
