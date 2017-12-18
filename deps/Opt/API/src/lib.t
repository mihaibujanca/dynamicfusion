return function(P)
    local terms = terralib.newlist()

    local L = {}

    function L.Energy(...)
        for i,e in ipairs {...} do
            terms:insert(e)
        end
    end

    function L.Result() return P:Cost(unpack(terms)) end
    function L.All(v)
        local r = 1
        for i = 0,v:size() - 1 do
            r = r * v(i)
        end
        return r
    end

    function L.Reduce(fn,init)
        return function(...)
            local r = init
            for _,e in ipairs {...} do
                r = fn(r,e)
            end
            return r
        end
    end
    L.And = L.Reduce(1,ad.and_)
    L.Or = L.Reduce(0,ad.or_)
    L.Not = ad.not_
    
    function L.UsePreconditioner(...) return P:UsePreconditioner(...) end
    -- alas for Image/Array
    function L.Array(...) return P:Image(...) end
    function L.ComputedArray(...) return P:ComputedImage(...) end

    function L.Matrix3x3Mul(matrix,v)
        return ad.Vector(
            matrix(0)*v(0)+matrix(1)*v(1)+matrix(2)*v(2),
            matrix(3)*v(0)+matrix(4)*v(1)+matrix(5)*v(2),
            matrix(6)*v(0)+matrix(7)*v(1)+matrix(8)*v(2))
    end

    function L.Dot3(v0,v1)
        return v0(0)*v1(0)+v0(1)*v1(1)+v0(2)*v1(2)
    end

    function L.Sqrt(v)
        return ad.sqrt(v)
    end

    function L.normalize(v)
        return v / ad.sqrt(L.Dot3(v, v))
    end

    function L.length(v0, v1) 
        local diff = v0 - v1
        return ad.sqrt(L.Dot3(diff, diff))
    end

    function L.Slice(im,s,e)
        return setmetatable({},{
            __call = function(self,ind)
                if s + 1 == e then return im(ind)(s) end
                local t = terralib.newlist()
                for i = s,e - 1 do
                    local val = im(ind)
                    local chan = val(i)
                    t:insert(chan)
                end
                return ad.Vector(unpack(t))
            end })
    end

    function L.Rotate3D(a,v)
        local alpha, beta, gamma = a(0), a(1), a(2)
        local  CosAlpha, CosBeta, CosGamma, SinAlpha, SinBeta, SinGamma = ad.cos(alpha), ad.cos(beta), ad.cos(gamma), ad.sin(alpha), ad.sin(beta), ad.sin(gamma)
        local matrix = ad.Vector(
            CosGamma*CosBeta, 
            -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha, 
            SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha,
            SinGamma*CosBeta,
            CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha,
            -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha,
            -SinBeta,
            CosBeta*SinAlpha,
            CosBeta*CosAlpha)
        return L.Matrix3x3Mul(matrix,v)
    end
    function L.Rotate2D(angle, v)
	    local CosAlpha, SinAlpha = ad.cos(angle), ad.sin(angle)
        local matrix = ad.Vector(CosAlpha, -SinAlpha, SinAlpha, CosAlpha)
	    return ad.Vector(matrix(0)*v(0)+matrix(1)*v(1), matrix(2)*v(0)+matrix(3)*v(1))
    end
    L.Index = ad.Index
    L.SampledImage = ad.sampledimage

--
    function L.L_2_norm(v)
        -- TODO: check if scalar and just return
        return ad.sqrt(v:dot(v))
    end
    L.L_p_counter = 1
    function L.L_p(val, val_const, p, dims)
        local dist_const = L.L_2_norm(val_const)
        local eps = 0.0000001
        local C = ad.pow(dist_const+eps,(p-2))
        local sqrtC = ad.sqrt(C)
        local sqrtCImage = L.ComputedArray("L_p"..tostring(L.L_p_counter),dims,sqrtC)
        L.L_p_counter = L.L_p_counter + 1
        return sqrtCImage(0,0)*val
    end

    L.Select = ad.select
    function L.Stencil (lst)
        local i = 0
        return function()
            i = i + 1
            if not lst[i] then return nil
            else return unpack(lst[i]) end
        end
    end
    setmetatable(L,{__index = function(self,key)
        if type(P[key]) == "function" then
            return function(...) return P[key](P,...) end
        end
        if key ~= "select" and ad[key] then return ad[key] end
        if opt[key] then return opt[key] end
        return _G[key]
    end})
    return L
end