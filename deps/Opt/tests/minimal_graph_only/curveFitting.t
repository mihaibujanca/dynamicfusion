N,U = Dim("N",0), Dim("U",1)
funcParams =   Unknown("funcParams", opt_float2, {U}, 0)
data =         Image("data", opt_float2, {N}, 1)
local G = Graph("G", 2, "d", {N}, 3, "p", {U}, 4)
UsePreconditioner(true)

x,y = data(G.d)(0),data(G.d)(1)
a,b = funcParams(G.p)(0),funcParams(G.p)(1)
Energy(y - (a*cos(b*x) + b*sin(a*x))) 
