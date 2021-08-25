import Flux; gradient = Flux.gradient
include("ELSolution.jl")

"""
# solveEulerLagrangeGD(F, a, b; optionals...)
Approximates the solution *y* of\n
    min[I = ∫(F[x, y, y'])dx]\n
where ∫ is from a to b.\n
## Arguments:
    F(x, y, y')::Number (takes three numbers, returns a number)
    a - tuple/vector (x0, y0)
    b - tuple/vector (x1, y1)
## Optional arguments:
    early_stop    - a fraction of iterations after which to allow early stop or nothing (default=2//3)
    approx_degree - degree of the approximation polynomial (impacts the error in solution significantly) (default=3)
    maxiter       - number of iterations (default=100)
    sections      - number of sections to divide the interval into (default=100)
    α             - the "learning rate" for gradient descent (default=0.0005)
    δ             - the differential approximation used in computations (default=1e-8)
    β             - number by which we multiply the previous ∇Error (default=0)
Note that the algorithm has complexity O(maxiter*sections).
"""
function solveEulerLagrangeGD(F::Function, a, b; early_stop::Union{Number, Nothing}=2//3, approx_degree::Integer=3, maxiter::Integer=100, sections::Integer=100, α=0.0005, δ=1e-8, β=0)::ELSolution
    x0 = a[1]; x1 = b[1]; θ0 = a[2]; θ1 = b[2]
    w = rand(approx_degree+1)
    y = (x,w)->sum([w[i]*x^(length(w)-i) for i in 1:length(w)])
    dyd = (x,w)->sum([(length(w)-i)*w[i]*x^(length(w)-i-1) for i in 1:length(w)-1])
    Fdydx = (x, _y, dydx)->gradient(F,x,_y,dydx)[3]
    EL = function (x, _y, dydx, w)
         ∇F=[(e==nothing ? 0 : e) for e in gradient(F,x,_y,dydx)]
        # ∂F/∂y - d/dx(∂F/∂y')
        ∇F[2] - (Fdydx(x+δ,y(x+δ,w),dyd(x+δ,w))-Fdydx(x,_y,dydx))/δ
    end
    errf(x, w) = (EL(x, y(x,w), dyd(x,w), w))+abs((x-x0)/(x1-x0)*(y(x1,w)-θ1))+abs((x1-x+x0)/(x1-x0)*(y(x0,w)-θ0))
    losses = []
    prev∇ = zeros(length(w))
    for iter in 1:maxiter
        xn = x0
        ∇ = zeros(length(w))
        loss = 0
        for i in 1:sections
            xn += (x1-x0)/sections
            ∇ .+= gradient(errf, xn, w)[2]
            loss += errf(xn,w)
        end
        ∇ ./= sections
        push!(losses, loss/sections)
        if early_stop != nothing && iter/maxiter >= early_stop && losses[end] > losses[end-1]
            w .+= α*prev∇
            return ELSolution(x->y(x,w), losses[1:end-1])
        end
        ∇ = ∇ + β*prev∇
        w .-= α*∇
        prev∇ = ∇
    end
    ELSolution(x->y(x,w), losses)
end
