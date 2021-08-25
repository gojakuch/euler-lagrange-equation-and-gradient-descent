using Plots
include("VarCalcSolver.jl")

sol = solveEulerLagrangeGD((x, y, dydx)->sqrt(1+dydx^2), (-5, -5), (5, 5), approx_degree=2, maxiter=2000, sections=200)
y = sol
losses = sol.losses

xs = range(-5, stop=5, length=5000)
yf = [y(x) for x in xs]
println("y(-5): $(y(-5))")
println("y(5): $(y(5))")
p1 = plot(losses, label="loss(time)", lw=1, color=:purple)
p2 = plot(xs, yf, label="y(x)", lw=1, color=:pink); plot!(p2, xs, [x for x in xs], label="true", lw=1, color=:purple);
plot(p1, p2, layout=(1, 2), title="∂F/∂y - d/dx(∂F/∂y') ≈ 0")
