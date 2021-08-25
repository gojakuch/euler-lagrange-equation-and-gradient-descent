"""
A callable type that wraps:\n
    f      - the solution function itself;\n
    losses - vector of values of the loss function;
"""
struct ELSolution
    f::Function
    losses::Vector{Float64}
end

(sol::ELSolution)(args...) = sol.f(args...)
