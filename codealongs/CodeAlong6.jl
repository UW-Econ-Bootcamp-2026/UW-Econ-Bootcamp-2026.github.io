################################################################################
# Load Packages
################################################################################

using Optim, Interpolations, Plots

################################################################################
# Fibonacci: store the sub-problem
################################################################################

# Straight from the definition: F(n) = F(n-1) + F(n-2).
function fib_naive(n::Int64)
    if n <= 1
        return n
    else
        return fib_naive(n-1) + fib_naive(n-2)
    end
end

# fib_naive(6) = fib_naive(5) + fib_naive(4)
#              = [fib_naive(4) + fib_naive(3)] + fib_naive(4)
#                 ^^^^^^^^^^^^                   ^^^^^^^^^^^^ computed twice!
# The number of calls grows exponentially, even though there are only n
# distinct answers.

# Dynamic programming: compute each sub-problem once, store it, reuse it.
function fib_dp(n::Int64)

    fib_vec = zeros(n+1) # include fib(0), index = n + 1
    fib_vec[2] = 1

    for i = 3:n+1
        fib_vec[i] = fib_vec[i-1] + fib_vec[i-2]
    end

    return fib_vec[n+1]
end

#Fast:
@time fib_dp(42)
#Slow:
@time fib_naive(42)

@time fib_dp(1000)
# @time fib_naive(1000) #don't even try to run this.

# fib_vec is a value function: it stores the answer to "what is the nth
# Fibonacci number" the way V stores "what is the best you can do from k".


################################################################################
# The Stochastic Growth Model
################################################################################

# V(k, z) = max_{k'} { log(c) + β * Σ_{z'} Π(z'|z) V(k', z') }
#     c = z*k^α + (1-δ)*k - k'
#
# We keep the optimization machinery interpolate V, let Optim choose k' continuously,
# so 100 grid points is plenty.

@kwdef struct ModelParameters

    β::Float64 = 0.99
    δ::Float64 = 0.025
    α::Float64 = 0.36

    k_grid::Vector{Float64} = collect(range(0.1, length = 100, stop = 45.0))
    N_k::Int64 = length(k_grid)

    # Π[i, j] = P(z' = z_grid[j] | z = z_grid[i]), so rows sum to one.
    z_grid::Vector{Float64} = [1.25, 0.2]
    N_z::Int64 = length(z_grid)
    Π::Matrix{Float64} = [0.977 0.023; 0.074 0.926]

    tol::Float64 = 10^-5
    max_iter::Int64 = 10^4


end

# V and kp are now matrices: rows index capital, columns index productivity.
struct ModelSolutions

    V::Matrix{Float64}
    kp::Matrix{Float64}

end

function initialize(; kwargs...)
    para = ModelParameters(; kwargs...)
    V = zeros(para.N_k, para.N_z)
    kp = zeros(para.N_k, para.N_z)
    sols = ModelSolutions(V, kp)
    return para, sols
end



### Bellman operator
function bellman(para, sols)
    (; β, δ, α, k_grid, z_grid, Π) = para
    (; V, kp) = sols

    V_next = zero(V)
    kp_next = zero(kp)

    for i_z in eachindex(z_grid)

        # EV[i] = Σ_{z'} Π[i_z, z'] * V(k_grid[i], z'). Note the ROW of Π: we
        # condition on today's z and sum over tomorrow's. This depends only on
        # i_z, so compute it once out here rather than N_k times below.
        EV = V * Π[i_z, :]

        # Interpolate the expected continuation value, not each V(., z')
        # separately -- expectation is linear, so this is equivalent and cheaper.
        EV_interp = linear_interpolation(k_grid, EV, extrapolation_bc = Line())

        for (i_k, k) = enumerate(k_grid)

            budget = z_grid[i_z]*k^α + (1-δ)*k

            lo = minimum(k_grid)
            high = min(budget - 1e-8, maximum(k_grid))
            obj(kp) = -log(budget-kp) - β*EV_interp(kp)  #maximize log(c) + β*EV_interp(kp)

            # The max is inside the expectation: k' is chosen before z' realizes.
            opt = optimize(obj, lo, high)

            V_next[i_k, i_z] = -opt.minimum
            kp_next[i_k, i_z] = opt.minimizer

        end
    end

    return V_next, kp_next

end


### Solve model
function solve_vfi!(para, sols)
    (; tol, max_iter) = para

    max_diff = tol + 10.0
    n = 0

    while max_diff > tol && n < max_iter
        n += 1
        V_next, kp_next = bellman(para, sols)

        max_diff = maximum(abs.(V_next .- sols.V))
        sols.V .= V_next
        sols.kp .= kp_next
    end

    if n == max_iter
        @warn "VFI did not converge in $max_iter iterations"
    end

    return n
end

# para, sols = initialize(; β=0.98)
para, sols = initialize()

# Cheap insurance against the most common bug in this model:
@assert all(sum(para.Π, dims = 2) .≈ 1.0)

@time n_vfi = solve_vfi!(para, sols)

plot(para.k_grid, sols.V, label = ["good state" "bad state"], lw = 2)

plot(para.k_grid, sols.kp, label = ["good state" "bad state"], lw = 2)
plot!(para.k_grid, para.k_grid, label = "45-degree line", ls = :dot, c = :black)

# Where each policy crosses the 45-degree line is the steady state capital
# stock conditional on staying in that productivity state forever. Above it the
# household runs capital down; below it, builds capital up.


################################################################################
# The Endogenous Grid Method
################################################################################

# VFI searched for k' numerically at every grid point, every iteration. We don't
# have to. Three changes on paper, for the deterministic model (HW3 adds the shock):
#
#   1. State is cash on hand, y = k^α + (1-δ)k. Choose c, save kp = y - c, and
#      tomorrow y' = h(kp) = kp^α + (1-δ)*kp.
#
#   2. FOC + envelope give the Euler equation. With log utility:
#           1/c(y) = β * h'(kp) / c(y'),   h'(kp) = α*kp^(α-1) + (1-δ)
#      No max, and no V -- just the policy c(.).
#
#   3. Put the grid on kp instead. Then the Euler RHS is a number, u' inverts in
#      closed form, and the budget constraint y = c + kp hands back today's grid
#      point. That state grid is endogenous -- it falls out of the algorithm.

@kwdef struct EGMParameters

    β::Float64 = 0.99
    δ::Float64 = 0.025
    α::Float64 = 0.36

    # Same numbers as ModelParameters, but read as the grid over TOMORROW's
    # capital. That is the whole trick.
    k_grid::Vector{Float64} = collect(range(0.1, length = 100, stop = 45.0))
    N_k::Int64 = length(k_grid)

    # h(kp) and h'(kp) are pinned to the kp grid, so build them once.
    # y_grid does double duty: the exogenous grid over today's state (where we
    # store the policy), and tomorrow's cash on hand if you save k_grid[j].
    y_grid::Vector{Float64} = k_grid.^α .+ (1-δ).*k_grid
    hp::Vector{Float64} = α.*k_grid.^(α-1) .+ (1-δ)

    tol::Float64 = 10^-5
    max_iter::Int64 = 10^4

end

# The EGM analogue of ModelSolutions, stored on the grid we chose up front. The
# endogenous grid lives inside one iteration and is gone by the end of it.
struct EGMSolutions

    c_pol::Vector{Float64}   #c(y) on y_grid
    kp_pol::Vector{Float64}  #k'(y) = y - c(y), same grid

end

function initialize_egm(; kwargs...)
    para = EGMParameters(; kwargs...)

    #Guess: consume everything, c(y) = y.
    c_pol = copy(para.y_grid)
    kp_pol = zero(para.y_grid)
    sols = EGMSolutions(c_pol, kp_pol)

    return para, sols
end

# Note what is missing: u itself. All EGM needs is marginal utility and -- the
# requirement for the method -- its closed-form inverse.
u_prime(c) = 1/c
u_prime_inv(x) = 1/x


### Euler operator (the EGM counterpart of bellman)
function euler(para, sols)
    (; β, k_grid, y_grid, hp) = para
    (; c_pol) = sols

    c_endog = zero(c_pol)
    y_endog = zero(c_pol)

    # Same shape as bellman's loop over the capital grid -- but look at the body.
    for (i_kp, kp) = enumerate(k_grid)

        # With kp fixed the Euler RHS is a number: invert u' instead of optimizing.
        # y_grid[i_kp] is tomorrow's cash on hand, so c_pol[i_kp] is c(y') exactly.
        rhs = β * hp[i_kp] * u_prime(c_pol[i_kp])
        c_endog[i_kp] = u_prime_inv(rhs)

        # The budget constraint hands back today's state -- the endogenous grid.
        y_endog[i_kp] = c_endog[i_kp] + kp

    end

    # Here the endogenous points bracket y_grid on every pass, so Line() never
    # fires. Check it: silently extrapolating a policy converges to a wrong answer.
    # @assert y_endog[1] <= y_grid[1] && y_endog[end] >= y_grid[end]

    # Put the policy back on the grid we chose -- the only interpolation, and last.
    c_interp = linear_interpolation(y_endog, c_endog, extrapolation_bc = Line())

    c_next = c_interp(y_grid)  #c(y) on y_grid

    return c_next

end


### Solve model
function solve_egm!(para, sols)
    (; tol, max_iter, y_grid) = para

    max_diff = tol + 10.0
    n = 0

    while max_diff > tol && n < max_iter
        n += 1
        c_next = euler(para, sols)

        #We iterate on the POLICY -- there is no V here. Both sides sit on y_grid.
        max_diff = maximum(abs.(c_next .- sols.c_pol))
        sols.c_pol .= c_next
    end

    if n == max_iter
        @warn "EGM did not converge in $max_iter iterations"
    end

    #The savings policy is one subtraction away.
    sols.kp_pol .= y_grid .- sols.c_pol

    return n
end

# Race them on the same model. No need to touch the VFI solver: one productivity
# state with a 1x1 transition matrix IS the deterministic model.
para_d, sols_d = initialize(z_grid = [1.0], Π = ones(1,1));
para_egm, sols_egm = initialize_egm();

@time n_d = solve_vfi!(para_d, sols_d)
@time n_egm = solve_egm!(para_egm, sols_egm)


# Fine print: the stopping rules differ. VFI stops when the VALUE stops moving,
# EGM when the POLICY does, and policies converge faster. But tighten EGM's tol
# to 1e-6 and it is still far ahead -- the per-iteration cost is the real win.

# Nothing to line up: y_grid[i] IS cash on hand at k_grid[i], so kp_pol[i] is
# already the savings policy at the same k VFI used.
kp_egm = sols_egm.kp_pol

@show maximum(abs.(kp_egm .- sols_d.kp[:, 1]))          #whole grid
@show maximum(abs.(kp_egm .- sols_d.kp[:, 1])[5:end])   #dropping the lowest 4 points

plot(para_d.k_grid, sols_d.kp[:, 1], label = "VFI (interpolation + Brent)", lw = 3)
plot!(para_d.k_grid, kp_egm, label = "EGM", ls = :dash, lw = 3)
plot!(para_d.k_grid, para_d.k_grid, label = "45-degree line", ls = :dot, c = :black)

# The policies lie on top of each other except at the BOTTOM, and that gap is
# VFI's grid showing through: near k = 0.1 the value function is sharply curved,
# and VFI interpolates it with straight lines 0.45 apart -- wider than the answer
# itself. EGM never interpolates V at all. Which do you trust down there?
#
# Settle it by re-solving VFI on 2000 points and treating that as the truth: on
# the same 100 points, EGM is off by 0.011 and VFI by 0.050. EGM's coarse-grid
# answer is the more accurate one. (Try k_grid = exp.(range(...)) to put VFI's
# points where the curvature is -- a fix worth knowing, but one EGM never needs.)
#
# Same answer, a fraction of the time. Both operators loop over the same 100 grid
# points; the difference is what happens inside. bellman calls an optimizer at
# every point, euler does arithmetic. Lectures 2/4/5 made the same algorithm
# faster -- better search, fewer grid points, more cores. Today we changed the
# algorithm, and the paper work did all of it. Reach for a better algorithm
# before you micro-optimize.
