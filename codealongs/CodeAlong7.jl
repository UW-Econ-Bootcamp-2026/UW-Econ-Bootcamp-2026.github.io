################################################################################
# Load Packages
################################################################################

using Random, Statistics, LinearAlgebra, Interpolations, Optim, Plots


################################################################################
# The household block
################################################################################

# The economy (Huggett 1993): a continuum of households, income y alternating
# between "employed" (y = 1.0) and "unemployed" (y = 0.5) on a Markov chain,
# saving in a one-period bond at rate r, with an ad hoc borrowing limit a_min:
#
#     V(a, y) = max_{a' >= a_min} { u(c) + β Σ_{y'} Π(y'|y) V(a', y') },
#     c = (1 + r)a + y - a'
#
# The bond is in ZERO net supply -- one household's borrowing is another's
# saving -- so equilibrium is the r at which aggregate assets are exactly zero.


@kwdef struct HuggettParameters

    γ::Float64 = 2.0                              # CRRA
    β::Float64 = 0.96

    # Π[i, j] = P(y' = y_grid[j] | y = y_grid[i]), so rows sum to one.
    y_grid::Vector{Float64} = [1.0, 0.5]          # employed, unemployed
    Π::Matrix{Float64} = [0.9 0.1; 0.5 0.5]
    N_y::Int64 = length(y_grid)

    # The grid the HOUSEHOLD PROBLEM is solved on. As in Lecture 6 we interpolate
    # the continuation value and let Optim pick a' continuously, so this grid only
    # has to be fine enough to interpolate V well.
    a_min::Float64 = -1.0                         # borrowing limit
    a_max::Float64 = 4.0
    N_a::Int64 = 101
    a_grid::Vector{Float64} = collect(range(a_min, a_max; length = N_a))

    # Simulation settings (Method 1).
    N_sim::Int64 = 10_000
    T_sim::Int64 = 500
    seed::Int64 = 1234

    tol::Float64 = 10^-6
    max_iter::Int64 = 10^4

    # The distribution is a LINEAR fixed point, so iterating it is cheap and we
    # can afford to ask for many more digits than we do from VFI.
    tol_dist::Float64 = 10^-9
    max_iter_dist::Int64 = 10^5

end

# Everything the model produces, in one container. V and a_pol are the household
# block; μ, T_star, a_sim and y_sim are the distribution, computed two ways. Mutable
# so that r -- a scalar, unlike the array fields -- can be reassigned in place.
mutable struct HuggettSolutions

    V::Matrix{Float64}                            # N_a × N_y
    a_pol::Matrix{Float64}                        # N_a × N_y: a'(a, y) on a_grid

    # Method 2, the histogram: T_star is the transition matrix over the flattened
    # (asset, income) state, μ its stationary distribution back in (a, y) shape.
    μ::Matrix{Float64}                            # N_a × N_y
    T_star::Matrix{Float64}                       # (N_a*N_y) × (N_a*N_y)

    # Method 1, Monte Carlo: the terminal cross-section of a simulated panel.
    a_sim::Vector{Float64}                        # N_sim
    y_sim::Vector{Int64}                          # N_sim

    r::Float64 # price

end

function initialize(; kwargs...)
    para = HuggettParameters(; kwargs...)

    V = zeros(para.N_a, para.N_y)
    a_pol = zeros(para.N_a, para.N_y)

    # Any distribution will do as a starting guess; we'll guess uniform.
    μ = ones(para.N_a, para.N_y) ./ (para.N_a * para.N_y)
    T_star = zeros(para.N_a * para.N_y, para.N_a * para.N_y)

    a_sim = zeros(para.N_sim)
    y_sim = ones(Int64, para.N_sim)

    r = 0.03 # initial guess

    sols = HuggettSolutions(V, a_pol, μ, T_star, a_sim, y_sim, r)

    return para, sols
end

u(c, γ) = c^(1 - γ) / (1 - γ)


### Bellman operator
function bellman(para, sols)
    (; γ, β, y_grid, Π, a_grid, a_min, a_max, N_y) = para
    (; V, r) = sols

    V_next = zero(V)
    a_next = zero(V)

    for i_y in 1:N_y

        # EV[i] = Σ_{y'} Π[i_y, y'] * V(a_grid[i], y'). Note the ROW of Π: we
        # condition on today's y and sum over tomorrow's. Interpolate the
        # expectation, not each V(., y') separately -- same answer, cheaper.
        EV = V * Π[i_y, :]
        EV_interp = linear_interpolation(a_grid, EV, extrapolation_bc = Line())

        for (i_a, a) in enumerate(a_grid)

            budget = (1 + r)*a + y_grid[i_y]

            lo = a_min                              # the borrowing constraint
            high = min(budget - 1e-8, a_max)        # keep c > 0, stay on the grid

            obj(ap) = -( u(budget - ap, γ) + β*EV_interp(ap) )

            # The max is inside the expectation: a' is chosen before y' realizes.
            opt = optimize(obj, lo, high)

            V_next[i_a, i_y] = -opt.minimum
            a_next[i_a, i_y] = opt.minimizer

        end
    end

    return V_next, a_next

end


### Solve the household problem
function solve_vfi!(para, sols)
    (; tol, max_iter) = para

    max_diff = tol + 10.0
    n = 0

    while max_diff > tol && n < max_iter
        n += 1
        V_next, a_next = bellman(para, sols)

        max_diff = maximum(abs.(V_next .- sols.V))
        sols.V .= V_next
        sols.a_pol .= a_next
    end

    if n == max_iter
        @warn "VFI did not converge in $max_iter iterations"
    end

    return n
end

# a'(a, y) at ANY a, not just at the grid points we solved on. BOTH distribution
# methods need this. We rebuild it from a_pol.
a_pol_interp(para, sols) =
    [linear_interpolation(para.a_grid, sols.a_pol[:, i_y], extrapolation_bc = Line())
     for i_y in 1:para.N_y]


para, sols = initialize()
@time n_vfi = solve_vfi!(para, sols)
println("VFI: $n_vfi sweeps")

plot(para.a_grid, sols.a_pol; label = ["employed" "unemployed"], lw = 2,
     xlabel = "a", ylabel = "a'", title = "Savings policy at r = $(sols.r)")
plot!(para.a_grid, para.a_grid; label = "45-degree line", ls = :dot, c = :black)

# Two things to notice, because everything below lives off them:
#   1. The unemployed policy is pinned at a_min over a stretch at the bottom --
#      the borrowing constraint binds, and mass will pile up there.
#   2. Each policy crosses the 45-degree line once. That crossing is the target
#      wealth level a household drifts toward while it stays in that income
#      state. The income shock is what keeps knocking it off, and the fight
#      between the two is the distribution we are about to compute.

@assert maximum(sols.a_pol) < para.a_max         # grid is not binding at the top

# From here on the household block is a black box. It hands us a policy we can
# evaluate anywhere. That is all the distribution needs.
g = a_pol_interp(para, sols)


################################################################################
# Method 1: put 10,000 households on the policy and let them run
################################################################################

# Two panels, two functions. The INCOME panel is a Markov
# chain: it does not depend on r, on V, or on assets, so we draw it ONCE and reuse
# it everywhere below. The ASSET panel is redone whenever the policy changes.


# One Markov step, isolated from the RNG: given today's state j, the transition
# matrix Π, return tomorrow's state.
function markov_step(j::Int, Π::Matrix{Float64})
    x = rand()
    cum_prob = 0.0
    j_next = size(Π, 2)
    
    for k in 1:size(Π, 2)
        cum_prob += Π[j, k]
        if x <= cum_prob
            j_next = k
            break
        end
    end
    
    return j_next
end


# ---------------------------------------------------------------- live AI demo
# The income panel is small, fully specified and tedious -- the shape of task these
# tools are good at. Note what the prompt pins down: the signature, the fields it
# may use, the return type and shape, the RNG, the packages it may NOT reach for,
# and a length budget.
#
#   Write a Julia function `simulate_y_panel(para)` which iterates a markov chain to make a 
#   pseudo-panel of draws.
#  `para` has fields `Π` (an N_y x N_y Markov transition matrix), `N_y`, `N_sim`, `T_sim` and `seed`. 
#   Return an `N_sim x T_sim` matrix of income-state INDICES (1 or 2), not income
#   levels: every household starts in state 1, and column t is drawn from the row
#   of Π given by column t-1, using one uniform draw per household per period.
#   Call `Random.seed!(para.seed)` once at the top so the panel is reproducible.
#   Don't use any packages other than Random to set the seed.

function simulate_y_panel(para)
    (; Π, N_sim, T_sim, seed) = para
    Random.seed!(seed)
    y_panel = ones(Int64, N_sim, T_sim)          # everyone starts employed

    for t in 2:T_sim, i in 1:N_sim               # i innermost: column-major
        y_panel[i, t] = markov_step(y_panel[i, t-1], Π)
    end

    return y_panel
end

@time y_panel = simulate_y_panel(para)

# It runs. That is not the same as it being right, and "the output is 1s and 2s" is
# not a test. Test the PROPERTY you asked for: the pseudo-panel should match the
# long-run distribution implied by Π, up to sampling noise.
function invariant_distribution(Π, T)
    return vec((Π^T)[1,:])
end

y_probs = invariant_distribution(para.Π, 1000)
y_probs_hat = sum(y_panel[:, end]) / para.N_sim - 1

# -----------------------------------------------------------------------------

function simulate_a_panel(para, sols, y_panel; a0 = 0.0)     # assets, given income
    (; N_sim, T_sim) = para
    g = a_pol_interp(para, sols)
    a_panel = fill(a0, N_sim, T_sim)

    for t in 2:T_sim, i in 1:N_sim
        a_panel[i, t] = g[y_panel[i, t-1]](a_panel[i, t-1])
    end

    return a_panel
end

function simulate!(para, sols, y_panel; kwargs...)     # keep the last cross-section
    a_panel = simulate_a_panel(para, sols, y_panel; kwargs...)
    sols.a_sim .= a_panel[:, end]
    sols.y_sim .= y_panel[:, end]
    return a_panel
end

@time a_panel = simulate!(para, sols, y_panel)

# Burn-in: everyone started at the same (a, y), so the first stretch is the panel
# forgetting where we put it. Run it again from the borrowing limit, on the other
# side of the mean; where the two paths meet is the burn-in you need.
plot(vec(mean(a_panel, dims = 1)); lw = 2, label = "started at a = 0",
     xlabel = "period", ylabel = "mean assets", title = "Burn-in")
plot!(vec(mean(simulate_a_panel(para, sols, y_panel; a0 = para.a_min), dims = 1));
      lw = 2, ls = :dash, label = "started at a = a_min")

histogram(sols.a_sim; bins = 60, normalize = :probability, legend = false, xlabel = "a",
          ylabel = "share", title = "Simulated stationary distribution of assets")

# "At the borrowing limit" needs a LOOSE tolerance: a constrained household's a' is
# whatever Brent returned, a whisker short of the endpoint it searched.
at_limit(a, para) = mean(a .<= para.a_min + 1e-5)

println("MC:  mean assets = ", round(mean(sols.a_sim), digits = 4),
        "   median = ", round(median(sols.a_sim), digits = 4),
        "   frac at borrowing limit = ", round(at_limit(sols.a_sim, para), digits = 4))

# (Ergodicity: `initialize(; N_sim = 1, T_sim = 200_000)` and these same two
# functions give the same distribution from a single household's long path.)


################################################################################
# Method 2: iterate the distribution directly (histogram)
################################################################################

# No randomness at all: with T_star the transition matrix the policy and Π induce, the
# stationary μ solves μ' = T_star'μ -- iterated exactly like the Bellman operator. The
# fiddly step: our solver picks a' CONTINUOUSLY, so a'(a, y) lands between two grid
# points. Split the household's mass between them (Young, 2010), with the weight
# chosen so the mass sits at a' ON AVERAGE.
function lottery(ap, grid)
    ap = clamp(ap, first(grid), last(grid))
    i = min(searchsortedlast(grid, ap), length(grid) - 1)
    w = (grid[i+1] - ap) / (grid[i+1] - grid[i])     # distance to the UPPER point
    return i, i + 1, w
end

# Test what the lottery is FOR: the mass has to land at ap.
@assert all(begin
                i, k, w = lottery(ap, para.a_grid)
                0 <= w <= 1 && w*para.a_grid[i] + (1 - w)*para.a_grid[k] ≈ ap
            end for ap in range(para.a_min, para.a_max; length = 1000))

# Build T_star. A state is a pair (a_i, y_j), flattened to one index. Assets move to
# a' = g_j(a_i), split across two nodes; income moves to y' with probability Π[j,j'].
# Independent given today's state, so the entry is the product.
flat(i, j, N_a) = (j - 1) * N_a + i

function transition_matrix(para, sols)
    (; Π, a_grid, N_a, N_y) = para
    g = a_pol_interp(para, sols)
    T_star = zeros(N_a * N_y, N_a * N_y)

    for j in 1:N_y, i in 1:N_a
        lo, hi, w = lottery(g[j](a_grid[i]), a_grid)
        for jp in 1:N_y
            T_star[flat(i, j, N_a), flat(lo, jp, N_a)] += w * Π[j, jp]
            T_star[flat(i, j, N_a), flat(hi, jp, N_a)] += (1 - w) * Π[j, jp]
        end
    end

    return T_star
end

function stationary(T_star, μ0; tol = 10^-12, max_iter = 10^5)     # as VFI, but linear
    μ = copy(μ0)
    max_diff = tol + 10.0
    n = 0
    while max_diff > tol && n < max_iter
        n += 1
        μ_next = T_star' * μ
        max_diff = maximum(abs.(μ_next .- μ))
        μ .= μ_next
    end
    
    if n == max_iter
        @warn "Stationary distribution did not converge in $max_iter iterations"
    end
    
    return μ, n
end

function solve_distribution!(para, sols)     # the distribution's solve_vfi!
    (; N_a, N_y, tol_dist, max_iter_dist) = para
    sols.T_star .= transition_matrix(para, sols)

    # Every row of T_star is a distribution over tomorrow's states. If this fails, mass
    # is leaking and nothing downstream means anything.
    @assert all(sum(sols.T_star, dims = 2) .≈ 1.0)

    # Warm start: seed the iteration with sols.μ (already the stationary distribution
    # from the previous r, if this isn't the first call) instead of re-guessing uniform.
    μ_flat, n = stationary(sols.T_star, vec(sols.μ); tol = tol_dist, max_iter = max_iter_dist)
    sols.μ .= reshape(μ_flat, N_a, N_y)              # back to (asset, income)
    return n
end

@time n_dist = solve_distribution!(para, sols)
@assert sum(sols.μ) ≈ 1.0

mass = vec(sum(sols.μ, dims = 2))                    # marginal distribution of assets
println("hist: $n_dist iterations, mean assets = ", round(sum(mass .* para.a_grid), digits = 4),
        "   at borrowing limit = ", round(mass[1], digits = 4))


################################################################################
# Same object, two ways: compare them
################################################################################

# Compare CDFs, not histograms: a CDF does not depend on anyone's choice of bins, so
# any daylight between the curves is method, not plotting.
cdf_hist = cumsum(mass)
cdf_mc = [mean(sols.a_sim .<= a) for a in para.a_grid]

plot(para.a_grid, cdf_hist; lw = 3, label = "histogram method",
     xlabel = "a", ylabel = "P(assets <= a)", title = "Two roads to the same distribution")
plot!(para.a_grid, cdf_mc; lw = 3, ls = :dash, label = "Monte Carlo, N = 10,000")

println("max |CDF gap| = ", round(maximum(abs.(cdf_hist .- cdf_mc)), digits = 5),
        "   at a = ", round(para.a_grid[argmax(abs.(cdf_hist .- cdf_mc))], digits = 3))

# Means agree to two decimals; the CDFs sit about two points apart at worst. Both
# errors live in that gap, and you tell them apart by moving one at a time: take N_a
# from 101 to 801 and the histogram's CDF at a = -0.2 falls 0.283 -> 0.272
# (discretization), while changing the seed alone moves the simulation's 0.262 ->
# 0.275 (sampling). Neither curve is the truth; they are wrong differently.
#
# Monte Carlo's error is O(1/sqrt(N)) and its cost is in the sample; the histogram
# has no sampling noise at all, so it is SMOOTH in r, but its cost is in the grid and
# grows fast with the state space. In an equilibrium loop, smoothness usually wins.


################################################################################
# The price: a Huggett equilibrium
################################################################################

# Everything above assumed r = 0.03. Nothing pinned it down. The bond is in zero net
# supply, so equilibrium is the r at which aggregate assets are exactly zero:
#
#     A(r) = Σ_{a,y} μ_r(a, y) · a = 0
#
# A root-finding problem you know how to solve; what is new is the price of one
# evaluation: a full household solve AND a full distribution.

function aggregate_assets(r; kwargs...)
    para, sols = initialize(; kwargs...)
    sols.r = r

    solve_vfi!(para, sols)
    solve_distribution!(para, sols)

    A = sum(vec(sum(sols.μ, dims = 2)) .* para.a_grid)
    return A, para, sols
end

excess(r) = aggregate_assets(r)[1]

# A(r) is increasing, and r* must sit below 1/β - 1: at that rate households never
# stop accumulating and no stationary distribution exists.
r_lo, r_hi = -0.02, 0.035
println("1/β - 1 = ", round(1/para.β - 1, digits = 4),
        "   bracket: A($r_lo) = ", round(excess(r_lo), digits = 4),
        ",  A($r_hi) = ", round(excess(r_hi), digits = 4))

# Bracketed and monotone, so hand it to a solver instead of writing a loop. With
# Roots.jl this is `find_zero(excess, (r_lo, r_hi))`; to stay inside the
# packages already loaded here, we minimize the squared residual with Optim.
@time opt = optimize(r -> excess(r)^2, r_lo, r_hi; abs_tol = 1e-5)

r_star = opt.minimizer
A_star, para_star, sols_star = aggregate_assets(r_star)
mass_star = vec(sum(sols_star.μ, dims = 2))

println("equilibrium r = ", round(r_star, digits = 5), "   A(r*) = ", round(A_star, digits = 6),
        "   at borrowing limit = ", round(mass_star[1], digits = 4))

plot(para.a_grid, mass; lw = 3, label = "at r = $(sols.r) (assumed)",
     xlabel = "a", ylabel = "share of households",
     title = "The distribution depends on the price")
plot!(para_star.a_grid, mass_star; lw = 3, ls = :dash,
      label = "at r* = $(round(r_star, digits = 4)) (equilibrium)")

# How many of those digits are real? The solver stopped at 1e-5, but the number it
# converged TO is only as good as the grid underneath it, and the only way to know is
# to change it and re-solve: N_a = 51, 101, 201 give r* = 0.02715, 0.02735, 0.02742
# -- under half a basis point across a fourfold refinement.
#
# What moved, and why: at the assumed r = 0.03 this economy held positive net assets
# (A = 0.14 > 0), so r has to FALL to clear the bond market, and with saving paid
# less, more households end up pinned at the borrowing limit. The distribution and
# the price are one object.
#
# It scales unchanged in structure -- Aiyagari puts r in a firm's FOC, a tax changes
# the budget constraint, transition dynamics push μ forward date by date -- out of
# tools you already had: root finding (Lecture 3), interpolation (Lecture 4), VFI
# (Lecture 6). The new idea today is that the model has a distribution in it.
