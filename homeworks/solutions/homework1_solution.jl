# homework1_solution.jl
# Computational Bootcamp, Summer 2026 -- Homework 1 solutions
#
# Packages used: Plots, Random, Statistics, LinearAlgebra, BenchmarkTools
#   pkg> add Plots Random Statistics BenchmarkTools
#
# Run each question's driver at the bottom, or `include("homework1.jl")`.

using Random, Statistics, LinearAlgebra, Plots
using BenchmarkTools

FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

# =====================================================================
# Question 1: Present values
# =====================================================================
# PV = sum_{t=1}^T y_t / (1+r)^{t-1}
function present_value(payments, r)
    pv = 0.0
    for (t, y) in enumerate(payments)   # t runs 1..T
        pv += y / (1 + r)^(t - 1)
    end
    return pv
end

function q1()
    r = 0.05
    T = 20
    payments = ones(T)                              # constant stream of 1
    pv_loop = present_value(payments, r)
    # Closed form for y_t = 1: sum_{t=1}^T (1/(1+r))^{t-1} is a geometric sum
    # with ratio q = 1/(1+r): (1 - q^T) / (1 - q).
    q = 1 / (1 + r)
    pv_closed = (1 - q^T) / (1 - q)
    println("Q1: PV(loop) = ", pv_loop, ",  PV(closed) = ", pv_closed)
    @assert isapprox(pv_loop, pv_closed; atol = 1e-10)
    return pv_loop
end

# =====================================================================
# Question 2: Age-earnings profiles
# =====================================================================
# p(x) = sum_{i=0}^n a_i x^i.  coeff[i] holds a_{i-1} (1-indexed).
function p(x, coeff)
    total = 0.0
    for (i, a) in enumerate(coeff)      # i = 1..n+1  -> power i-1
        total += a * x^(i - 1)
    end
    return total
end

function q2()
    coeff = [2.6, 0.08, -0.0015]
    xs = range(0, 40; length = 401)
    earnings = [exp(p(x, coeff)) for x in xs]
    plt = plot(xs, earnings; xlabel = "experience x", ylabel = "exp(p(x))",
               title = "Age-earnings profile", legend = false)
    savefig(plt, joinpath(FIGDIR, "hw1_q2_earnings.png"))
    return plt
end

# =====================================================================
# Question 3: Monte Carlo pi
# =====================================================================
# Fraction of uniform draws in unit square landing inside the quarter
# unit circle -> pi/4.
function mc_pi(n; seed = 1)
    Random.seed!(seed)
    inside = 0
    for _ in 1:n
        x = rand(); y = rand()
        if x^2 + y^2 <= 1.0
            inside += 1
        end
    end
    return 4 * inside / n
end

function q3()
    for n in (10^3, 10^5, 10^7)
        println("Q3: n = ", n, "  pi ~ ", mc_pi(n))
    end
end

# =====================================================================
# Question 4: Simulating and estimating a Mincer regression
# =====================================================================
# log(w) = b0 + b1*educ + b2*exp + b3*exp^2 + sigma*eps
# Design matrix columns ordered [1, educ, exp, exp^2] so beta = [b0,b1,b2,b3].

# Manual OLS estimator: β̂ = (X'X)^{-1} X'y. Uses \ (a solve, not an explicit
# inverse) for numerical stability -- same answer, better conditioned.
ols(X, y) = (X'X) \ (X'y)

function q4(; N = 50, S = 200, seed = 42)
    Random.seed!(seed)
    β0, β1, β2, β3, σ = 1.0, 0.08, 0.04, -0.0008, 0.35

    # Draw regressors ONCE and hold fixed.
    educ = 14 .+ 2 .* randn(N)          # N(14, 4) -> sd 2
    expr = 15 .+ 5 .* randn(N)          # N(15, 25) -> sd 5
    X = hcat(ones(N), educ, expr, expr .^ 2)   # N x 4

    βtrue = [β0, β1, β2, β3]

    estimates = zeros(S, 4)
    for s in 1:S
        ε = randn(N)
        y = X * βtrue .+ σ .* ε
        estimates[s, :] = ols(X, y)     # β̂ = (X'X)^{-1} X'y, computed manually
    end

    labels = ["beta0", "beta1", "beta2", "beta3"]
    plots = [histogram(estimates[:, k]; title = labels[k], legend = false,
                       xlabel = "estimate") for k in 1:4]
    for k in 1:4
        vline!(plots[k], [βtrue[k]]; color = :red, lw = 2)
    end
    plt = plot(plots...; layout = (2, 2), size = (800, 600))
    savefig(plt, joinpath(FIGDIR, "hw1_q4_mincer.png"))

    println("Q4: mean estimates = ", vec(mean(estimates; dims = 1)))
    println("    truth          = ", βtrue)
    # OLS is unbiased under exogeneity, so histograms center on the truth.
    return estimates
end

# =====================================================================
# Question 5: Firm survival
# =====================================================================
# x_{t+1} = alpha*x_t + sigma*eps, x_0 = 1. Exit at first t with x_t <= 0.
function simulate_exit_time(α, σ; tmax = 200)
    x = 1.0
    for t in 1:tmax
        x = α * x + σ * randn()
        if x <= 0
            return t
        end
    end
    return tmax
end

function q5(; seed = 7)
    Random.seed!(seed)
    # Part 1: sigma = 0.2, alpha = 1.0, 100 paths.
    exits = [simulate_exit_time(1.0, 0.2) for _ in 1:100]
    plt1 = histogram(exits; xlabel = "exit time T0", legend = false,
                     title = "Exit times (alpha=1.0, sigma=0.2)")
    savefig(plt1, joinpath(FIGDIR, "hw1_q5_exits.png"))

    # Part 2: mean exit time for several alpha, more paths for a stable mean.
    # Track the censored share too -- T0 is capped at tmax by construction.
    αs = [0.8, 1.0, 1.2]
    tmax = 200
    sims = [[simulate_exit_time(α, 0.2; tmax = tmax) for _ in 1:1000] for α in αs]
    means = [mean(s) for s in sims]
    censored = [mean(s .== tmax) for s in sims]
    plt2 = plot(αs, means; marker = :circle, xlabel = "alpha",
                ylabel = "mean T0", legend = false, title = "Mean exit time")
    savefig(plt2, joinpath(FIGDIR, "hw1_q5_meanexit.png"))

    println("Q5: mean exit times   = ", means, " for alpha = ", αs)
    println("    frac hitting tmax = ", censored)
    # alpha governs persistence, not drift: there is no constant term in
    # x_{t+1} = alpha*x_t + sigma*eps. With alpha < 1 the process is mean
    # reverting toward 0, so x is dragged to the exit barrier quickly and
    # firms die fast (mean T0 ~ 11 at alpha = 0.8). At alpha = 1 it is a
    # driftless random walk, which hits 0 with probability 1 but has an
    # INFINITE expected hitting time. At alpha > 1 the process is explosive
    # and typically escapes the barrier for good.
    #
    # So read the mean-T0 plot with care: T0 is censored at tmax. Here ~34%
    # of paths are censored at alpha = 1.0 and ~99.9% at alpha = 1.2, where
    # the "mean" of ~200 is just tmax and carries no information about
    # survival. The rising curve reflects the cap as much as the economics --
    # raising tmax would keep pushing the last two points up.
    return means, censored
end

# =====================================================================
# Question 6: Newton's method
# =====================================================================
function newton(f, f_prime, x0; tol = 1e-8, maxiter = 1000)
    x = float(x0)
    for iter in 1:maxiter
        fx = f(x)
        dfx = f_prime(x)
        step = fx / dfx
        x_new = x - step
        if abs(x_new - x) < tol
            return x_new, iter
        end
        x = x_new
    end
    @warn "Newton did not converge in $maxiter iterations"
    return x, maxiter
end

function q6()
    # Test 1: f(x) = (x-1)^3, root at 1 (convergence is slow -- triple root).
    f1(x) = (x - 1)^3
    f1p(x) = 3 * (x - 1)^2
    root1, it1 = newton(f1, f1p, 2.0; tol = 1e-8, maxiter = 10_000)
    println("Q6a: root of (x-1)^3 ~ ", root1, " in ", it1, " iters")

    # Test 2: bond pricing. f(r) = sum_{t=1}^10 5/(1+r)^t + 100/(1+r)^10 - 95.
    coupon, face, price, T = 5.0, 100.0, 95.0, 10
    f2(r) = sum(coupon / (1 + r)^t for t in 1:T) + face / (1 + r)^T - price
    # f'(r) = sum -t*coupon/(1+r)^{t+1} - T*face/(1+r)^{T+1}
    f2p(r) = sum(-t * coupon / (1 + r)^(t + 1) for t in 1:T) -
             T * face / (1 + r)^(T + 1)
    ytm, it2 = newton(f2, f2p, 0.05)
    println("Q6b: bond YTM ~ ", ytm, " (", 100 * ytm, "%) in ", it2, " iters")
    @assert abs(f2(ytm)) < 1e-6
    return ytm
end

# =====================================================================
# Question 7: Make it fast -- rewrite of slow_growth.jl
# =====================================================================
@kwdef struct GrowthPrimitives
    θ::Float64 = 0.36
    δ::Float64 = 0.025
    β::Float64 = 0.99
    nk::Int = 1000
    # @kwdef lets later defaults refer to earlier fields, so the grid always
    # matches nk -- don't hardcode the length here.
    k_grid::Vector{Float64} = collect(range(0.01, 45.0; length = nk))
    tol::Float64 = 1e-4
    maxiter::Int = 10_000
end

# One Bellman sweep: fills V_next and policy in place from current V.
function bellman!(V_next, policy, V, prim::GrowthPrimitives)
    (; θ, δ, β, nk, k_grid) = prim
    @inbounds for i in 1:nk
        budget = k_grid[i]^θ + (1 - δ) * k_grid[i]
        max_util = -1e10
        best = k_grid[1]
        for j in 1:nk
            c = budget - k_grid[j]
            c <= 0 && break            # k_grid increasing -> all later j worse
            util = log(c) + β * V[j]
            if util > max_util
                max_util = util
                best = k_grid[j]
            end
        end
        V_next[i] = max_util
        policy[i] = best
    end
    return V_next
end

function solve_growth(prim::GrowthPrimitives = GrowthPrimitives())
    V = zeros(prim.nk)
    V_next = similar(V)
    policy = similar(V)
    diff = Inf
    n = 0
    while diff > prim.tol && n < prim.maxiter
        n += 1
        bellman!(V_next, policy, V, prim)
        diff = 0.0
        @inbounds for i in 1:prim.nk
            d = abs(V_next[i] - V[i])
            d > diff && (diff = d)
        end
        V, V_next = V_next, V          # swap buffers, no allocation
    end
    return V, policy, n               # V now holds the converged values
end

function q7()
    V, policy, n = solve_growth()
    println("Q7: converged after ", n, " iterations")
    # Benchmark. The slow version relies on globals; the rewrite is a function.
    @btime solve_growth()
    # Biggest wins, in rough order: (1) wrapping everything in functions with
    # concrete types (kills type instability from globals); (2) buffer swap
    # instead of V = copy(V_next) each iteration; (3) the `c <= 0 && break`
    # early exit; (4) @inbounds. 
    return V, policy
end

# ---------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    q1(); q2(); q3(); q4(); q5(); q6(); q7()
end
