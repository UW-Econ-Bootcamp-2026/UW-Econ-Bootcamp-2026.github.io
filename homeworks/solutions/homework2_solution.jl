# homework2_solution.jl
# Computational Bootcamp, Summer 2026 -- Homework 2 solutions
#
# Packages:
#   pkg> add Optim Roots Interpolations FastGaussQuadrature Distributions Plots
#
# Run each question's driver at the bottom, or `include("homework2.jl")`.
#
# The pinned function names (also listed in hw2-template/README.md and checked by
# the hidden tests) are:
#
#   optimal_hours(w; α, y), hours_closed(w; α, y)       # Q1
#   estimate_logit(x, d; method)                        # Q2
#   himmelblau(v)                                       # Q3
#   newton_min(grad, hess, x0; tol, maxiter)            # Q3
#   excess_demand(p)                                    # Q4
#   bisection(f, a, b; tol, maxiter)                    # Q4
#   tatonnement(p0; λ, tol, maxiter)                    # Q4
#   interp_error(interior), total_error(interior), optimal_grid()  # Q5
#   eu_quadrature(n), eu_montecarlo(M; seed)            # Q6
#
# Everything else here (simulate_logit, neg_loglik, the analytic gradient and
# Hessian, newton, excess_demand_prime, eu_exact) is work the prompt asks for but
# does not pin a name for — students may call those whatever they like.

using Optim, Plots, Random, Statistics, LinearAlgebra
using Distributions, Roots, Interpolations, FastGaussQuadrature

FIGDIR = joinpath(@__DIR__, "figures")
mkpath(FIGDIR)

# =====================================================================
# Question 1: Labor supply
# =====================================================================
# max_h log(c) + α log(1-h),  c = w h + y,  h ∈ [0,1].
function optimal_hours(w; α = 1.5, y = 0.5)
    # Optim MINIMIZES -> minimize the negative of utility.
    negu(h) = -(log(w * h + y) + α * log(1 - h))
    # Shrink the interval: log(1-h) blows up at h = 1, and log(c) at h = -y/w.
    res = optimize(negu, 1e-8, 1 - 1e-8, Brent())
    return Optim.minimizer(res)
end

# Part B: the interior FOC.
#   du/dh = w/(w h + y) - α/(1-h) = 0
#   => w(1-h) = α(w h + y)
#   => w - w h = α w h + α y
#   => h* = (w - α y) / (w (1 + α)),  clamped to [0,1].
# The worker stops working (h* = 0) exactly when w ≤ α y: the reservation wage is
# the marginal rate of substitution between leisure and consumption evaluated at
# h = 0, which here is α y = 1.5 * 0.5 = 0.75. Below that, an hour of work is not
# worth the leisure it costs.
function hours_closed(w; α = 1.5, y = 0.5)
    h = (w - α * y) / (w * (1 + α))
    return clamp(h, 0.0, 1.0)
end

function q1()
    ws = range(0.5, 5; length = 200)
    h_num = optimal_hours.(ws)
    h_cf  = hours_closed.(ws)
    println("Q1B: max |numeric - closed form| = ", maximum(abs.(h_num .- h_cf)))
    println("     reservation wage α*y = ", 1.5 * 0.5)

    # Part C: higher non-labor income y = 2.0.
    h_y2 = [optimal_hours(w; y = 2.0) for w in ws]

    plt = plot(ws, h_num; label = "y = 0.5", xlabel = "wage w",
               ylabel = "optimal hours h*", title = "Labor supply", lw = 2)
    plot!(plt, ws, h_y2; label = "y = 2.0", lw = 2)
    savefig(plt, joinpath(FIGDIR, "hw2_q1_labor.png"))
    # Interpretation: leisure is a normal good, so higher non-labor income raises
    # the reservation wage (to α*y = 3.0) and lowers hours at every wage -- a pure
    # income effect, since y does not change the price of leisure.
    return h_num, h_cf
end

# =====================================================================
# Question 2: Maximum likelihood for a binary choice (logit)
# =====================================================================
Λ(z) = 1 / (1 + exp(-z))

function simulate_logit(N; β0 = -0.4, β1 = 1.2, seed = 1)
    Random.seed!(seed)
    x = randn(N)
    u = rand(N)
    d = Float64.(Λ.(β0 .+ β1 .* x) .> u)
    return x, d
end

# NEGATIVE log likelihood, so it can go straight into Optim.
# Written generically (no ::Vector{Float64} on β): not needed for the finite
# difference gradient below, but it is what lets a student who takes the optional
# `autodiff = :forward` route push ForwardDiff dual numbers through it.
function neg_loglik(β, x, d)
    z = β[1] .+ β[2] .* x
    p = Λ.(z)
    p = clamp.(p, 1e-12, 1 - 1e-12)     # guard log(0)
    return -sum(d .* log.(p) .+ (1 .- d) .* log.(1 .- p))
end

function estimate_logit(x, d; method = :lbfgs)
    obj(β) = neg_loglik(β, x, d)
    start = [0.0, 0.0]
    res = if method == :nm
        optimize(obj, start, NelderMead())
    elseif method == :lbfgs
        # No gradient supplied, so Optim builds one by finite differences.
        # Adding `autodiff = :forward` is the prompt's optional route: exact
        # derivatives, the same β to 1e-10 here, 7 iterations instead of 8.
        optimize(obj, start, LBFGS())
    else
        error("unknown method $method (use :nm or :lbfgs)")
    end
    return Optim.minimizer(res), Optim.iterations(res)
end

function q2()
    # Part A: one simulated dataset.
    x, d = simulate_logit(5000)

    # Part B: estimate by MLE with both optimizers and compare iteration counts.
    β_nm, it_nm = estimate_logit(x, d; method = :nm)
    β_bf, it_bf = estimate_logit(x, d; method = :lbfgs)
    println("Q2B: NelderMead  β = ", β_nm, "  iters = ", it_nm)
    println("Q2B: LBFGS       β = ", β_bf, "  iters = ", it_bf)
    # Truth is (-0.4, 1.2); both recover it, LBFGS in far fewer iterations
    # because the gradient tells it which way to go. Nelder-Mead only ever sees
    # function values, so it has to map out the shape of the likelihood by trial.
    #
    # The point of the question: finding the estimator numerically instead of in
    # closed form does not change its statistical properties. The MLE is whatever
    # it is; the optimizer only has to locate it. (It is consistent, not exactly
    # unbiased, so agreement with the truth here is an N = 5000 statement, not an
    # algebraic one.)
    return β_nm, β_bf
end

# =====================================================================
# Question 3: Himmelblau function
# =====================================================================
himmelblau(v) = (v[1]^2 + v[2] - 11)^2 + (v[1] + v[2]^2 - 7)^2
himmelblau(x, y) = himmelblau((x, y))     # convenience method for plotting

function himmelblau_grad(v)
    x, y = v[1], v[2]
    dx = 4 * x * (x^2 + y - 11) + 2 * (x + y^2 - 7)
    dy = 2 * (x^2 + y - 11) + 4 * y * (x + y^2 - 7)
    return [dx, dy]
end

function himmelblau_hess(v)
    x, y = v[1], v[2]
    hxx = 12 * x^2 + 4 * y - 42
    hxy = 4 * x + 4 * y
    hyy = 4 * x + 12 * y^2 - 26
    return [hxx hxy; hxy hyy]
end

# Newton's method applied to the gradient: x <- x - H(x)^{-1} ∇f(x).
# Univariate Newton divides by f'; in many dimensions that division becomes a
# linear solve. Use \ (a solve), never inv(H)*g.
function newton_min(grad, hess, x0; tol = 1e-10, maxiter = 100)
    x = float.(collect(x0))
    for iter in 1:maxiter
        step = hess(x) \ grad(x)
        x_new = x - step
        if norm(x_new - x) < tol
            return x_new, iter
        end
        x = x_new
    end
    return x, maxiter
end

function q3()
    # Part A: surface + contour. Four local minima, all with value 0.
    xs = range(-4, 4; length = 200)
    ys = range(-4, 4; length = 200)
    Z = [himmelblau(x, y) for y in ys, x in xs]   # rows = y, cols = x
    p_surf = surface(xs, ys, Z; title = "Himmelblau surface")
    p_cont = contour(xs, ys, Z; levels = 50, title = "Himmelblau contour")
    savefig(plot(p_surf, p_cont; size = (1000, 450)),
            joinpath(FIGDIR, "hw2_q3_himmelblau.png"))

    # Part B: check the analytic gradient against a central difference.
    v = [1.0, 2.0]
    h = 1e-6
    fd = [(himmelblau(v .+ h .* e) - himmelblau(v .- h .* e)) / (2h)
          for e in ([1.0, 0.0], [0.0, 1.0])]
    println("Q3B: analytic grad = ", himmelblau_grad(v), "  finite diff = ", fd)

    starts = [[3.0, 2.0], [-3.0, 3.0], [-3.0, -3.0], [3.0, -2.0], [0.0, 0.0]]
    println("Q3B: Newton from various starts:")
    for s in starts
        m, it = newton_min(himmelblau_grad, himmelblau_hess, s)
        println("   start ", s, " -> ", round.(m; digits = 4),
                "  f = ", round(himmelblau(m); digits = 8), "  (", it, " iters)")
    end
    # The four minima are near (3, 2), (-2.805, 3.131), (-3.779, -3.283) and
    # (3.584, -1.848); Newton converges to whichever basin the start lands in.
    # NOT every limit point is a minimum: newton_min solves ∇f = 0, so it is
    # equally happy at a saddle or a local max. Starting from (0,0) it walks to
    # the local MAXIMUM near (-0.271, -0.923), where f ≈ 181. Checking that the
    # Hessian is positive definite at the answer is the fix.

    println("Q3C: Nelder-Mead from the same starts:")
    for s in starts
        res = optimize(himmelblau, s, NelderMead())
        println("   start ", s, " -> ", round.(Optim.minimizer(res); digits = 4),
                "  f = ", round(Optim.minimum(res); digits = 8),
                "  (", Optim.iterations(res), " iters)")
    end
    # Nelder-Mead needs no derivatives but takes ~10x the iterations, and it too
    # converges to whichever minimum is nearest. From a single start you would
    # have concluded there is one minimum and reported it as global -- which is
    # exactly the mistake multistart is insurance against.
end

# =====================================================================
# Question 4: Market equilibrium -- roots and fixed points
# =====================================================================
# Qd = 2 p^{-0.5}, Qs = p^{0.7}.
# Z(p) = 2 p^{-0.5} - p^{0.7};  Z'(p) = -p^{-1.5} - 0.7 p^{-0.3}.
# Closed form: 2 p^{-0.5} = p^{0.7}  =>  p^{1.2} = 2  =>  p* = 2^(1/1.2).
excess_demand(p) = 2 * p^(-0.5) - p^0.7
excess_demand_prime(p) = -p^(-1.5) - 0.7 * p^(-0.3)
p_star_closed() = 2^(1 / 1.2)

# Univariate Newton for the root of Z: p <- p - Z(p)/Z'(p). Not a pinned name —
# the prompt gives the update rule, so this file stands on its own.
function newton(f, f_prime, x0; tol = 1e-10, maxiter = 1000)
    x = float(x0)
    for iter in 1:maxiter
        x_new = x - f(x) / f_prime(x)
        abs(x_new - x) < tol && return x_new, iter
        x = x_new
    end
    return x, maxiter
end

function bisection(f, a, b; tol = 1e-10, maxiter = 1000)
    fa, fb = f(a), f(b)
    @assert fa * fb < 0 "no sign change on [a, b]"
    for iter in 1:maxiter
        m = (a + b) / 2
        (b - a) / 2 < tol && return m, iter
        fm = f(m)
        if fa * fm < 0
            b = m
        else
            a, fa = m, fm
        end
    end
    return (a + b) / 2, maxiter
end

# Tâtonnement: p <- p + λ Z(p). A fixed point of T is a root of Z.
# Guard p > 0: Qd is undefined at p ≤ 0, and a large λ will overshoot there.
function tatonnement(p0; λ = 0.5, tol = 1e-10, maxiter = 10_000)
    p = float(p0)
    for iter in 1:maxiter
        p_new = max(p + λ * excess_demand(p), 1e-10)
        abs(p_new - p) < tol && return p_new, iter
        p = p_new
    end
    return p, maxiter
end

function q4()
    p_exact = p_star_closed()
    p_newton, it_n = newton(excess_demand, excess_demand_prime, 1.0)
    p_bisect, it_b = bisection(excess_demand, 0.1, 10.0)
    p_roots = find_zero(excess_demand, 1.0)
    println("Q4A: closed form p* = ", p_exact)
    println("     Newton    p* = ", p_newton, " (", it_n, " iters)")
    println("     Bisection p* = ", p_bisect, " (", it_b, " iters)")
    println("     find_zero p* = ", p_roots)
    @assert isapprox(p_newton, p_exact; atol = 1e-8)
    @assert isapprox(p_bisect, p_exact; atol = 1e-6)
    @assert isapprox(p_roots,  p_exact; atol = 1e-6)

    # Part B: Newton converges quadratically (a handful of iterations); bisection
    # halves the bracket every step, so it needs about log2((b-a)/tol) ≈ 37 of
    # them. You still prefer bisection when you have no derivative, when the
    # derivative is expensive or unreliable, or when you need a guarantee: given
    # a sign-changing bracket on a continuous function, bisection CANNOT fail.
    #
    # Newton fails from p0 = 100: Z is nearly flat out there (Z' ≈ -0.18) while
    # Z ≈ -24.9, so the tangent overshoots to p ≈ -40 and p^(-0.5) throws a
    # DomainError. A globally convergent method never has this problem.
    println("Q4B: Newton from p0 = 100 ...")
    try
        newton(excess_demand, excess_demand_prime, 100.0)
    catch e
        println("     failed as expected: ", sprint(showerror, e))
    end

    # Part C: tâtonnement. T(p) = p + λ Z(p) => T'(p*) = 1 + λ Z'(p*).
    Zp = excess_demand_prime(p_exact)
    println("Q4C: Z'(p*) = ", Zp,
            "  => converges for 0 < λ < ", -2 / Zp,
            ", fastest at λ = ", -1 / Zp)
    for λ in (0.05, 0.5, 1.0, 2.5)
        p, it = tatonnement(1.0; λ = λ)
        conv = isapprox(p, p_exact; atol = 1e-6)
        println("     λ = ", rpad(λ, 5), " -> p = ", round(p; digits = 8),
                "  iters = ", it, conv ? "  (converged)" : "  (NO convergence)")
    end
    # Z'(p*) ≈ -1.009, so |T'(p*)| < 1 requires 0 < λ < 1.98, and the modulus is
    # minimized at λ = -1/Z'(p*) ≈ 0.991 -- where T'(p*) ≈ 0 and convergence is
    # essentially Newton-fast. λ = 0.05 converges but crawls (T' ≈ 0.95);
    # λ = 0.5 is comfortable; λ ≈ 1.0 is near-optimal; λ = 2.5 is outside the
    # stable range (T'(p*) ≈ -1.52), so it oscillates with GROWING amplitude:
    # 1 -> 3.5 -> 0.16 -> 11.8 -> and then past zero, where the p > 0 guard
    # catches it at 1e-10, Z(1e-10) ≈ 2e5 fires it up to p ≈ 5e5, and it crawls
    # back down for the remaining 10,000 iterations without arriving. The exact
    # number it stops on is meaningless -- the point is that it never converges.
    # This is Lecture 3's dampening
    # parameter, and it is why an equilibrium loop that "won't converge" is
    # usually a λ problem, not a model problem.
    return p_newton
end

# =====================================================================
# Question 5: Interpolation and grid placement
# =====================================================================
ftrue(x) = log(x + 1)
const EVAL_PTS = 0:0.1:100
const EVEN_INTERIOR = collect(10.0:10.0:90.0)

# Build the 11-point grid, interpolate linearly, and report the error on a fine
# evaluation grid.
function interp_error(interior)
    grid = sort(vcat(0.0, collect(float.(interior)), 100.0))
    itp = linear_interpolation(grid, ftrue.(grid); extrapolation_bc = Line())
    fx   = ftrue.(EVAL_PTS)
    fhat = itp.(EVAL_PTS)
    return collect(fx), collect(fhat), abs.(collect(fhat) .- collect(fx))
end

total_error(interior) = sum(interp_error(interior)[3])

# Nelder-Mead has no notion of constraints, so we impose them with a penalty:
# any grid that is out of range or non-monotone gets a huge objective value and
# the simplex walks away from it.
function penalized_error(interior)
    if any(interior .<= 0) || any(interior .>= 100) || any(diff(vcat(0.0, collect(interior), 100.0)) .<= 0)
        return 1e6
    end
    return total_error(interior)
end

function optimal_grid()
    res = optimize(penalized_error, copy(EVEN_INTERIOR), NelderMead())
    return sort(Optim.minimizer(res))
end

# Part C helper: same exercise with a cubic spline. Interpolations.jl's
# cubic_spline_interpolation requires an EVENLY spaced grid, so this only works
# on the part-A grid -- which is the point.
function interp_error_cubic(interior)
    grid = range(0.0, 100.0; length = length(interior) + 2)
    itp = cubic_spline_interpolation(grid, ftrue.(grid); extrapolation_bc = Line())
    fx   = ftrue.(EVAL_PTS)
    fhat = itp.(EVAL_PTS)
    return collect(fx), collect(fhat), abs.(collect(fhat) .- collect(fx))
end

function q5()
    # Part A: evenly spaced grid.
    fx, fhat, err = interp_error(EVEN_INTERIOR)
    p1 = plot(EVAL_PTS, fx; label = "f", xlabel = "x", lw = 2)
    plot!(p1, EVAL_PTS, fhat; label = "fhat (linear, even grid)", lw = 2)
    p2 = plot(EVAL_PTS, err; label = "|error|", xlabel = "x", lw = 2)
    savefig(plot(p1, p2; size = (1000, 400)),
            joinpath(FIGDIR, "hw2_q5_even.png"))
    println("Q5A: total error (even grid, linear) = ", total_error(EVEN_INTERIOR))
    println("     max error = ", maximum(err), " at x = ", EVAL_PTS[argmax(err)])
    # The error is concentrated in the first interval and decays fast. Linear
    # interpolation's error on [x_i, x_{i+1}] is O(h^2 * max|f''|), and
    # f''(x) = -1/(x+1)^2 is 10,000 times larger at x = 0 than at x = 99. The
    # error tracks CURVATURE, not the level of f.

    # Part B: optimize the 9 interior points.
    interior_opt = optimal_grid()
    println("Q5B: optimized interior points = ", round.(interior_opt; digits = 3))
    println("     total error: even = ", round(total_error(EVEN_INTERIOR); digits = 3),
            " -> optimized = ", round(total_error(interior_opt); digits = 3))
    fx2, fhat2, err2 = interp_error(interior_opt)
    p3 = plot(EVAL_PTS, fx2; label = "f", xlabel = "x", lw = 2)
    plot!(p3, EVAL_PTS, fhat2; label = "fhat (optimized grid)", lw = 2)
    p4 = plot(EVAL_PTS, err2; label = "|error|", xlabel = "x", lw = 2)
    savefig(plot(p3, p4; size = (1000, 400)),
            joinpath(FIGDIR, "hw2_q5_opt.png"))
    # The optimizer clusters points near x = 0 and spreads them out where f is
    # nearly linear, roughly equalizing the error across intervals -- which is
    # exactly the logic behind log-spacing asset and capital grids.

    # Part C: cubic spline on the same even grid.
    _, _, err_cub = interp_error_cubic(EVEN_INTERIOR)
    println("Q5C: total error (even grid, cubic) = ", sum(err_cub))
    # The three numbers to compare: even+linear 50.44, even+cubic 41.29,
    # optimized+linear 11.03. The cubic spline buys only about 18% on the same
    # nodes; moving the nodes buys 78%. So with a fixed budget of 11 points,
    # WHERE they go matters far more than what you draw between them.
    #
    # Why the spline disappoints, which is the "what is it about this f" part:
    # spline error is O(h^4 * max|f''''|), and for f = log(x+1) we have
    # f''''(x) = -6/(x+1)^4, which is ~10^8 times larger at x = 0 than at x = 99.
    # No interpolant recovers from a first interval of [0, 10] on a function this
    # curved at the left end -- you have to put nodes there. Same argument as
    # part B, and the same reason asset and capital grids are log-spaced.
    #
    # Both together is better still, but an uneven grid rules out
    # cubic_spline_interpolation (it wants even spacing) -- PCHIP from
    # DataInterpolations.jl is the way to get both.

    return interior_opt
end

# =====================================================================
# Question 6: Expected utility via quadrature
# =====================================================================
const γ, μ, σ = 2.0, 0.0, 0.5
u_crra(c) = c^(1 - γ) / (1 - γ)

# For log c ~ N(μ, σ²), E[c^(1-γ)] = exp((1-γ)μ + (1-γ)²σ²/2) (lognormal moment),
# so E[u(c)] = exp((1-γ)μ + (1-γ)²σ²/2) / (1-γ).  With γ = 2, u(c) = -1/c and
# this is just -E[1/c] = -exp(-μ + σ²/2).
eu_exact() = exp((1 - γ) * μ + (1 - γ)^2 * σ^2 / 2) / (1 - γ)

# Gauss-Hermite: nodes/weights are for the weight e^{-t²}, so for x ~ N(μ,σ²)
# substitute x = μ + √2 σ t and divide by √π.
function eu_quadrature(n)
    t, w = gausshermite(n)
    s = 0.0
    for k in eachindex(t)
        s += w[k] * u_crra(exp(μ + sqrt(2) * σ * t[k]))
    end
    return s / sqrt(pi)
end

function eu_montecarlo(M; seed = 1)
    Random.seed!(seed)
    return mean(u_crra(exp(μ + σ * randn())) for _ in 1:M)
end

function q6()
    # Part C's closed form is computed first because parts A and B report their
    # errors against it; it is labelled and discussed as part C below.
    exact = eu_exact()

    println("Q6A: Gauss-Hermite quadrature")
    for n in (5, 9, 15)
        gh = eu_quadrature(n)
        println("   n = ", rpad(n, 3), " E[u] = ", gh,
                "  abs err = ", abs(gh - exact), "  (", n, " evaluations)")
    end

    println("Q6B: Monte Carlo")
    for M in (100, 10_000, 1_000_000)
        mc = eu_montecarlo(M)
        println("   M = ", rpad(M, 9), " E[u] = ", mc,
                "  abs err = ", abs(mc - exact), "  (", M, " evaluations)")
    end

    println("Q6C: exact E[u(c)] = ", exact)
    # 9 quadrature nodes beat a million Monte Carlo draws by several orders of
    # magnitude: Gaussian quadrature is exact for polynomials up to degree 2n-1
    # and the integrand here is analytic, while MC error falls only like
    # 1/sqrt(M). Go back to Monte Carlo when the integrand is not smooth (kinks,
    # discrete choices, occasionally binding constraints) or when the dimension
    # is high enough that a tensor grid is hopeless -- MC's 1/sqrt(M) rate does
    # not care about dimension, which is its one great virtue.
    return exact
end

# ---------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    q1(); q2(); q3(); q4(); q5(); q6()
end
