################################################################################
# CodeAlong 8: Estimation
################################################################################

# Estimation is optimization. Across the four estimators below:
#
#   OLS   the sum of squared residuals                         (closed form)
#   MLE   -log of the probability of the data you saw          (Optim)
#   GMM   ḡ(θ)' W ḡ(θ), sample moments the model says are 0    (Optim)
#   SMM   the same, with ḡ simulated                           (Optim + draws)
#
# data/wages.csv holds 2,000 workers (lwage, educ, exp), simulated -- which is
# the only reason the "did it work?" checks below mean anything. The truth:
# lwage = 1.00 + 0.09*educ + 0.015*exp + e, e ~ N(0, 0.35²). make_wages.jl
# regenerates the file from an explicit seed.

using CSV, DataFrames, GLM, Statistics, LinearAlgebra, Optim, Plots, Random, Sobol

β_true, σ_true = [1.0, 0.09, 0.015], 0.35


################################################################################
# 1. Getting the data in
################################################################################

# Path relative to THIS file, not to wherever your REPL happens to be sitting.
csv_path = joinpath(@__DIR__, "data", "wages.csv")
isfile(csv_path) || (csv_path = joinpath(@__DIR__, "wages.csv"))  # both in one folder

df = CSV.read(csv_path, DataFrame)

first(df, 5)
describe(df, :mean, :std, :min, :max)

# A DataFrame is a table with named columns -- df.lwage is just a Vector -- and
# the verbs are select / filter / groupby + combine / the joins.
educ_groups = groupby(df, :educ)

# use the "=>" operator as: source column => function => output column name
combine(educ_groups, :lwage => mean => :mean_lwage, nrow => :n)
nrow(filter(:educ => >=(16), df))

# Statistics is a stdlib, and it's where your DATA MOMENTS live -- GMM and SMM
# below both work by matching these to what the model predicts.
mean(df.lwage), std(df.lwage), cor(df.educ, df.lwage)

# Estimators want arrays, not tables. This is the line that gets you there.
y = df.lwage # basically a vector
y = df[:, :lwage] # actually a vector
X = [ones(nrow(df)) Matrix(df[:, [:educ, :exp]])]
n, k = size(X)

# A new regressor is a new COLUMN, e.g. df.exp2 = df.exp .^ 2. Do not write
# exp^2 inside a StatsModels @formula -- there ^ means interaction expansion,
# not a power. GLM.jl gives you OLS as the one-liner the rest of the world
# uses, lm(@formula(...), df). MLE, GMM, and SMM don't have
# that luxury: nobody packaged your likelihood, so from Section 3 on we write
# the estimator ourselves.


################################################################################
# 2. OLS
################################################################################

cols = ["const", "educ", "exp"]

# We wrote the normal equations by hand, (X'X) \ (X'y), in an earlier
# code-along -- OLS is the one estimator here with a closed form, which is
# why it's worth seeing from scratch once. Nobody does it that way twice:
# GLM.jl fits the same model and hands back standard errors, t-stats, and
# p-values for free.
ols = lm(@formula(lwage ~ educ + exp), df)
ols                                        # the coefficient table, for free

b_ols = coef(ols)
se_ols = stderror(ols)

################################################################################
# 3. Maximum likelihood
################################################################################

# y_i ~ N(x_i'b, s²), independent across i, so
#
#     log L(b, s) = Σ_i [ -log s - ½ log(2π) - ½ ((y_i - x_i'b)/s)² ]
#
# Two implementation details matter here:
#   1. Optimizers MINIMIZE. Hand them -log L, and mind the sign afterwards.
#   2. s must stay positive and the optimizer does not know that. Optimize over
#      log s and exponentiate inside: the parameter space becomes all of R⁴, no
#      constraint machinery, no crash when a line search tries s = -3. The same
#      trick handles probabilities and anything else bounded.

function neg_loglik(θ, X, y)
    b, s = θ[1:end-1], exp(θ[end])                       # <- the transform
    e = y .- X * b
    n = length(y)
    return n * log(s) + n * log(2π) / 2 + sum(abs2, e) / (2 * s^2)
end

# Standard errors are CURVATURE: the Hessian of -log L at θ̂ is the observed
# information, its inverse the asymptotic variance. Central differences -- crude,
# but simple. (ForwardDiff.jl would compute this exactly instead.)
function num_hessian(f, x; h = 1e-5)
    k = length(x)
    H = zeros(k, k)
    for i in 1:k, j in 1:k
        ei = zeros(k); ei[i] = h
        ej = zeros(k); ej[j] = h
        H[i, j] = (f(x+ei+ej) - f(x+ei-ej) - f(x-ei+ej) + f(x-ei-ej)) / (4h^2)
    end
    return H
end

obj = θ -> neg_loglik(θ, X, y)
res = optimize(obj, [zeros(k); log(std(y))], LBFGS())
θ̂  = Optim.minimizer(res)
se  = sqrt.(diag(inv(num_hessian(obj, θ̂))))

# Report σ̂, not log σ̂. The delta method for a scalar transform is the chain
# rule: if g(x) = exp(x), then se(g(x̂)) = exp(x̂) · se(x̂).
θ_mle  = [θ̂[1:end-1]; exp(θ̂[end])]
se_mle = [se[1:end-1]; exp(θ̂[end]) * se[end]]

# check positive definiteness
isposdef(num_hessian(obj, θ̂))
println("max |β̂_MLE - β̂_OLS| = ", maximum(abs.(θ_mle[1:3] .- b_ols)))


################################################################################
# 4. GMM
################################################################################

# MLE needed the whole distribution of the error. GMM needs only a list of
# things the model says average to zero, E[g(w_i, θ₀)] = 0: swap E for a sample
# average and push it as close to zero as you can, θ̂ = argmin ḡ(θ)' W ḡ(θ).
# Nowhere below do we assume e is normal, which is why the same machinery fits
# an Euler equation or a BLP demand system, where no likelihood is available.

# Our model, lwage = b₀ + b₁·educ + b₂·exp + e with Var(e) = σ², has four
# parameters and makes four claims about e. One COLUMN per moment, one ROW per
# worker -- this is g(w_i, θ). Columns 5-6 wait until we over-identify.
function moments(θ)
    b, σ = θ[1:3], θ[4]
    e = y .- X * b
    return hcat(e,               # E[e]      = 0    the errors average out
                e .* df.educ,    # E[e·educ] = 0    ...and say nothing about educ
                e .* df.exp,     # E[e·exp]  = 0    ...or about exp
                e .^ 2 .- σ^2,   # E[e²]     = σ²   ...and their spread IS σ
                e .^ 3,          # E[e³]     = 0     } true only if
                e .^ 4 .- 3σ^4)  # E[e⁴]     = 3σ⁴   } e is normal
end

gbar(θ, use) = vec(mean(moments(θ)[:, use], dims = 1))

# G = ∂ḡ/∂θ' has no closed form once the moments are nonlinear in θ. Difference
# it, exactly like the Hessian in Section 3.
function num_jacobian(f, x; h = 1e-5)
    cols = map(eachindex(x)) do j
        ej = zeros(length(x)); ej[j] = h
        (f(x + ej) - f(x - ej)) / (2h)
    end
    return reduce(hcat, cols)
end

# `use` picks which moments to stack. Only σ² and σ⁴ ever appear above, so ±σ
# fit equally well -- start σ positive and it stays there.
function fit_gmm(use, W; start = [0.0, 0.0, 0.0, 1.0])
    obj = θ -> (g = gbar(θ, use); dot(g, W * g))
    θ̂  = Optim.minimizer(optimize(obj, start, LBFGS()))

    # Sandwich variance: bread * meat * bread, where S carries the actual noise.
    G     = num_jacobian(θ -> gbar(θ, use), θ̂)
    gi    = moments(θ̂)[:, use]
    S     = (gi' * gi) ./ n                       # Var(√n ḡ), estimated at θ̂
    bread = inv(G' * W * G)
    V     = bread * (G' * W * S * W * G) * bread ./ n
    return θ̂, sqrt.(diag(V)), S
end

# JUST-IDENTIFIED: 4 moments, 4 parameters. You can drive every moment to
# exactly zero, so W has nothing to trade off -- and "residuals uncorrelated
# with the regressors" is what OLS assumes, so this had better reproduce OLS.
θ_gmm, se_gmm, _ = fit_gmm(1:4, Matrix(1.0I, 4, 4))
println("max |β̂_GMM - β̂_OLS| = ", maximum(abs.(θ_gmm[1:3] .- b_ols)))
println("σ̂_GMM = ", θ_gmm[4], "   σ̂_MLE = ", θ_mle[4])

θ_alt, _, _ = fit_gmm(1:4, Matrix(100.0I, 4, 4))
println("W scaled by 100:  max |Δθ̂| = ", maximum(abs.(θ_alt .- θ_gmm)))

# OVER-IDENTIFIED: columns 5-6 are the normality MLE assumed, added back a piece
# at a time -- true, but not implied by the first four. Six equations, four
# unknowns: they cannot all hold at once, and now W decides which ones give.
θ_I, se_I, S_I = fit_gmm(1:6, Matrix(1.0I, 6, 6))

# The efficient W is S⁻¹ -- down-weight the moments your data measure noisily --
# but S needs a θ̂ and a good θ̂ wanted W. Hence TWO-STEP.
θ_2s, se_2s, _ = fit_gmm(1:6, inv(S_I); start = θ_I)
println("W = I vs two-step:  max |Δθ̂| = ", maximum(abs.(θ_2s .- θ_I)))
println("se(b₁):  W = I ", se_I[2], "   two-step ", se_2s[2])

# Both are consistent; two-step is tighter. Look at the scale of what you stack:
# e is around 0.35, so e⁴ is around 0.015 -- W = I is a claim about units.


################################################################################
# 5. Sobol sequences: quasi-random draws
################################################################################

# Monte Carlo is a mediocre way to compute an integral: random points clump and
# leave gaps, so the error falls only like 1/√R. A Sobol sequence fills the unit
# cube much more evenly by construction, spacing each new point to land in
# whichever gap the earlier points left biggest. Sobol.jl builds one:

function sobol_points(n, d)
    s = SobolSeq(d)
    pts = zeros(n, d)
    for i in 1:n
        pts[i, :] = next!(s)
    end
    return pts
end

Random.seed!(1)
pseudo = rand(300, 2)
sobol  = sobol_points(300, 2)

plot(scatter(pseudo[:, 1], pseudo[:, 2]; title = "pseudorandom"),
     scatter(sobol[:, 1], sobol[:, 2]; title = "Sobol"),
     layout = (1, 2), ms = 2.5, msw = 0, legend = false,
     xlims = (0, 1), ylims = (0, 1), aspect_ratio = 1, size = (900, 430))

# To use these as shocks you push them through a quantile function: uniforms
# become normals via quantile(Normal(), u) with Distributions.jl


################################################################################
# 6. Habits worth keeping
################################################################################

# One file, because this is a short demo. A project that outlives the week
# wants three -- structs.jl (what the model IS), functions.jl (what you DO to
# it), master.jl (the script) -- plus separate code/, data/ and output/ folders,
# and nothing in output/ ever hand-edited. If a number in your paper cannot be
# regenerated by re-running the script, it is not a result.
#
# Two habits get you there. Environments:
#
#     julia> ]                  (enter Pkg mode)
#     pkg> activate .           an environment that lives in this folder
#     pkg> add CSV DataFrames   writes Project.toml AND Manifest.toml
#     pkg> instantiate          what a co-author runs to get YOUR versions
#
# Commit both files. And seeds: every random draw here was preceded by an
# explicit Random.seed! call, so results are reproducible even though the RNG
# itself is the global one.
#
# Further references:
# - QuantEcon (https://julia.quantecon.org/)
# - Fernández-Villaverde's notes (https://www.sas.upenn.edu/~jesusfv/)
#
# And the best teacher: pick an economic question YOU care about, write the
# model down, and build it. You now have every tool you need to do that badly,
# which is the only way anyone has ever started.
