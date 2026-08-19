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

# MLE needed the whole distribution of the error. GMM needs only things the
# model says are zero at the true parameter, E[g(w_i, θ₀)] = 0. Here that is
# exogeneity, E[z_i (y_i - x_i'b)] = 0 -- but the machinery is identical for an
# Euler equation, an IV restriction, or BLP's E[ξ_j z_j] = 0.

moments(b, X, y, Z) = Z .* (y .- X * b)              # n × m, one row per worker
gbar(b, X, y, Z)     = vec(mean(moments(b, X, y, Z), dims = 1))

function gmm_obj(b, X, y, Z, W)
    g = gbar(b, X, y, Z)
    return dot(g, W * g)
end

function fit_gmm(X, y, Z, W; start = zeros(size(X, 2)))
    res = optimize(b -> gmm_obj(b, X, y, Z, W), start, LBFGS())
    b̂ = Optim.minimizer(res)

    # Sandwich variance: bread * meat * bread, with G = ∂ḡ/∂b'. Here g is
    # linear in b, so G = -(1/n) Z'X exactly. GMM is only as hard as your
    # model is. "Bread" is (G'WG)⁻¹, the same object you'd invert to solve
    # the GMM first-order conditions if this were linear; "meat" is the
    # G'WSWG in the middle that brings in the actual moment variance S.
    n     = size(X, 1)
    G     = -(Z' * X) ./ n
    g     = moments(b̂, X, y, Z)
    S     = (g' * g) ./ n                            # Var(√n ḡ), estimated at b̂
    bread = inv(G' * W * G)
    meat  = G' * W * S * W * G
    V     = bread * meat * bread ./ n
    return b̂, sqrt.(diag(V)), S
end

# JUST-IDENTIFIED first: 3 moments, 3 parameters, Z = X. "The residual is
# uncorrelated with the regressors" is the assumption OLS runs on, so this had
# better reproduce OLS exactly -- and W had better be irrelevant, because with
# as many equations as unknowns you can drive every moment to exactly zero.
b_gmm, se_gmm, _ = fit_gmm(X, y, X, Matrix(1.0I, 3, 3))
println("max |β̂_GMM - β̂_OLS| = ", maximum(abs.(b_gmm .- b_ols)))

b_alt, _, _ = fit_gmm(X, y, X, Matrix(100.0I, 3, 3))
println("W scaled by 100:  max |Δβ̂| = ", maximum(abs.(b_alt .- b_gmm)))

# OVER-IDENTIFIED: add moments. Under exogeneity ANY function of x is
# uncorrelated with the error, so educ² and exp² are legitimate extra moment
# conditions. Five equations, three unknowns -- they cannot all hold at once,
# and now W decides which ones give.
Z = [X df.educ .^ 2 df.exp .^ 2]
b_I, se_I, S_I = fit_gmm(X, y, Z, Matrix(1.0I, 5, 5))


# The efficient W is S⁻¹ -- down-weight the moments your data measure noisily --
# but S has to be estimated at some b̂, and a good b̂ wanted W. Hence TWO-STEP.
b_2s, se_2s, _ = fit_gmm(X, y, Z, inv(S_I); start = b_I)
println("W = I vs two-step:  max |Δβ̂| = ", maximum(abs.(b_2s .- b_I)))

# Both are consistent; they differ because they weight the same five moments
# differently. (Look at the scale of Z: educ² is in the hundreds, so W = I is a
# claim about units, not economics.)



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
