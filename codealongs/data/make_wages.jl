################################################################################
# make_wages.jl -- generates data/wages.csv, the dataset CodeAlong8.jl reads
################################################################################
#
# Run once, from the codealongs/ folder:
#     julia --project=. data/make_wages.jl
#
# The data are simulated so that we know the truth and can check every estimator
# in CodeAlong8.jl against it. The seed is explicit, so this file IS the dataset:
# anyone can regenerate wages.csv byte-for-byte.
#
#     lwage_i = 1.00 + 0.09*educ_i + 0.015*exp_i + e_i,     e_i ~ N(0, 0.35^2)

using Random, Printf

n, β, σ, seed = 2_000, [1.0, 0.09, 0.015], 0.35, 20260819

rng   = Random.seed!(seed)
educ  = rand(8:20, n)
exper = rand(0:40, n)
lwage = β[1] .+ β[2] .* educ .+ β[3] .* exper .+ σ .* randn(n)

open(joinpath(@__DIR__, "wages.csv"), "w") do io
    println(io, "lwage,educ,exp")
    for i in 1:n
        @printf(io, "%.6f,%d,%d\n", lwage[i], educ[i], exper[i])
    end
end

println("wrote data/wages.csv: ", n, " workers")
