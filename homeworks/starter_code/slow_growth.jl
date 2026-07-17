# slow_growth.jl -- Homework 1, Question 7
# This code solves the deterministic optimal investment problem... slowly.
# Your job: make it fast (functions, @kwdef struct, type stability,
# pre-allocation) and report the speedup with BenchmarkTools.jl.
# Do NOT change what it computes -- check your V against this one!

theta = 0.36
delta = 0.025
beta = 0.99
nk = 1000
k_grid = collect(range(0.01, 45.0, length = nk))

V = zeros(nk)
V_next = zeros(nk)
policy = zeros(nk)

tol = 1e-4
max_diff = 10
n = 0

while max_diff > tol
    global n += 1
    global max_diff = 0        # globals galore!
    for i = 1:nk
        max_util = -1e10
        budget = k_grid[i]^theta + (1 - delta) * k_grid[i]
        for j = 1:nk
            c = budget - k_grid[j]
            if c > 0
                # rebuilding this array access every time, mixed types, etc.
                util = log(c) + beta * V[j]
                if util > max_util
                    max_util = util
                    policy[i] = k_grid[j]     # policy is Float64, fine...
                end
            end
        end
        V_next[i] = max_util
    end
    for i = 1:nk
        d = abs(V_next[i] - V[i])
        if d > max_diff
            max_diff = d
        end
    end
    global V = copy(V_next)    # allocates a fresh copy every iteration
end

println("Converged after ", n, " iterations")
