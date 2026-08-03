################################################################################
# Load Packages
################################################################################

using Optim, Plots

################################################################################
# Univariate Boxed Constrained Optimization
################################################################################

f(x, y) = (x-y)^2


####Brent's method is the default for univariate boxed-constrained optimization
opt = optimize(x->f(x, 1.0),   #function
                -5.0,           #lower bound
                5.0             # upper bound
                )

opt.minimizer
opt.minimum

#The "x->f(x)" operator defines a function inline that maps x to f(x)
# this is useful for when optimizing over a subset of parameters. 
# Here, we optimize over x for a fixed y. 

################################################################################
# Multivariate Optimization Optimization Derivative Free
################################################################################


# Rosenbrock(x, y) = (1 - x)² + 100 * (y - x²)²
# minimizer = (1, 1), minimum = 0

Rosenbrock(x::Vector{Float64}) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2


##evaluate function at a bunch of points so we can plot it
x_grid = collect(-3:0.01:3)
nx = length(x_grid)
z_grid = zeros(nx, nx)

@time for i = 1:nx, j = 1:nx
    guess = [x_grid[i], x_grid[j]]
    z_grid[i,j] = Rosenbrock(guess)
end

print("min, minimizer = ", minimum(z_grid), ", (", x_grid[argmin(z_grid)[1]], ", ", x_grid[argmin(z_grid)[2]], ")")

surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera = (45,45))
contourf(x_grid, x_grid, log.(1 .+ z_grid), 
            seriescolor=:inferno, xlabel="x_1", ylabel = "x_2")
scatter!([1],[1], color = "white", label="Minimizer")

# use a gif to visualize 3d function
@gif for i = 1:2:360
    surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera = (i,45))
end


#Nelder-Mead is default option for multivariate optimization
guess = [0.0, 0.0]
@time opt = optimize(Rosenbrock, guess)
opt.minimizer #Success!
opt.minimum

###Let's specify LBFGS to be the solver.
opt = optimize(Rosenbrock, [-5.0;-5.0], LBFGS())
opt.minimizer #also worked
opt.minimum



################################################################################
# Multivariate Optimization Optimization with Derivates
################################################################################

#Define Gradient of the Rosenbrock function
function g(G, x::Vector{Float64})
    G[1] = -2.0 * (1.0 - x[1]) -400.0 * x[1] * (x[2] - x[1]^2)
    G[2] = 200.0 * (x[2] - x[1]^2)
    G #return
end

#Rosenbrock's Hessian
function h(H, x::Vector{Float64})
    H[1,1] = 2 - 400.0*x[2] + 1200.0*x[1]^2
    H[1,2] = -400.0 * x[1]
    H[2,1] = -400.0 * x[2]
    H[2,2] = 200.0
    H #return
end

#Newton's Method is default when providing gradient and Hessian
guess = [0.0, 0.0]
opt = optimize(Rosenbrock, g, h, guess)
opt.minimizer #works!
opt.minimum

################################################################################
# Many local minima
################################################################################

function Greiwank(x::Array{Float64,1})
    val = (1/4000)*sum(x.^2) - prod(cos.(x./sqrt(length(x)))) + 1
    val
end

##evaluate function at a bunch of points
x_grid = collect(-5:0.01:5)
nx = length(x_grid)
z_grid = zeros(nx, nx)

for i = 1:nx, j = 1:nx
    guess = [x_grid[i], x_grid[j]]
    z_grid[i,j] = Greiwank(guess)
end

##plots
surface(x_grid, x_grid, z_grid, seriescolor=:viridis, camera = (50,70))
contourf(x_grid, x_grid, z_grid, seriescolor=:inferno)


#global optimum at (0,0)
guess_init = [3.0, 3.0]
opt = optimize(Greiwank, guess_init) #this fails!
opt.minimizer
opt.minimum 

#now this works!
guess_init = [2.0, 2.0]
opt = optimize(Greiwank, guess_init) #this works!
opt.minimum
opt.minimizer

#try multiple starts!
function Multistart()
    x_grid = collect(-5:2.0:5)
    nx = length(x_grid)
    minimum, minimizers = 100, [100, 100] #preallocate bad values for minimum and minimizes

    for i = 1:nx, j = 1:nx
        guess = [x_grid[i], x_grid[j]] #starting guess
        opt = optimize(Greiwank, guess) #nelder-mead with new starting guess
        if opt.minimum<minimum #new minimum!
            minimum = opt.minimum #update
            minimizers = opt.minimizer #update
        end
    end
    minimum, minimizers #return
end
min, minimizers = Multistart()

# global optimization packages: BlackBoxOptim.jl, Optimization.jl

################################################################################
# OLS Example
################################################################################

using Distributions, Random

##run the same OLS as first class
dist = Normal(0,1)
β_0 = 1.0
β_1 = 2.0
β_2 = 3.0
n = 10000
x = rand(n).*10
x2 = x.^2
Random.seed!(1234)  ###important to remember!!
ϵ = rand(dist, n)
Y_true = β_0 .+ β_1.*x + β_2.*x2 .+ ϵ
X = hcat(ones(n), x, x2)
β_ols = inv(X' * X) * X' * Y_true

####Bad Squared function: This will draw new ϵ every iteration
function sq_error(β::Array{Float64,1})
    β_0, β_1, β_2 = β[1], β[2], β[3] #unpack β
    Random.seed!(1234) #Must uncomment this to get it to work
    ϵ = rand(dist, n) #draw epsilons
    Y_true = 1.0 .+ 2.0.*x + 3.0.*x2 .+ ϵ
    Y_predict = β_0 .+ β_1.*x + β_2.*x2 #true and predicted value
    error = sum((Y_true.-Y_predict).^2) #sum of squared error
    error #return
end


#do OLS with nelder-mead
guess_init = [0.0, 0.0, 0.0]
opt = optimize(sq_error, guess_init) #it works

opt.minimizer

################################################################################
# Automatic differentiation
################################################################################

# Instead of hand-coding the gradient/Hessian (see above) or letting Optim use
# finite differences, we can ask for exact derivatives via automatic
# differentiation (ForwardDiff.jl under the hood):

opt = optimize(Rosenbrock, [-5.0, -5.0], LBFGS(); autodiff = :forward)
opt.minimizer
opt.minimum

# Requirement: your function must be written in generic Julia. If you hard-coded
# Float64 everywhere (like Rosenbrock above does with ::Vector{Float64}),
# ForwardDiff's dual numbers can't flow through. Loosen the signature:
Rosenbrock2(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
opt = optimize(Rosenbrock2, [-5.0, -5.0], LBFGS(); autodiff = :forward)

################################################################################
# Root Finding
################################################################################

using Roots

# Economic example: market clearing. Find the price where excess demand = 0.
demand(p) = 10.0 * p^(-0.5)
supply(p) = 2.0 * p^(0.7)
excess(p) = demand(p) - supply(p)

# Bracketing method (bisection/Brent-style): give an interval with a sign change.
# This CANNOT fail if the function is continuous on the bracket.
p_star = find_zero(excess, (0.1, 10.0))
excess(p_star)   # ~0

# Newton-type from a single starting guess (fast, but can wander):
p_star2 = find_zero(excess, 1.0)

# Manual bisection, so you see there is no magic:
function bisect(f, a, b; tol = 1e-10)
    fa = f(a)
    while b - a > tol
        m = (a + b) / 2
        if sign(f(m)) == sign(fa)
            a, fa = m, f(m)
        else
            b = m
        end
    end
    (a + b) / 2
end
bisect(excess, 0.1, 10.0)

# Systems of nonlinear equations: NonlinearSolve.jl
using NonlinearSolve

# Write the residuals as a function of the unknowns x and parameters p.
# Every equation is written as "something = 0".
residuals(x, p) = [x[1]^2 + x[2]^2 - 1.0,   # unit circle
                   x[1] - x[2]]             # 45-degree line

prob = NonlinearProblem(residuals, [0.5, 0.5])   # residuals + starting guess
sol = solve(prob)                                # let it pick the algorithm
sol.u    # (1/sqrt(2), 1/sqrt(2))

# Parameters go in p, so you can re-solve without rewriting the function:
residuals_p(x, p) = [x[1]^2 + x[2]^2 - p, x[1] - x[2]]
sol2 = solve(NonlinearProblem(residuals_p, [0.5, 0.5], 4.0))
sol2.u   # circle of radius 2

# You can name an algorithm if you want; derivatives are handled by AD for you.
solve(prob, NewtonRaphson()).u

################################################################################
# Fixed Points
################################################################################

# A fixed point of T is x with T(x) = x. Simple iteration: x <- T(x).
# Example: x = cos(x) (a contraction near its fixed point)
T(x) = cos(x)

function fixed_point(T, x0; tol = 1e-12, maxiter = 10_000, λ = 1.0)
    x = x0
    for it in 1:maxiter
        x_new = λ * T(x) + (1 - λ) * x    # λ < 1 dampens unstable iterations
        if abs(x_new - x) < tol
            return x_new, it
        end
        x = x_new
    end
    error("no convergence")
end

xstar, iters = fixed_point(T, 1.0)
xstar - cos(xstar)   # ~0

# Dampening (λ < 1) trades speed for stability -- essential when T is not a
# contraction, common in equilibrium-price iterations.
fixed_point(T, 1.0; λ = 0.5)

# Note: a fixed point is a root of f(x) = T(x) - x, so Roots.jl works too:
find_zero(x -> T(x) - x, 1.0)

# Next week: the value function is *the* fixed point in economics
# (the Bellman operator is a contraction with modulus β).
