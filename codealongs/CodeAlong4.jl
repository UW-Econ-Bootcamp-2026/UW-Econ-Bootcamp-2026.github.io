

################################################################################
# Packages
################################################################################

using Optim, Interpolations, Plots, Polynomials



################################################################################
# Runge's function and phenomenon
################################################################################

Runge(x) = 1 /(1 + 25*x^2)
x_fine = collect(-1.0:0.01:1.0)
x_coarse = collect(range(-1.0, length = 6, stop = 1.0))
plot(x_fine, Runge.(x_fine))
#The ! adds the plot/scatter on top of the previous plot rather than make a new plot
scatter!(x_coarse, Runge.(x_coarse))

#5-degree polynomial fit
poly_fit = fit(x_coarse, Runge.(x_coarse), 5)
plot!(x_fine, poly_fit.(x_fine))

#try a higher-degree -- surely that will help!
x_coarse_2 = collect(range(-1.0, length = 10, stop = 1.0))
poly_fit_2 = fit(x_coarse_2, Runge.(x_coarse_2), 9)
plot!(x_fine, poly_fit_2.(x_fine))

# Let's Try Interpolation Instead
x_range = -1.0:0.25:1.0
runge_range = Runge.(x_range)
runge_linear = linear_interpolation(x_range, runge_range)
runge_spline = cubic_spline_interpolation(x_range, runge_range)
plot(x_fine, 
    [Runge.(x_fine), 
        runge_linear.(x_fine), 
        runge_spline.(x_fine)],
    linewidth=3)
plot(x_fine, [Runge.(x_fine), runge_linear.(x_fine)])
plot(x_fine, [Runge.(x_fine), runge_spline.(x_fine)])

################################################################################
# Interpolation 
################################################################################

####Linearly Interpolating log function
x = 0.1:2:10.1
y = log.(x)
x_fine = collect(0.1:0.1:10.1)
y_fine = log.(x_fine)

#plot
plot(x_fine, y_fine)
scatter!(x, y, markersize = 4)

#Linear Interpolation
interp_y = linear_interpolation(x, y)
y_fine_interp = interp_y.(x_fine)
plot!(x_fine, y_fine_interp)

# put the points where there is the most curvature
x2=[0.1,0.5,1.0,1.5,2.0,5.0,10.1]
y2 = log.(x2)
interp_y2 = linear_interpolation(x2,y2)
y_interp = interp_y2.(x_fine)
plot(x_fine, [y_fine, interp_y2.(x_fine)])
scatter!(x2, y2, markersize = 4)

#Extrapolation
log(11)
interp_y(11) #This will give you an error because we haven't allowed for extrapolation

#Need to set the extrapolation boundary coundition
#Line() will just extend the line from the last region
#Flat() will set everything equal to the last gridpoint. 
interp_y_extra = linear_interpolation(x,y,extrapolation_bc = Line()) #Flat()
interp_y_extra(11)

# add a point far out
x2=[0.1,0.5,1.0,1.5,2.0,5.0,10.1,19.0]
y2 = log.(x2)
interp_y2 = linear_interpolation(x2,y2)
y_interp = interp_y2.(x_fine)
x_fine_2  = collect(0.1:0.1:20.1)
y_fine_2 = log.(x_fine_2)
plot(x_fine_2, y_fine_2)
scatter!(x2, y2, markersize = 4)
scatter!([10.0,20.0], [log.(10.0),interp_y_extra(20)])
#Extrapolation works okay when close to actual gird, but can struggle far outside grid
log(11)
interp_y_extra(11)
log(20)
interp_y_extra(20)

#We can make our approximation better with uneven grid
x_uneven = 0.1 .+ 10.0 .*collect(0.0:0.2:1.0).^2
y_uneven = log.(x_uneven)
plot(x_fine, y_fine)
scatter!(x_uneven, y_uneven, markersize = 4)

interp_y = linear_interpolation(x_uneven, y_uneven)
y_fine_interp = interp_y.(x_fine)
plot(x_fine, [y_fine, y_fine_interp])
scatter!(x_uneven, y_uneven, markersize = 4)


####Cubic Interpolating log function
x = 0.1:2:10.1
y = log.(x)
x_fine = collect(0.1:0.1:10.1)
y_fine = log.(x_fine)

#plot
plot(x_fine, y_fine)
scatter!(x, y, markersize = 4)

#Cubic Interpolation
cubic_interp_y = cubic_spline_interpolation(x, y)
y_fine_cubic = cubic_interp_y.(x_fine)
plot(x_fine, [y_fine, y_fine_cubic])
scatter!(x, y, markersize = 4)

###Extrapolation 
cubic_interp_y = cubic_spline_interpolation(x, y, extrapolation_bc = Line())
log(12)
cubic_interp_y(12)

#cubic_spline_interpolation does not support uneven grid in Interpolations.jl
x_uneven = 0.1 .+ 10.0 .*collect(0.0:0.2:1.0).^2
y_uneven = log.(x_uneven)
interp_y = cubic_spline_interpolation(x_uneven, y_uneven) #Error because x is not evenly spaced. 


# PCHIP
using DataInterpolations

pchip_y = PCHIPInterpolation(y, x, extrapolation=ExtrapolationType.Extension)
x_wide = collect(0.05:0.01:19.0)
plot(x_wide, [log.(x_wide), pchip_y.(x_wide), cubic_interp_y.(x_wide)], label=["True" "PCHIP" "Cubic Spline"])

################################################################################
# Quadrature
################################################################################

using FastGaussQuadrature, LinearAlgebra

#Gauss-Legendre Quadrature for approximating integral between -1 and 1
#Works well with n nodes if function is well approximated by polynomial of degree n
#Get nodes x and weights w
x, w = gausslegendre(3)

#Function F
F(x) = x^4 - 3x^3 - 4x^2
#Derivative of function f
f(x) = 4x^3 - 9x^2 - 8x
#Calculate intergral appoximation
I = w ⋅ f.(x)
#Calculate true value
F(1) - F(-1)


# Gauss-Hermite Quadrature for approximating integral of function g(x) = f(x)*exp(-x^2)
# Again, works well with n nodes if f is well approximated by polynomial of degree n
# Here we will approximate the expection E[f(x)] when x is distributed according to standard normal

#Get nodes and weights
x,w = gausshermite(9)
#Change of variable: https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature
g_tilde(x,f) = f(sqrt(2)x)/sqrt(π)

#Expectation approximation
w ⋅ g_tilde.(x,f)

#Monte Carlo Simulation Approximation
using Distributions
val = 0
N = 1_000_000
for i = 1:N
    val += f(rand(Normal()))/N
end
val

################################################################################
# Growth, Revisited: Interpolation + Optimization Instead of Grid Search
################################################################################

# Lecture 2 solved this model by grid search: at every k, loop over every k'
# on the grid and keep the best one. That works, but needs a fine grid to be
# accurate, and "the best k' on the grid" is never quite the true optimum.
# Now that we can interpolate V between grid points (this lecture) and
# optimize continuously over k' (Lecture 3), we can do much better with far
# fewer grid points.

# We use Julia's built-in @kwdef macro to define default values in our structs.
# (Older code often uses the Parameters.jl package and its @with_kw macro for this;
#  @kwdef is built in and does the same job.)

### Struct for our model paramters
@kwdef struct ModelParameters

    β::Float64 = 0.99 #We can define default values because of the @kwdef macro.
    δ::Float64 = 0.025
    α::Float64 = 0.36

    k_grid::Vector{Float64}=collect(range(0.1, length = 100, stop = 45.0)) #### Far fewer grid points than grid search!
    N_k::Int64 = length(k_grid)

    tol::Float64 = 10^-5
    max_iter::Int64 = 10^4

end


struct ModelSolutions

    V::Vector{Float64}
    kp::Vector{Float64}

end

function initialize()
    para = ModelParameters()
    V = zeros(para.N_k)
    kp = zeros(para.N_k)
    sols = ModelSolutions(V, kp)
    return para, sols
end

### Bellman operator
function bellman(para, sols)
    (; β, δ, α, k_grid) = para
    (; V, kp) = sols

    V_next = zero(V)
    kp_next = zero(kp)

    #Interpolate value function for continuation value
    V_interp = linear_interpolation(k_grid, V)

    for (i_k, k) = enumerate(k_grid)

        budget = k^α + (1-δ)*k

        #Replace grid search with box constrained optimization!
        obj(kp) = -(log(budget-kp) + β*V_interp(kp))
        lo = minimum(k_grid)
        # high = budget # problem: extrapolation. Extrapolation can introduce issues, we'll just truncate
        high = min(budget, maximum(k_grid))
        
        opt = optimize(obj, lo, high)

        V_next[i_k] = -opt.minimum
        kp_next[i_k] = opt.minimizer

    end

    return V_next, kp_next

end



### Solve model
function solve_model!(para, sols)
    (; tol, max_iter) = para
    (; V, kp) = sols

    V_next = zero(V)
    kp_next = zero(kp)
    max_diff = tol + 10.0
    n = 0
    while max_diff > tol && n < max_iter
        n +=1
        V_next, kp_next = bellman(para,sols)

        max_diff = maximum(abs.(V_next - V))
        V .= V_next
        kp .= kp_next

        @show n, max_diff

    end
end

para, sols = initialize();

#So fast and just as accurate!
@time solve_model!(para,sols)

plot(para.k_grid, sols.V)
plot(para.k_grid, sols.kp)
plot!(collect(0:45), collect(0:45))


################################################################
# Bilinear interpolation
################################################################

f(x, y) = 1 + x^2 + y^2 

grid_coarse = collect(0.0:1.0:5.0)
grid_fine = collect(0.0:0.01:5.0)
ncoarse, nfine = length(grid_coarse), length(grid_fine)
z_fine = zeros(nfine, nfine)

for i in eachindex(grid_fine), j in eachindex(grid_fine)
    x, y = grid_fine[i], grid_fine[j]
    z_fine[i,j] = f(x, y)
end

contourf(grid_fine, grid_fine, z_fine)

#now get to bilinear interp
z_coarse = zeros(ncoarse, ncoarse)
for i = 1:ncoarse, j = 1:ncoarse
    x, y = grid_coarse[i], grid_coarse[j]
    z_coarse[i,j] = f(x, y)
end

#create the interpolation
interp_z = linear_interpolation((grid_coarse, grid_coarse), z_coarse) 
z_interp = zeros(nfine, nfine)
for i = eachindex(grid_fine), j = eachindex(grid_fine)
    z_interp[i,j] = interp_z(grid_fine[i], grid_fine[j])
end

#compare contour plots
contourf(grid_fine, grid_fine, z_interp) #slightly more jagged, but otherwise pretty good!
contourf(grid_fine, grid_fine, (z_interp - z_fine)) #Difference
contourf(grid_fine, grid_fine, (z_interp - z_fine)./z_fine) #Percent Difference
surface(grid_fine,grid_fine,z_fine)
surface!(grid_fine,grid_fine, z_interp)








################################################################################
# Correlated normal shocks via the Cholesky decomposition
################################################################################

using Distributions, LinearAlgebra, Random

# Model has two correlated normal shocks with covariance matrix Σ.
Σ = [1.0 0.6;
     0.6 2.0]
L = cholesky(Σ).L        # Σ = L * L'

# Transform iid standard normals into correlated shocks: X = L * Z
Random.seed!(1234)
Z = randn(2, 100_000)
X = L * Z

cov(X')                  # ~Σ. It works!

# Why economists like this: the entries of L enter *linearly* and are
# unconstrained, so in estimation you can search over L freely and recover
# Σ = L*L' -- positive semi-definiteness guaranteed by construction.
# The same trick applies to quadrature: build a product rule for (Z₁, Z₂),
# then transform the nodes by L.
