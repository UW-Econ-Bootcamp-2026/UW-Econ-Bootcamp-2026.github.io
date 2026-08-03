

################################################################################
# Packages
################################################################################

using Optim, Interpolations, Plots, Polynomials



################################################################################
# Runge's function and phenomenon
################################################################################

Runge(x) = 1 /(1 + 25*x^2)
function Runge_2(x)
    val = 1/(1+25*x^2)
    return val
end
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
plot(x_fine, [y_fine, y_fine_interp])
scatter!(x, y, markersize = 4)

x2=[0.1,0.5,1.0,1.5,2.0,5.0,10.1]
y2 = log.(x2)
interp_y2 = linear_interpolation(x2,y2)
y_interp = interp_y2.(x_fine)
plot(x_fine, [y_fine, interp_y2.(x_fine)])
##Extrapolation
log(11)
interp_y(11) #This will give you an error because we haven't allowed for extrapolation

#Need to set the extrapolation boundary coundition
#Line() will just extend the line from the last region
#Flat() will set everything equal to the last gridpoint. 
interp_y_extra = linear_interpolation(x,y,extrapolation_bc = Line()) #Flat()
interp_y_extra(11)

x2=[0.1,0.5,1.0,1.5,2.0,5.0,10.1,19.0]
y2 = log.(x2)
interp_y2 = linear_interpolation(x2,y2)
y_interp = interp_y2.(x_fine)
x_fine_2  = collect(0.1:0.1:20.1)
y_fine_2 = log.(x_fine_2)
plot(x_fine_2, y_fine_2)
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
interp_y = cubic_spline_interpolation(x, y)
y_fine_interp = interp_y.(x_fine)
plot(x_fine, [y_fine, y_fine_interp])
scatter!(x, y, markersize = 4)

###Extrapolation 
interp_y = cubic_spline_interpolation(x, y, extrapolation_bc = Line())
log(12)
interp_y(12)

#cubic_spline_interpolation does not support uneven grid in Interpolations.jl
x_uneven = 0.1 .+ 10.0 .*collect(0.0:0.2:1.0).^2
y_uneven = log.(x_uneven)
interp_y = cubic_spline_interpolation(x_uneven, y_uneven) #Error because x is not evenly spaced. 


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
I = sum(w .* f.(x))
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
sum(w.*g_tilde.(x,f))

#Monte Carlo Simulation Approximation
val = 0
N = 1000000
for i = 1:N
    val += f(rand(Normal()))/N
end
val






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
