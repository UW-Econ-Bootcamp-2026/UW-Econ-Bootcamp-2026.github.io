
###################################################################################################
# Machine code
###################################################################################################

# Julia is the translator between you and your computer. You learn the Julia programming language, 
# and Julia converts it into machine language that your computer understands. If you ever want to 
# see what Julia is say to your computer, you can use the @code_native macro. 

@code_native 2+2

# That garbled nonsense that appears in the terminal is machine language. Not very useful to us.



###################################################################################################
# Local Scope vs. Global Scope
###################################################################################################

# The scope of a variable is where in the code the region is accessible. 

# Global scope are essentially anything that are defined outside of 
# functions, for loops, conditionals, etc. 

x = 2 #Here, x is a global variable. 

# We can operate on x in the global scope (i.e. directly from the command line/REPL). 
x + 4

# When you make a function, you define a new local scope.
function f()
    # Things defined inside this function are only accessible within our function. 
    z = 3
    return z
end

f()

# Inside the local scope of the function f, we defined the variable 'z'. 
# But we can't access 'z' outside of the function in the global scope
z 
# You get an error if you try to use z in global scope because it only exists in the local scope of 
# the function f. 


#For loops also define a local scope. 
for i in 1:10
    y = 7
    @show y + i 
end

#y is not defined in global scope. Only in the local scope of the above for loop.
y

# If we define x in global scope, then it will be updated when we operated on it in local scope. 
x

for i in 1:10
    x = 7
    @show x + i 
end

# x is now 7
x


### functions with multiple outputs

# if we want a function to return multiple objects (e.g. the market price and quantity), we can do that!

function find_eq()
    quantity = 10.0
    price = 2.0
    return quantity, price
end

#we can do this a few different ways; the first is to assign each output its own variable:
eq_q, eq_p = find_eq()
println("Equilibrium quantity is: ", eq_q)
println("Equilibrium price is: ", eq_p)


#or, you can define one variable to hold all the function outputs and then access each element using its index: 
eq = find_eq()

#you can see that eq returns (10.0, 2.0) - eq is a `tuple` of values. Intuitively, it is kind of like a vector (but treated differently by Julia). 
print(eq)

println("Equilibrium quantity is: ", eq[1]) #quantity was the first item returned
println("Equilibrium price is: ", eq[2]) #price was the second item 

###################################################################################################
# Tip 1: Write things inside functions
###################################################################################################

# Julia is fast because it performs just-in-time compilation. This means when you run your code, 
# Julia will compile it and then run it. A piece of code only needs to be compiled once. This is why
# things in Julia will often be slow the first time you run them, but much faster the second time. 
# The first run includes both the compilation and the execution. The second time, you are only
# executing the code. 



###Example 1: Very bad: Everything here is defined in global. 
# We are not taking advantage of compilation at all here. 

m = 3
count = 0

@time for i = 1:10^7
    count += m*i
    count -= m*i
end


###Example 2: A little better

function update(x,m,i)
    x += m*i
    x -= m*i
    return x
end

@time for i = 1:10^7
    count = update(count,m,i)
end



###Example 3: Good, used a function. Everything happens in local scope
function g(m)

    count = 0.0

    for i = 1:10^7
        count = update(count, m, i)
    end

    return count

end


@time g(m)

#this is an improvement of two orders of magnitude - that's insane!! When you start doing more complex tasks (e.g. estimating/simulating a model a bunch of times), those improvements really add up. To put that in context: this is the difference between your code taking one day to run vs. taking 15 minutes - efficiency matters!


###################################################################################################
# Types and Structs
###################################################################################################

# Types describes the structure and behavior of an element. 
# We have seen lots of types already in the first  lecture

# Integer with 64 bits 
typeof(1)
#(for those curious: bits are the amount of computer memory needed to store an object; an integer with 64 bits has a minimum value of -9,223,372,036,854,775,808 and a maximum value of 9,223,372,036,854,775,807. 
#Conversely, a 32 bit integer has a min value of -2,147,483,648 and a max of 2,147,483,647. )
#There's a tradeoff: using Int64 allows for significantly more possible values, but takes up double the space. This can lead to slower performance/more memory consumption. 
#that being said, you should probably always use 64 bit Integers/Floating numbers (there's a good reason why this is the default)

# Floating number with 64 bits
typeof(1.0)

x = 4
y = 5
z = x/y
x=4.0
y=5.0
z=x/y
# String 
typeof("abc")
# Character
typeof('a')
# Vector of Floats, which is a 1-D array
typeof(zeros(2))
# Matrix of Floats, which is a 2-D Array
typeof(ones(3,2))
# 4-D Array of integers
typeof(zeros(Int64, 3,3,3,3))


#You can define the type of a variable using "::" when you call it. 
x = 32.0::Float64 
typeof(x)
#This is entirely unneccesary, though, because Julia will figure out your types for you. 
y = 32.0
typeof(y)

# You can also specify the type of your function arguments.
function h(x::Int64)
    return x * 100
end



h(10) #Your function works with integers
h(10.0) #But it will not work with floats. 

# You can avoid this by allowing julia to infer the type, but this can make your code run slower 

#=it is generally good practice for you specify the type in the function argument.
The short reason is that if you don't use type annotations, it takes time for julia to figure out the type. This can slow down your code.
Also, it can help you catch bugs/errors in the way you've written your function or in the inputs you're passing to the function. Requiring things to be of the correct type will help you better diagnose issues with your code.
There are some more subtle reasons too, but I won't dwell on those for now.
=#


function h2(x)
    return x * 100
end

#Now this will work with both types. 
h2(10)
h2(10.0)

# You can define your own types using the struct ("structure") keyword
# This type has two fields, a and b. 
struct MyType
    a 
    b
end

#We can define an instance of MyType
example = MyType(25.0, "abc")

#We can access the fields of example like this
example.a
example.b

# Structs are very very useful when coding up big models with many parameters or output. 
# Instead of carrying around all these different things, you just need to carry around your struct 
# container.

###################################################################################################
# Tip 2: Declare concrete types inside your structs. 
###################################################################################################

# Let's define a struct here without declaring the type of the element. 
# Here, "a" could be anthying. A float, integer, string, character, matrix, or anything at all.  
struct Container_Any
    a
end

# Here, we specified that "a" must be a float. If we don't make assign a float to "a", we will get
# an error. 
struct Container_Float
    a::Float64
end


#Let's assign a float to both anyways. 
c_any = Container_Any(1.0)
c_float = Container_Float(1.0)
typeof(c_any)
typeof(c_float)
typeof(c_any.a)
typeof(c_float.a)

#Let's define a function that operates on our containers. 
function f(x)
    tmp = x.a
    #A bunch of calculations
    for i in 1:10^7
        tmp = i + x.a
    end
    return tmp
end

#Let's see how fast our function is
@time f(c_any)
@time f(c_float)

#But now run them a second time
@time f(c_any)
@time f(c_float)
# Now f is much faster on c2 than on c1
# This is because f has now been compiled. Because the type is declared for element within c2, 
# Julia is able to write very efficient machine code to execute the program instantly. 
# While for c1, it doesn't know the type will always be float, and has to do many useless calculations. 

## another remark: note the huge difference in memory allocations! For f(c_float), Julia needed to allocate 16 bytes in a total of one allocation (very small, basically impossible to make it more efficient than that). For f(c_any), Julia needed to allocate 305.17 ~~Megabytes~~ of memory (305 million bytes!), and makes 19,999,489 total allocations!! These differences are huge and can make a dramatic difference in the speed of your code. 


#note: when timing things precisely, I like to use the package BenchmarkTools.
#This package essentially runs your code a bunch of times and reports the minimum amount of time it took to run (as well as other statistics)

#for example, here's the comparison of f(c_any) vs f(c_float) using @benchmark:
using BenchmarkTools
@benchmark f(c_any)
@benchmark f(c_float)

#isn't that cool??

#you can also use @btime if you just want the number and not all the distribution stuff
@btime f(c_any)
@btime f(c_float)

###################################################################################################
# Optimal Investment Problem
###################################################################################################

# We use Julia's built-in @kwdef macro to define default values in our structs.
# (Older code often uses the Parameters.jl package and its @with_kw macro for this;
#  @kwdef is built in and does the same job.)

### Struct for our model paramters
@kwdef struct ModelParameters

    β::Float64 = 0.99 #We can define default values because of the @kwdef macro. 
    δ::Float64 = 0.025
    α::Float64 = 0.36

    k_grid::Vector{Float64}=collect(range(0.1, length = 1800, stop = 45.0))   
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

    V = zeros(Float64,para.N_k)
    kp = zeros(Float64,para.N_k)

    sols = ModelSolutions(V, kp)

    return para, sols
end

### Bellman operator
function bellman(para, sols)
    # destructure the fields we need (built-in Julia syntax; replaces Parameters.jl's @unpack)
    # unpack all: (; β, δ, α, k_grid, N_k, tol, max_iter) = para

    (; β, δ, α, k_grid, N_k) = para # just unpack the fields we need
    (; V) = sols

    #note: when creating new arrays of zeros, it is good to declare the type! This way Julia doesn't have to figure it out later
    V_next = zeros(Float64, N_k) 
    kp_next = zeros_like(kp) # better: let Julia do it

    for (i_k, k) in enumerate(k_grid)
        max_util = -1e10
        k = k_grid[i_k]
        budget = k^α + (1-δ)*k
        for (i_kp, kp) in enumerate(k_grid)
            
            c = budget - kp

            if c > 0 # check feasibility
                
                V_temp = log(c) + β*V[i_kp]
            
                if V_temp > max_util
                    max_util = V_temp
                    kp_next[i_k] = k_grid[i_kp]
                end
                
            end

        end
        V_next[i_k] = max_util
    end

    return V_next, kp_next

end



### Solve model
function solve_model(para, sols)
    # destructure the fields we need (built-in Julia syntax; replaces Parameters.jl's @unpack)
    (; tol, max_iter) = para
    (; V) = sols

    V_next = zeros_like(V)
    max_diff = tol * 100.0
    n = 0
    while max_diff > tol
        n +=1
        V_next, kp_next = bellman(para,sols)
        
        max_diff = maximum(abs.(V_next - V))
        sols.V .= V_next
        sols.kp .= kp_next

        @show n, max_diff

    end

    sols

end

para, sols = initialize();

@time sols = solve_model(para,sols)
 

using Plots
plot(para.k_grid, sols.V)
plot(para.k_grid, sols.kp)
plot!(collect(0:45), collect(0:45))




#### optional

#the order in which you loop over arrays matters! Consider the following:

function loop_1()
    n_r=3000
    n_c=3000
    A = zeros(Float64,n_r,n_c)
    count = zeros(Float64, n_r,n_c)
    for i=1:n_r, j=1:n_c
        count[i,j] = 0.0
    end
    return count
end

function loop_2()
    n_r=3000
    n_c=3000
    A = zeros(Float64,n_r,n_c)
    count = zeros(Float64, n_r,n_c)
    for j=1:n_c, i=1:n_r
        count[i,j] = 0.0
    end
    return count
end

using BenchmarkTools

@benchmark loop_1()
@benchmark loop_2()

#basically: we did the exact same operation (created a square matrix with all zero entries). One way, we filled in the rows first. The second way, we filled in the columns first. These get us the exact same result, but the second one gives us the result in basically ~half the time~! This shows how subtle things (like the order in which you do a nested loop) can have a big impact on the results you get.

#however, it can be sometimes confusing to find the right way to iterate through a multi-dimensional array (especially if you have 3 or more dimensions). In order to always load them in the most efficient way, use eachindex! It will always load the elements in the most efficient order.
function loop_each_index()
    n_r=3000
    n_c=3000
    A = zeros(Float64,n_r,n_c)
    count = zeros(Float64, n_r,n_c)
    for ind in eachindex(count)
        count[ind] = 0.0
    end
    return count
end
loop_each_index()
@benchmark loop_each_index()
#


function copy_col_row(x::Vector{Float64})
    n = length(x)
    out = zeros(Float64, n, n)
    for col = 1:n, row = 1:n
        out[row, col] = x[row]
    end
    return out
end

@elapsed copy_col_row(rand(10000))

function copy_row_col(x::Vector{Float64})
    n = length(x)
    out = zeros(Float64, n, n)
    for row = 1:n, col = 1:n
        out[row, col] = x[col]
    end
    return out
end

@elapsed copy_row_col(rand(10000))



### use views!!

function no_views()
    n_r = 1000
    n_c = 1000
    x = zeros(Float64,n_r, n_c)
    output = zeros(Float64,n_r, n_c)
    for i=1:n_r
        output[i,:] = x[i,:] .+ i.*50.0
    end
    return output
end


function with_views()
    n_r = 1000
    n_c = 1000
    x = zeros(Float64,n_r, n_c)
    output = zeros(Float64, n_r, n_c)
    for i=1:n_r
        view(output,i,:) .= view(x,i,:) .+ i.*50.0
    end
    return output
end


a = no_views()
b = with_views()
@elapsed no_views()
@elapsed with_views()

@benchmark no_views()
#minimum of 5.5 milliseconds, median of 8.12 milliseconds
@benchmark with_views()
#minimum of 2.636 milliseconds, median of 4.25 milliseconds!

#this is literally the same operation, same result, but in almost half of the time!! Also, much less memory being allocated. 
#If you are comfortable using them, views can sometimes be another useful huge way to speed up your code
#One big warning: you should be careful, since views can be a bit tricky to use and you should make sure you understand how to use them. 

#one way for you to have your cake and eat it too: use the @views macro! It allows you to use array slices that you're likely more comfortable with yet still get the performance benefits of using views!

#sorry this function name is trash
function no_views_with_a_view()
    n_r = 1000
    n_c = 1000
    x = zeros(Float64,n_r, n_c)
    output = zeros(Float64,n_r, n_c)
    for i=1:n_r
        #only line that changes - we added the macro at the beginning
        @views output[i,:] = x[i,:] .+ i.*50.0
    end
    return output
end

@benchmark no_views_with_a_view()
#whoa! That's ~basically as fast as the fast one! I would recommend doing this if you don't want to bother too much with rewriting your code for maximum efficiency


#now I know you may be wondering: what would happen if we iterated over the columns first and fill in the rows? Wouldn't that be faster, as we saw before? That's a great question. Rewriting the view function but looping over the columns, we see another big improvement in speed. Crazy!
function with_views_columns()
    n_r = 1000
    n_c = 1000
    x = zeros(Float64,n_r, n_c)
    output = zeros(Float64, n_r, n_c)
    for i=1:n_c
        view(output,:,i) .= view(x,:,i) .+ i.*50.0
    end
    #since we looped over the columns, we need to transpose to get the same output as the other functions
    output = transpose(output)
    return output
end

@benchmark with_views_columns()
#minimum of 1.473 milliseconds, median of 1.963 milliseconds!
#again, this shows that looping over columns vs rows can make a big difference in terms of speed!


###################################################################################################
# Dot fusion: vectorized operations without temporary arrays
###################################################################################################

x, y, w = rand(10^6), rand(10^6), rand(10^6)

# If you mix dotted and un-dotted operators, Julia computes the pieces separately
# and allocates temporary arrays for each piece:
f_mixed(x, y, w) = x .+ y - w ./ 4    # temporaries for (x .+ y) and (w ./ 4)

# If every operator is dotted, Julia "fuses" everything into one loop, no temporaries:
f_fused(x, y, w) = x .+ y .- w ./ 4

using BenchmarkTools
@btime f_mixed($x, $y, $w)
@btime f_fused($x, $y, $w)

# Even better: write the result into a pre-allocated array with .=
z = similar(x)
@btime $z .= $x .+ $y .- $w ./ 4      # zero allocations!

# The @. macro dots everything in the expression for you:
@btime @. $z = $x + $y - $w / 4


###################################################################################################
# Diagnosing type instability: @code_warntype
###################################################################################################

# A function is "type stable" if Julia can infer one concrete type for every variable.
function unstable(flag)
    x = 1            # starts as Int
    if flag
        x = 2.5      # ...might become Float64. Julia must plan for both -> slow
    end
    return x * 2
end

function stable(flag)
    x = 1.0
    if flag
        x = 2.5
    end
    return x * 2
end

# Run these in the REPL and look at the output:
# red "Any" or "Union{...}" entries mean Julia couldn't pin down a type.
@code_warntype unstable(true)
@code_warntype stable(true)
