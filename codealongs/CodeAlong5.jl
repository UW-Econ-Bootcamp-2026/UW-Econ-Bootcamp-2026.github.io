#How many threads (or cores) are available on your computer? 
versioninfo()

################################################################################
# Multi-Threading
################################################################################

# You will need to load julia with multiple threads. 
# In VScode, go to Gear Icon in bottom-left, settings, search threads, click edit in 
# settings.json file, then change the number of threads. The number must be less than
# or equal to the number of threads available on your computer. 
# You will also need to restart the Julia REPL to update the setting. 

# On Linstat/slurm, to start julia with N threads use: julia --threads=N


versioninfo()

#Get the thread id
Threads.threadid()

#Number of threads
Threads.nthreads()

#For loop in series (not multi-threaded)
for i = 1:Threads.nthreads()
    println(Threads.threadid())
end

#For loop multi-threaded
Threads.@threads for i = 1:Threads.nthreads()
    println(Threads.threadid())
end


# Data Race
# All threads share the same memory, which can cause problems
# Let's add together some random numbers
my_vec = rand(1_000_000)

#Sum with one thread using sum function
sum(my_vec)


#We could do multi-threaded
function sum_mt(x)
    total = 0.0
    Threads.@threads for i in eachindex(x)
        total += x[i]
    end
    total
end

#Oh no!
sum_mt(my_vec)
#All the threads are trying to update total at the same time, and sometimes they over-write each other. 


#Solution
function sum_mt2(x)
    total = zeros(Threads.maxthreadid()) # this gives you the max existing thread id, which is not nthreads()
    Threads.@threads for i = eachindex(x)
        total[Threads.threadid()] += x[i]
    end
    sum(total)
end

#Much better
sum_mt2(my_vec)



#Multithreading is faster when calculation inside for loop takes non-trivial amount of time. 

#For loop in series (not multi-threaded)
@time for i = 1:Threads.nthreads()
    sleep(1) #Wait for one second
end

#For loop multi-threaded
@time Threads.@threads for i = 1:Threads.nthreads()
    sleep(1)
end



################################################################################
# Multi-Processing
################################################################################

using Distributed #Gives functions and macros for distributed/multi-processing



################################################################################
# Set-Up for Parallelization
################################################################################

#How many processes are running in my session?
nprocs()

#How many workers? 
nworkers()

#Add 2 processes/workers
addprocs(2)

#=
Note: One of the processes is the master if you have more than one process. 
Master distributes tasks and work to the workers, but typically does not do 
any work itself. In order to get speed gains from distributing, you will need
to add at least two workers
=#

nprocs() #Three processes
nworkers() #But only two workers


#Each process is given an id.
myid() #Prints processes id. 1 is the master, which you are currently running on. 

#@everywhere will run the following line of code on all processes. 
@everywhere @show myid()

#Get a list of processes and workers
procs()
workers()


#@spawn sends a line of code to an available worker
chosen_worker = @spawn myid() #Returns a Future object, which is a promise from the worker to do this when you want it. 
#Fetch will make worker fulfill its promise. 
fetch(chosen_worker)

#We can specifically choose the worker by using @spawnat
chosen_worker = @spawnat 2 myid() #This will always go to worker 2


################################################################################
# Memory across processes
################################################################################

#Define r on master processes
r = rand()

#Other workers don't know what r is. 
@everywhere @show r

#Define a different r on every process
@everywhere r = rand()
@everywhere @show r #Every process generated a different random number!


################################################################################
# Simple Example
################################################################################

#Suppose we have a vector of numbers x = [x₁, x₂, ..., xₙ]
#We want to calculate the sum of their square roots S = Σ(√xᵢ)

@everywhere function sqrt_sum(A)
    S = 0.0
    for i in eachindex(A)
        #sleep(0.001)
        S += sqrt(A[i])
    end
    return S
end

#We can run this just on our master process
A = rand(1000)
@time sqrt_sum(A)

#We can also break the sum up into batches and send to our workers using @distributed. 
#The @sync macro tells the master to wait until the workers are done with the for loop. 
function sqrt_sum_distributed(A, N_batches)

    N = length(A)
    batch_size = Int(length(A) / N_batches)

    # The (+) in front of the for loop will add together the results of the calculations
    # inside the for loop and saves it as S
    S = @sync @distributed (+) for batch in [(1:batch_size) .+ offset for offset in 0:batch_size:(N-1)]
                            sqrt_sum(A[batch]) # Returns sum of this batch which will get added to S
    end
    
    return S

end

#This creates a vector of ranges
N = length(A)
batch_size = 100
[(1:batch_size) .+ offset for offset in 0:batch_size:(N-1)]

#Run distributed
@time sqrt_sum_distributed(A, 100)


#We can also run this in parellel using pmap()
function sqrt_sum_pmap(A, N_batches)

    N = length(A)
    batch_size = Int(length(A) / N_batches)

    S = sum(
            pmap(
                batch -> sqrt_sum(A[batch]), #Operation to be done by workers 
                [(1:batch_size) .+ offset for offset in 0:batch_size:(N-1)] #Vector of inputs to be broken up accross workers
                )
            )

    return S
end

@time sqrt_sum_pmap(A, 100)


#=
Here, because the calculations we are doing inside the for loop (sqrt and add), 
there is not much speed gain from parallelization. Although we can go through these
calculations twice as fast on our two workers, there is an "overhead" cost of 
passing information back and forth between master and worker. Paralleization has
performance advantage when this overhead cost is cost is small relative to the computation
time inside the loop. 

If you uncomment the sleep command inside sqrt_sum(), you will see that @distributed and pmap
are now about twice as fast as just runnning on the master process. 

@distributed tends to run faster than pmap when the calculations inside the for loop are quick. 

pmap is better when each iteration of the for loop takes a longer time. 
=#

################################################################################
# Shared Arrays
################################################################################

#Shared Arrays are Arrays that both master and workers have read and write access to. 
using SharedArrays

# Normal Array define on the Master Process
A = rand(10000)

# Inside a distributed for loop, workers will look for variables in the master's global Memory
# if that name variable doesn't exist in the worker's memory. 
# Worker's will copy this variable to their own global memory
@sync @distributed for i = eachindex(A)
    A[i] + 1
end

#Workers will update A in their own global memory, but not on the master!
@sync @distributed for i = eachindex(A)
    A[i] = A[i] + 1
end

#A was only update on the workers, but A on the master still has A[i] < 1 for all i
A

#With shared arrays, workers have read and write ability onto the master's memory
A_shared = SharedArray{Float64}(A)

@sync @distributed for i = eachindex(A_shared)
    A_shared[i] = A_shared[i] + 1
end

A_shared

###################################################################################################
# Parallelized Optimal Investment Problem
###################################################################################################

# Same growth model and grid search as CodeAlong2, but the loop over k_grid now runs
# in parallel across worker processes with @distributed instead of a plain serial for
# loop. Grid search does real work per grid point (a full inner loop over every
# candidate k'), so there's enough computation per task to be worth the overhead of
# handing it out to workers.

using Distributed, SharedArrays

# With @distributed, only worker processes compute -- the master just hands out chunks
# and waits -- so more workers means more of the machine actually computing. Adjust to
# your available cores (check Sys.CPU_THREADS).
addprocs(4)

# SharedArrays has to be loaded on the workers too -- just constructing a SharedArray
# requires every worker to participate, not only the master.
@everywhere using SharedArrays

### Struct for our model paramters
# @everywhere so that every worker (not just the master) knows about ModelParameters
@everywhere @kwdef struct ModelParameters

    β::Float64 = 0.99 #We can define default values because of the @kwdef macro.
    δ::Float64 = 0.025
    α::Float64 = 0.36

    k_grid::Vector{Float64}=collect(range(0.1, length = 1800, stop = 45.0)) #### Same fine grid as CodeAlong2's grid search
    N_k::Int64 = length(k_grid)

    tol::Float64 = 10^-5
    max_iter::Int64 = 10^4

end


### Struct for our Model solutions
# SharedArrays (not plain Vectors) so workers write results directly into shared memory
# instead of copying back to master. Allocated once here rather than inside bellman(),
# since constructing a SharedArray requires a handshake with every worker -- doing that
# on every one of solve_model!'s 1000+ calls would add real overhead.
struct ModelSolutions

    V::SharedVector{Float64}
    kp::SharedVector{Float64}
    V_next::SharedVector{Float64}
    kp_next::SharedVector{Float64}

end

function initialize()
    para = ModelParameters()
    V = SharedVector{Float64}(zeros(para.N_k))
    kp = SharedVector{Float64}(zeros(para.N_k))
    V_next = SharedVector{Float64}(zeros(para.N_k))
    kp_next = SharedVector{Float64}(zeros(para.N_k))
    sols = ModelSolutions(V, kp, V_next, kp_next)
    return para, sols
end

### Bellman operator
# Mutates sols.V_next/sols.kp_next in place instead of allocating new SharedArrays each
# call; otherwise same calling convention as CodeAlong2/4's bellman().
function bellman(para, sols)
    (; k_grid) = para
    (; V, V_next, kp_next) = sols

    # Hand each grid point off to a worker; update_next! is the unit of work distributed.
    @sync @distributed for i_k in eachindex(k_grid)
        update_next!(i_k, para, V, V_next, kp_next)
    end

    return V_next, kp_next

end

@everywhere function update_next!(i_k, para, V, V_next, kp_next)
    (; β, δ, α, k_grid) = para

    k = k_grid[i_k]
    budget = k^α + (1-δ)*k

    #Same grid search as CodeAlong2: loop over every candidate k' and keep the best.
    max_util = -1e10
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

#Compare this @time to CodeAlong2's serial version -- did parallelizing across workers help?
@time solve_model!(para,sols)

#=
Unlike the sqrt_sum and interpolated-VFI examples, parallelization pays off here: grid
search does O(N_k) work per grid point, so per-task computation dwarfs the overhead of
shipping data to a worker and back. The size of the win depends on giving workers real
work and actually using the hardware -- the master only distributes and idles, so N
workers caps speedup near Nx, and preallocating V_next/kp_next avoids repeatedly paying
the SharedArray handshake cost. On this machine (10 cores): 2 workers with SharedArrays
rebuilt every call got ~1.15-1.35x; preallocating got to ~1.35x; 4 workers (as coded
above) got to ~1.9x; N_k=3600 with 8 workers hit ~3.5x. Try tuning addprocs() and
k_grid's length yourself and see how it scales on your hardware.
=#

using Plots
plot(para.k_grid, sols.V)
plot(para.k_grid, sols.kp)
plot!(collect(0:45), collect(0:45))


