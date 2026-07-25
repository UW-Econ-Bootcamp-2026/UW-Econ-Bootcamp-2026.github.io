
########################################################################
# 0. Commments
########################################################################

# This is a commment. Julia will not run this line of code. 
# You can use comments to talk about what your code is doing. 
# Comment are an important part of having readable code. 


#=
This
is
a 
multiline
comment.

=#



########################################################################
# 1. Math
########################################################################

#Julia is just a fancy calculator.

2+2

10 + (20 - 24^3) / 5000 * 123



########################################################################
# 2. Suppressing and Printing Output
########################################################################

#This prints to the terminal
2+2

#Semi-colon will suppress output
ans = 2+3;
ans

#The print function will write output into REPL
print(2+2);
print("Hello!")

# The @show macro will print a line of code and its output to terminal. 
@show 2+2



########################################################################
# 3. Variables
########################################################################

# We can save the results of our calculations as variables
x = 2^3
x

y = x + 5

#emojis!!
🐟 = 112
🤑 = 300
🐟 + 🤑
🍎 = (🐟 + 🤑) / 2
#type '\', then enter the desired emoji name

#you can also do greek letters:
β= 3
β
# δ β ϵ ε ζ

# There are special operations for updating variables
x = 8

#x = x + 1
x += 1 #This adds one to x
x += 10#this adds ten to x
# x = x - 1
x -= 1
#x = 2 * x 
x *= 2
#x = x / 10
x /= 10


########################################################################
# 4. Arrays, Vectors, and Matrices
########################################################################

# An array is a collection of elements. 

# A vector is a 1 dimensional array. 

v1 = rand(4) # Four random numbers between 0 and 1
v2 = collect(1:2:41)
v3 = collect(range(start = 1, stop = 10, length = 100))
#zero vector of length four:
v_zero = zeros(4)

#if you want to create your own (column) vector with specified values, just enter it like this:
v1 = [4.0, 0.3, 17.0, -0.2]

#note the comma between the entries
#the comma tells julia that the next entry should be in the next row 


#same, but for a row vector
v2 = [4.0 0.3 17.0 -0.2]
#note the lack of commas or any separator

# A matrix is 2-D array
A3 = rand(4,5)

#to create one of these manually, you can do it like this:
A4 = [1.0 2.0; 3.0 4.0]
#(this is my preferred way)

#OR you could do it like this: (needlessly confusing, in my opinion)
A4 = [[1.0, 3.0]  [2.0, 4.0]]

#OR you could even do this (please don't do this, even though it technically works it just feels wrong)
A5 = [1.0 2.0
3.0 4.0]

#the confusing thing for me to learn initially: knowing which way I should enter rows vs. columns - I would ~highly~ recommend printing the matrix you just created to the console to make sure that the matrix actually looks like what you want it to look like

# Julia has built-in matrix operations
x = rand(4)
Z = rand(4,4)

# Matrix multiplication
Z * x
x' * Z

#Transpose
Z'
#or
transpose(Z)

#Matrix inverse
inv(Z)


# A . broadcasts a function to all the elements of an array
z_log = log.(Z) #Take log of all the elements of Z

z_sqrt = sqrt.(Z)

########################################################################
# 5. Packages
########################################################################

# The right square bracket ] opens the package interface to add packages. To install a package, you will need to type the ] (which will then show the prompt 'pkg>' instead of 'julia>'. From there, you will want to type add 'package_name', where 'package_name' is the name of your package. Once installed, you can load the package by typing 'using package_name')

using Distributions, Plots



########################################################################
# 6. For Loops
########################################################################

# Construct data from a random walk x_{t+1} = x_{t} + ϵ where ϵ ~ N(0,1)

x_vec = zeros(10000)

x_vec[4]

x_vec[1] #start the process at zero
for i in 2:10000
    x_vec[i] = x_vec[i-1] + rand(Normal())
end

x_vec[4]

#by the way: the first time you plot something, it takes longer than subsequent times
plot(x_vec)



########################################################################
# 6. Functions
########################################################################

# You can write your own functions

function OLS(Y,X)
    β = inv(X'*X)*(X'*Y)
    return β
end


coefs = zeros(10000)

#running ols on 100 fake datasets of size 100
for i in 1:10000
    #generating a fake dataset of size 100

    #epsilon shocks of length 100
    ε_vec = rand(Normal(), 10000)
    #create a normal vector of length 100, which we call X
    X = rand(Normal(), 10000)
    #data generating process: assume Y = X + normal shock
    Y = X + ε_vec

    #compute the ols coefficient from this particular fake dataset and store it in the i-th value of the coefficient simulation vector
    coefs[i] = OLS(Y,X)
end

histogram(coefs)


########################################################################
# Booleans and Conditionals
########################################################################

# Booleans are true and false
true
false

#this definines x as 5 (one equal sign)
x = 5
#this checks if x is equal to five (two equal signs)
x == 5 

x = (x == 5)

2 == 3
2 ==2 
5 > 1
6 <= 7
10 >= 100
7 != 6


# Conditional statements

x = 9

if x > 5 
    print("Yes")
else #if x <= 5
    print("No")
end

x = rand(10)

#one way: using a for loop
indicator=zeros(length(x))
for i in eachindex(x)
    if x[i]>0.5
        indicator[i]=1
    else
        indicator[i]=0
    end
end

indicator_simple = x.>0.5



x = 0
while x < 100
    @show x += 1
end
@show x


#if you want to check if two conditions hold simultaneously, you can use the & operator!

#this will return true, since both 3 > 2 and 4 > 3 are true statements
if (3 > 2) & (4 > 3)
    print("True!")
else
    print("False!")
end

#however, this will return false; even though 3 > 2 is true, 3 > 4 is false.
#since one of the statements is false, it will return false. It would only return true if both of the conditions were true
if (3 > 2) & (3 > 4)
    print("True!")
else
    print("False!")
end

#if instead we want to check if either condition holds (i.e. at least one of the conditions is true), we can use the 'or' operator ||

#this will evaluate as true, since (3 > 2) is true. Even though (3 > 4) is false, || only checks that at least one of the conditions is true
if (3 > 2) || (3 > 4)
    print("True!")
else
    print("False!")
end

#this will evaluate as false, since both conditions are false:
if (3 < 2) || (3 > 4)
    print("True!")
else
    print("False!")
end


#if there are multiple cases you would like to check and handle separately, you can use the command elseif!
x = -6
if x < 0 
    #code for the cases when x < 0
    print("x is negative")
elseif x > 0
    #code for when x > 0
    print("x is positive")
else #the remaining possibility is that x = 0
    #code for x=0
    print("x = 0")
end

#when writing if/elseif/else statements, make sure you understand what cases your "else" block contains! It can be easy to get things mixed up/forget about some edge cases. 

#by the way: you can do multiple elseif blocks

if x == 0
    print("x = 0")
elseif x == 1
    print("x = 1")
elseif x == 2
    print("x = 2")
elseif x == 3
    print("x = 3")
else
    print("x is not 0, 1, 2, or 3")
end

########### printing to console ###########

#As you've seen above, the print command will print things to the console

#this prints the word "Hello!"
print("Hello!")

#Let's print hello twice and see what happens...
print("Hello!")
print("Hello!")

#oh no! We got "Hello!Hello!", which is not what we wanted. 
#If you want to print something on a new line, you can use the println command:
println("Hello!")
println("Hello!")

#as intended!


#if you want to print multiple things, you can do this using one print command:

#This will print "Hello friend!"
print("Hello ", "friend!")

#note that it is important to include spaces/whatever punctuation you intend; Julia will combine strings by smushing them together and won't automatically add spaces/punctuation for you:
print("Hello","friend!")

#you can alse print variables:
greeting = "Hello, "
student_name = "Bucky Badger"
print(greeting, student_name)

#this is useful if you want to print things that take variables as input

apple_count = 5
println("I have ", apple_count, " apples" )
apple_count += 1
println("I now have ", apple_count, " apples" )

#slightly more complicated example with a loop:

student_names = ["Student One", "Student Two", "Student Three"]
for name in student_names
    println("Hello, ", name, ". Welcome to the course!")
end



#### enumerate

#if you want to get an item and its corresponding index (i.e. position) in its array, you can use enumerate:

student_names = ["Chris Taber", "Dean Corbae", "JF Houde"]
for (index,name) in enumerate(student_names)
    println("Hi ",name,", you are student number ", index)
end

########################################################################
# 6. Version Control: Git and GitHub
########################################################################

# Git is a program that tracks the history of your code. GitHub is a website
# that hosts git repositories. You submit homework by pushing to your own
# private repository in the course GitHub organization.

# One-time setup (in a terminal, not the Julia REPL):
#   git config --global user.name "Your Name"
#   git config --global user.email "you@wisc.edu"

# The core workflow (terminal commands, or use VS Code's Source Control panel):
#   git clone <url>       -> copy a repository to your computer
#   git status            -> what has changed?
#   git add file.jl       -> stage changes for the next snapshot
#   git commit -m "msg"   -> take a snapshot with a descriptive message
#   git push              -> upload your snapshots to GitHub
#   git pull              -> download your collaborators' snapshots

# Try it now:
# 1. Accept the emailed GitHub invite to your hw1-<username> repository
# 2. Clone it: git clone <your-repo-url>
# 3. Create a file, then: git add . ; git commit -m "First commit" ; git push
# 4. Refresh the repository page on GitHub - your file is there

# Rules of thumb:
# - Commit early and often; small commits with informative messages
# - Don't commit large data files or output; list them in a .gitignore file
# - If you break everything, `git log` + `git checkout <commit> -- file.jl`
#   recovers any earlier version. This is the whole point!
