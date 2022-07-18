using DirectSearch
include("OneDayHoldingFunction.jl")
include("Data.jl")

p = DSProblem(1; objective=HoldingFunction, initial_point=[1])
SetGranularity(p, Dict( 1 => 1))
#SetGranularity(p, 1, 2)

cons(x) = x[1] >= -1
AddExtremeConstraint(p, cons)

cons2(x) = x[1] <= 1
AddExtremeConstraint(p, cons2)


Optimize!(p)

print("\n\n")
print("What is your capital in dollars? \n\n")
capital = readline()
capital = parse(Float64, capital)
print("\n\n")

print("How many stocks do you hold? \n\n")
num_stocks_held = readline()
num_stocks_held = parse(Int64, num_stocks_held)
print("\n\n")

print("What is your minimum return? \n\n")
min_return = readline()
min_return = parse(Float64, min_return)
print("\n\n")

if (p.x < 0)
    print("You should sell today and buy tomorrow")
    
else
    print("You should buy today and sell tomorrow")
end

println(p.x)
println(p.x_cost)