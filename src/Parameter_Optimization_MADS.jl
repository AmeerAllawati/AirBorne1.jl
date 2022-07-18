using DirectSearch
include("std_deviation.jl")
include("Data.jl")

ticker_symbol = "AMZN"
start_date = "2018-01-01"
end_date = "2021-01-01"
seperation = "1d"
all_data = get_data(ticker_symbol, start_date, end_date, "", seperation)

system_dimension = 2 #How many signals are added to the Hankel matrix

L_max = floor((size(all_data,1)+1)/(system_dimension+1)) #This is the maximum possible depth according to the BC theory. Added as a constraint
p = DSProblem(2; objective=BC_standard_deviation, initial_point=[14, 0.01])
# i = 1 #Index of variabels that are granular
# SetGranularity(p, i, 1.0)
gamma_granularity = 0.01
SetGranularity(p, Dict( 1 => 1, 2 => gamma_granularity ))
#cons(x) = [-x[1] -x[2]] #Constraints x[1] to be greater than or equal to 0
#Constraints on Length (depth) of the Hankel matrix
consL(x) = x[1] > 1
AddExtremeConstraint(p, consL)

consL2(x) = x[1] < L_max
AddExtremeConstraint(p, consL2)

#Constraints on gamma
consgamma(x) = x[2] > 0
AddExtremeConstraint(p, consgamma)

Optimize!(p)