include("rel_error_std_deviation.jl")

#This is the script for visualizing data, where the error standard deviation is plotted over varying parameters

L = 10
γ = 0.1
steps_ahead = 1
stock = "AMZN"
start_date = "2018-01-01"
end_date = "2021-01-01"
sample_frequency = "1d"

std_deviations_vector = Vector{Float64}()
# for i in L
#     #push!(std_deviations_vector, std_deviation(i, γ, steps_ahead, stock, start_date, end_date, sample_frequency))
#     println(std_deviation(i, γ, steps_ahead, stock, start_date, end_date, sample_frequency))
# end

println(std_deviation(L, γ, steps_ahead, stock, start_date, end_date, sample_frequency))
