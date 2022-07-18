include("Data.jl")  
include("Plotting.jl")
include("Behavioural_AM.jl")
include("Errors_AM.jl")
include("Buy_sell.jl")


L = 10
γ = 0.1
num_preds = 1

all_data = get_data("AMZN", "2018-01-01", "2021-01-01", "", "1d")

adj_close_new = all_data[:, 6]'
vol_new = all_data[:, 7]'

vol_train_new, vol_test_new = split_train_test(vol_new, "column")
adj_close_train_new , adj_close_test_new = split_train_test(adj_close_new, "column")


adj_close_standard_new = standardize(adj_close_new, adj_close_train_new)
vol_standard_new = standardize(vol_new, vol_train_new)

data_standard_new = [adj_close_standard_new ; vol_standard_new]
train_standard_new, test_standard_new = split_train_test(data_standard_new, "column")


# do predictions for all
predictions_new = behavioural_prediction_new(data_standard_new, train_standard_new, test_standard_new, L, num_preds, γ)
pred_rescaled_new = rescale_new(predictions_new, adj_close_train_new)

# split predictions in half 
clean_new = collect(skipmissing(pred_rescaled_new))
