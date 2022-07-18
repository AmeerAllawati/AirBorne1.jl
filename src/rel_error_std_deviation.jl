include("Data.jl")  
include("Plotting.jl")
include("Behavioural_AM.jl")
include("Errors_AM.jl")

#This is the function that would be minimized

#This function predicts prices on a moving window, calculates the errors and then outputs the standard deviation

#Test data in this case referes to the amount of data used to calculate the standard deviation
#Test data is considered to be 1/3 of the sample size
#The amount of data to be used for training is a PARAMETER that would be optimized
function std_deviation(L, gamma, ahead_predictions, stocks, start_date, end_date, sample_frequency)
    #This gets the data of a stock given some parameters.
    all_data = get_data(stocks, start_date, end_date, "", sample_frequency)

    #Extract the close prices
    adj_close_new = all_data[:, 6]'

    #The following lines extracts train and test data, standardize the data then finally 
    #extracts the train and test data based on standardized data.

    #Test data in this case represent how much data is used to calculate the standard deviation of the error
    adj_close_train_new, adj_close_test_new = split_train_test(adj_close_new, "column")

    #standardize the data to be able to use the lasso function.
    adj_close_standard_new = standardize(adj_close_new, adj_close_train_new)

    data_standard_new = adj_close_standard_new #data_standard_new = [adj_close_standard_new]
    train_standard_new, test_standard_new = split_train_test(data_standard_new, "column")

    # do predictions for all and rescale them
    predictions_new = behavioural_prediction_new(data_standard_new, train_standard_new, test_standard_new, L, ahead_predictions, gamma)
    pred_rescaled_new = rescale_new(predictions_new, adj_close_train_new)

    # skip the missing entries because predictions has missing entries 
    clean_new = collect(skipmissing(pred_rescaled_new))

    println(clean_new)
    println(adj_close_test_new)
    # calculate the relative errors and output the standard deviation of the relative errors
    rel_matx = get_error_matrix(clean_new[1:size(adj_close_test_new,2),:], adj_close_test_new')
    println(rel_matx)
    return std(rel_matx)

end