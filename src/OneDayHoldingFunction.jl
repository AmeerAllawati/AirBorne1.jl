include("Data.jl")
include("Behavioural.jl")
include("Errors.jl")

function HoldingFunction(decisions::Vector{Float64}, L)
    decision = decisions[1]
    println("The Depth in this iteration is $L")
    γ = 0.01
    num_preds = 1

    all_data = get_data("AMZN", "2018-01-01", "2021-01-01", "", "1d")
    adj_close = all_data[:, 6]
    vol = all_data[:, 7]

    vol_train, vol_test = split_train_test(vol)
    adj_close_train, adj_close_test = split_train_test(adj_close)

    adj_close_standard = standardize(adj_close, adj_close_train)
    vol_standard = standardize(vol, vol_train)

    data_standard = [adj_close_standard vol_standard]
    train_standard, test_standard = split_train_test(data_standard)

    # do predictions for all
    predictions = behavioural_prediction(data_standard, train_standard, test_standard, L, num_preds, γ)
    pred_rescaled = rescale(predictions, adj_close_train)
    # split adj close test values in half 
    adj_close_test_1 = adj_close_test[1:floor(Int, size(adj_close_test, 1) / 2)]
    adj_close_test_2 = adj_close_test[floor(Int, size(adj_close_test, 1) / 2)+1:size(adj_close_test, 1)]

    # split predictions in half 
    clean = collect(skipmissing(pred_rescaled))
    pred_rescaled_1 = clean[1:floor(Int, size(clean, 1) / 2)]
    pred_rescaled_2 = clean[floor(Int, size(clean, 1) / 2)+1:size(clean, 1)]

    price_today = adj_close_train[end] #Today's actual price
    price_tomorrow = clean[1] #Tomorrow's expected price, the index could also be size(adj_close_train,1)+1
    println("Today's price: $price_today")
    println("Expected price tomorrow $price_tomorrow" )
    Hold = -(1)*decision*(price_tomorrow-(price_today+0.01*price_today))
    return Hold
end