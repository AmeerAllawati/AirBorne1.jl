using LinearAlgebra
using Statistics
using Plots
using Random
using JuMP
using DelimitedFiles
using SCS
using Convex
using ShiftedArrays
using Flux
using MarketTechnicals
using TimeSeries
using Dates

import MathOptInterface

export hankel_vector
export form_w_given
export hankel_scalar
export Lasso
export find_w_given
export behavioural
export behavioural_new
export behavioural_prediction
export form_w_given_new

function hankel_vector(w_d::Array, L::Any) 
    L = convert(Int64, L)
    dim = size(w_d, 2) #Dimension (how many signals used)
    Tmin = (dim + 1)*L-1 #T_min=(m + 1)*L - 1
    T = size(w_d, 1) #Trajectory time length
    Lmax = (T+1)/(dim+1) #L_max = floor (T+1)/(q+1)

    if(T < Tmin) || (L > Lmax)
        println("Min length is $Tmin, length of matrix is $T")
        println("Max depth is $Lmax, depth selected is $L")
        return 
    else
        hank = Array{Float64}(undef, size(w_d, 2)*L, (size(w_d,1)-L+1))
        for i=1:size(w_d, 2)
            hank_temp = hankel_scalar(w_d[:, i], L)
            if (size(hank) == size(hank_temp))
                hank = hank_temp
            else
                row_count = 1
                for j = 1:size(hank_temp, 1)
                    hank[row_count+i-1, :] = hank_temp[j, :]
                    row_count = row_count + size(w_d, 2)
                end
            end
        end
    end
    hank
end

function hankel_vector_new(w_d::Array, L::Any) 
    L = convert(Int64, L)
    dim = size(w_d, 1) #Dimension (how many signals used)
    Tmin = (dim + 1)*L-1 #T_min=(m + 1)*L - 1
    T = size(w_d, 2) #Trajectory time length
    Lmax = floor((T+1)/(dim+1)) #L_max = floor (T+1)/(q+1)

    if(T < Tmin) || (L > Lmax)
        println("Min length is $Tmin, length of matrix is $T")
        println("Max depth is $Lmax, depth selected is $L")
        return 
    else
        hank = Array{Float64}(undef, dim*L, (T-L+1))
        for i=1:L
            hank[(i-1)*dim+1:(i-1)*dim+dim, :] = w_d[:, i:T-L+i]
        end
    end
    hank
end

function hankel_scalar(traj, L)
    hank = Array{Float64}(undef, L, (size(traj,1)-L+1))
    for i=1:(size(traj,1)-L+1)
        for j=1:L
            hank[j,i] = traj[j+i-1]
        end   
    end
    hank
end

function Lasso(Y, X, γ, λ = 0)  
    #taken from https://jump.dev/Convex.jl/stable/examples/general_examples/lasso_regression/
    # this function can't solve for matrix of matrices
    (T, K) = (size(X, 1), size(X, 2))
    b_ls = X \ Y                    #LS estimate of weights, no restrictions

    Q = X'X / T
    c = X'Y / T                      #c'b = Y'X*b

    b = Variable(K)              #define variables to optimize over
    L1 = quadform(b, Q)            #b'Q*b
    L2 = dot(c, b)                 #c'b
    L3 = norm(b, 1)                #sum(|b|)
    L4 = sumsquares(b)            #sum(b^2)

    if λ > 0
        Sol = minimize(L1 - 2 * L2 + γ * L3 + λ * L4)      #u'u/T + γ*sum(|b|) + λ*sum(b^2), where u = Y-Xb
    else
        Sol = minimize(L1 - 2 * L2 + γ * L3)               #u'u/T + γ*sum(|b|) where u = Y-Xb
    end
    solve!(Sol, SCS.Optimizer; silent_solver = true)
    Sol.status == Convex.MOI.OPTIMAL ? b_i = vec(evaluate(b)) : b_i = NaN

    return b_i, b_ls
end

function find_w_given(traj, times)
    w_given = zeros(0)
    for i= 1:size(times, 1)
        append!(w_given, traj[times[i]])
    end
    w_given
end

function behavioural(wd, w_given, L, T, γ)
    t_given = 1:size(wd, 2)*T #Potentially size(w_given, 1)?
    hank = hankel_vector(wd, L)
    #print(size(hank))

    b_opt, b_ls = Lasso(w_given, hank[t_given, :], γ) 
    values = hank*b_opt

    predictions = values[(size(wd, 2)*T)+1:size(wd, 2)*L]
    # if the hankel matrix and w_given contain more than just price data we need to 
    # extract only the adj close values since "prediction" will contain prediction values for 
    # all data used e.g will contain adj_close and volume predictions

    adj_close_predictions = get_adj_close_predictions(predictions, size(wd, 2))

    return adj_close_predictions
end 

function behavioural_new(wd, w_given, L, T, γ)
    t_given = 1:size(w_given, 1) #Potentially size(w_given, 1)?
    hank = hankel_vector_new(wd, L)
    #print(size(hank))

    b_opt, b_ls = Lasso(w_given, hank[t_given, :], γ) 
    values = hank*b_opt

    predictions = values[(size(wd, 1)*T)+1:size(wd, 1)*L]
    # if the hankel matrix and w_given contain more than just price data we need to 
    # extract only the adj close values since "prediction" will contain prediction values for 
    # all data used e.g will contain adj_close and volume predictions

    adj_close_predictions = get_adj_close_predictions(predictions, size(wd, 1))

    return adj_close_predictions
end 

function behavioural_prediction(data::Array, train_data::Array, test_data::Array, hankel_depth::Any, num_preds::Int, γ)
    #data is the concatenation of whatever signals is being used
    #train_data is the first 2/3 of the data
    #test_data is the remaining 1/3 of data. Really train is being used to get the best parameters
    hankel_depth = convert(Int64, hankel_depth)
    empty_array = Array{Float64}(undef, size(test_data,1)+1, num_preds)
    #Empty array is a place holder for the predictions of these days, +1 because you predict on a moving
    # window so you can predict for the interval after the "train_data"
    T = hankel_depth-num_preds #Size of w_given (t_given)
    size_wd = size(train_data, 1)-T

    for i in 1:size(test_data,1)+1
        w_given = form_w_given(data[i+size_wd:i+size(train_data, 1)-1, :]) #Flatten the price and volume data
        #w_given could be improved, the problem with reshape is that it is column major
        tmp_preds = behavioural(data[i:i+size_wd-1, :], w_given, hankel_depth, T, γ)
        tmp_preds = Array(transpose(tmp_preds))
        empty_array[i, :] =  tmp_preds
    end
    tmp_miss = Array{Union{Missing, Int}}(missing, size(train_data, 1), num_preds)
    predictions = empty_array
    predictions = (vcat(tmp_miss, empty_array))
    predictions
end

function behavioural_prediction_new(data::Array, train_data::Array, test_data::Array, hankel_depth::Any, num_preds::Int, γ)
    #data is the concatenation of whatever signals is being used
    #train_data is the first 2/3 of the data
    #test_data is the remaining 1/3 of data. Really train is being used to get the best parameters
    hankel_depth = convert(Int64, hankel_depth)
    empty_array = Array{Float64}(undef, size(test_data,2)+1, num_preds)
    #Empty array is a place holder for the predictions of these days, +1 because you predict on a moving
    # window so you can predict for the interval after the "train_data"
    w_given_size = hankel_depth-num_preds #Size of w_given (t_given)
    size_wd = size(train_data, 2)-w_given_size

    for i in 1:size(test_data,2)+1
        w_given = form_w_given_new(data[:,i+size_wd:i+size(train_data, 2)-1]) #Flatten the price and volume data
        #w_given could be improved, the problem with reshape is that it is column major
        tmp_preds = behavioural_new(data[:, i:i+size_wd-1], w_given, hankel_depth, w_given_size, γ)
        tmp_preds = Array(transpose(tmp_preds))
        empty_array[i, :] =  tmp_preds
    end
    tmp_miss = Array{Union{Missing, Int}}(missing, size(train_data, 1), num_preds)
    predictions = empty_array
    predictions = (vcat(tmp_miss, empty_array))
    predictions
end

function form_w_given(array::Array)
    empty_array = Array{Float64}(undef, size(array, 1)*size(array, 2), 1) 
    counter = 1
    for i=1:size(array, 1)
        for j=1:size(array, 2)
            empty_array[counter, 1] = array[i, j]
            counter = counter + 1
        end
    end
    empty_array
end

function form_w_given_new(array::Array)
    empty_array = Array{Float64}(undef, size(array, 1)*size(array, 2), 1) 
    empty_array = reshape(array, (size(array, 1)*size(array, 2), 1))
    empty_array
end

function get_adj_close_predictions(predictions::Array, space::Int)
    adj_close_predictions = Array{Float64}(undef, Int(size(predictions, 1)/space), 1)
    counter = 1
    for i=1:size(adj_close_predictions, 1)
        adj_close_predictions[i, 1] = predictions[counter]
        counter = counter + space
    end
    adj_close_predictions
end








