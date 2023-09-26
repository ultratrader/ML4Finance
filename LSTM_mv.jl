# Testing only!
# Method 1: Ref: https://discourse.julialang.org/t/lstm-training-for-a-sequence-of-multiple-features-using-a-batch-size-30/63238/2

using Pipe
using CSV: read
using Dates: Date
using DataFrames: DataFrame, groupby, combine, dropmissing!

# Data Preparations, always munging and cleansing before processed
df = read("data/paneldata.csv", DataFrame)

function lags(v, n::Int64)
    l = v[1:length(v)-n]
    m = [missing for i = 1:n]
    new_v = vcat(m,l)
    return new_v
end

data = @pipe df|>
        filter(:scode =>==("BBCA.JK"),_)|>
        sort(_, :date)|>
        combine(_,
            :close => (y -> y./lags(y,1).-1) => :rt,
            :volume => (v -> (v .- minimum(v)) ./ (maximum(v) .- minimum(v))) => :volz    
        )

dropmissing!(data)
# arrange so that y=f(lag(x,1))
X = Array{Float32}(data[1:(end-1),:])
Y = Array{Float32}(data[2:end,1])

# Method 1: using 3d Array
# 
# transform data to Flux RNN format
# reshape data to 3d array: (timestep,features,sequences/samples)
Xₜ = reshape(permutedims(X),(1,2,:))
Yₜ = reshape(Y,(1,1,:))

# prepare training data
function split_3darray(v, cutoff::Int64)
    train_vec = v[:,:,1:cutoff]
    test_vec = v[:,:,(cutoff+1):end]
    return train_vec, test_vec
end
# split data into training and testing
xtrain, xtest = split_3darray(Xₜ,4000)
ytrain, ytest = split_3darray(Yₜ,4000)

# batching...
using Flux.Data: DataLoader
xytrain = DataLoader((xtrain, ytrain), batchsize=30, shuffle=true)
size.(first(xytrain)) # check array size

# Training the models
using Flux
using Flux: params
using Flux.Optimise: update!
using Flux: mse

Flux.reset!(m)
m = Chain(
    LSTM(2, 10),
    Dropout(0.5),
    Dense(10, 1,σ)
    )
#evaluating prediction for the input sequence
#inputs = permutedims([xtrain[t,:,:] for t=1][1])
#x_vec = [inputs[i,:] for i = 1:size(inputs,1)]
function mat2vvec(M)
    M = M[1,:,:]
    Mₚ = permutedims(M)
    v = [Mₚ[i,:] for i = 1:size(Mₚ,1)]
    return v
end
xx = mat2vvec(xtrain)

m.(xx)
function eval_model(x)
    Flux.reset!(m)
    output = m.(mat2vvec(x))
end
L(x, y) = sum(mse.(eval_model(x), mat2vvec(y)))
L(xtrain, ytrain)

## 1.1 using train!()
ps = Flux.params(m)
opt= Adam(1e-3)
Flux.train!(L, ps, xytrain, opt) #ok


## 1.2 Using custom train
# Train model on 20 epochs
using Flux.Optimise: update!
ps = Flux.params(m)
opt= Adam(1e-3)
for epoch = 1:20
    @show epoch
    for d in xytrain
      gs = gradient(ps) do
        l = L(d...)
      end
      update!(opt, ps, gs)
    end
end

## Plotting to see how good the prediction made
yhat0 = m.(mat2vvec(xtest))
yhat = [yhat0[i][1] for i in 1:length(yhat0)]
y_act = [mat2vvec(ytest)[i][1] for i in 1:length(mat2vvec(ytest))]
x = 1:length(yhat)
# see the prediction in cumulative return views
yhatcum = cumsum(yhat)
ycum = cumsum(y_act)

# Plots
using PlotlyJS: plot, scatter
t1 = scatter(x=x, y=yhat, mode="lines", name="y prediction", opacity=0.5)
t2 = scatter(x=x, y=y_act, mode="lines", name="y actual", opacity=0.5)
plot([t1,t2])

# plotting cumulative return
t3 = scatter(x=x, y=yhatcum, mode="lines", name="prediction", opacity=0.7)
t4 = scatter(x=x, y=ycum, mode="lines", name="actual", opacity=0.7)
plot([t3,t4])


# Method 2: using 2d Array
# Ref: https://fluxml.ai/Flux.jl/stable/models/recurrence/
# starting from dataframe outputs
dropmissing!(data)
# arrange so that y=f(lag(x,1))
X = Array{Float32}(data[1:(end-1),:])
Y = Array{Float32}(data[2:end,1])
# prepare data x,y for training model
## 93 batches, 100 sequences, 2 features
Xₜ = [X[i,:] for i = 1:size(X,1)]
Yₜ = [Y[i,:] for i = 1:size(Y,1)]
# function to split 2d array
function split_2darray(v, cutoff::Int64)
    train_vec = v[1:cutoff]
    test_vec = v[(cutoff+1):end]
    return train_vec, test_vec
end

xtrain, xtest = split_2darray(Xₜ,4000)
ytrain, ytest = split_2darray(Yₜ,4000)
# function for slice vector in sequences
function vec2sequence(v::Vector, seq_len::Int64)
    new_v = []
    for i = range(1,length(v),step=seq_len)
        if i+(seq_len-1)>length(v)
            break
        else 
            vₛ = v[i:(i+(seq_len-1))]
        end
        push!(new_v,vₛ)
    end
    return new_v
end
xₜ = vec2sequence(xtrain, 20)
yₜ = vec2sequence(ytrain, 20)

# prepare testing data
# test data is data that not used in train after 
# slicing into sequences
#xt = xₜ[length(xₜ)-79:end]  # out-of-sample testing data for x
#yt = yₜ[length(yₜ)-79:end]  # out-of-sample testing data for y

using Flux
using Flux: params
using Flux: mse
using Flux.Optimise: update!
#using Base.Iterators
using StatsBase

# the loss function for training
function loss(x,y)
    Flux.reset!(m)
    #m(x[1]) # warm-up internal state
    sum(mse.([m(xi) for xi ∈ x], y))
end

# function to get prediction of y conditional on lags of y.
function predict(data)
    Flux.reset!(m)
    m(data[1]) # warm-up internal state
    [m(d)[1] for d ∈ data] # output predictions as a vector
end

# function for checking out-of-sample fit
function callback()
    error=mean(abs2.([t[1] for t ∈ ytest] - predict(xtest)))
    println("testing mse: ", error)
    return error
end

# the model: this is just a guess for illustrative purposes
# there may be better configurations
Flux.reset!(m)
#m = Chain(LSTM(3, 5), Dense(5, 3, tanh), Dense(3, 1))
m = Chain(
    LSTM(2, 10),
    Dropout(0.5),
    Dense(10, 1,σ)
    )

# set up the training
ps = Flux.params(m)
opt= Adam(1e-3)
for epoch = 1:20
    @show epoch
    for d in (xtrain,ytrain)
      gs = gradient(ps) do
        l = loss.(d...)
      end
      update!(opt, ps, gs)
    end
end


# train while out-of-sample improves, saving best model.
# stop when the out-of-sample has increased too many times   
function trains(model, epochs::Int64) ## max number of training loops
    bestmodel = model;
    numincreases = 0;
    maxnumincreases = Int64(0.1*epochs);
    ps = Flux.params(model);
    opt = ADAM();
    e = [];
    for i = 1:epochs
        gs = gradient(ps) do
            loss(xtrain,ytrain)  # change to trained data without sequence
        end
        Flux.update!(opt, ps, gs)
        c = callback()
        push!(e, c)
        if isless(c,minimum(e))
            bestmodel = deepcopy(model)
        else
            numincreases +=1
        end    
        numincreases > maxnumincreases ? break : nothing
    end
    model = bestmodel
    return model,e
end
m,e = trains(m, 100)
minimum(e)

# using test data, predict using optimized m
pred_lstm = predict(xtest)
y_actual = [i[1] for i in ytest]
mean_se = mean(abs2.(y_actual .- pred_lstm)) # check if mse == minimum(e)
cumrt_lstm = cumsum(pred_lstm)
cumrt = cumsum(y_actual)
x = 1:length(pred_lstm)

# Plots
using PlotlyJS
t1 = scatter(x=x, y=pred_lstm, mode="lines", name="y prediction", opacity=0.5)
t2 = scatter(x=x, y=y_actual, mode="lines", name="y actual", opacity=0.5)
plot([t1,t2])

# plotting cumulative return
t3 = scatter(x=x, y=cumrt_lstm, mode="lines", name="prediction", opacity=0.7)
t4 = scatter(x=x, y=cumrt, mode="lines", name="actual", opacity=0.7)
plot([t3,t4])

