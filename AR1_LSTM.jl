# All credits go to mcreel, see https://discourse.julialang.org/t/simple-flux-lstm-for-time-series/35494/42
# https://gist.github.com/JLDC/4287637e00b8eaf0dcb9e7c5626edff8

using Flux, Plots, Statistics
using Base.Iterators

# data generating process is autoregressive order 1
# modeling objective is to forecast y conditional
# on lags of y
function AR1(n)
    y = zeros(n)
    for t = 2:n
        y[t] = 0.9*y[t-1] + randn()
    end
    y
end    

function main()
# generate the data
n = 1000 # sample size
data = Float32.(AR1(n))
# for LSTMs, we need to scale our data, use a min-max scaling scheme
mn, mx = minimum(data), maximum(data)
data = (data .- mn) ./ (mx .- mn)

n_training = Int64(round(2*n/3))
training = [[d] for d ∈ data[1:n_training]]
testing = [[d] for d in data[n_training+1:end]]
# set up the training
epochs = 1000    # maximum number of training loops through data

# the model: this is just a guess for illustrative purposes
# there may be better configurations
m = Chain(LSTM(1, 10), Dense(10, 2, tanh), Dense(2, 1))

# the loss function for training
function loss(x,y)
    Flux.reset!(m)
    m(x[1]) # warm-up internal state
    sum(Flux.mse.([m(xi) for xi ∈ x[2:end]], y[2:end]))
end

# function to get prediction of y conditional on lags of y.
function predict(data)
    Flux.reset!(m)
    m(data[1]) # warm-up internal state
    [m(d)[1] for d ∈ data[2:end]] # output predictions as a vector
end

# function for checking out-of-sample fit
function callback()
    error=mean(abs2.([t[1] for t ∈ testing[3:end]] - predict(testing[1:end-1])))
    println("testing mse: ", error)
    error
end    

# train while out-of-sample improves, saving best model.
# stop when the out-of-sample has increased too many times
bestsofar = 1e6
bestmodel = m
numincreases = 0
maxnumincreases = 50
ps = Flux.params(m)
opt = ADAM()
for i = 1:epochs
    gs = gradient(ps) do
        loss(training[1:end-1], training[2:end])
    end
    Flux.update!(opt, ps, gs)
    c = callback()
    if c < bestsofar
        bestsofar = c
        bestmodel = deepcopy(m)
    else
        numincreases +=1
    end    
    numincreases > maxnumincreases ? break : nothing
end
m = bestmodel # use the best model found

# NN forecast
pred_nn = predict([[d] for d ∈ data]) # align with ML forecast
pred_nn = pred_nn .* (mx .- mn) .+ mn # scale LSTM predictions

# scale back our data for OLS
data = data .* (mx .- mn) .+ mn

# maximum likelihood forecast, for reference
# OLS applied to AR1 is ML estimator
y = data[2:end]
x = data[1:end-1]
ρhat = x\y
pred_ml = x*ρhat



return pred_nn, pred_ml, y
end

pred_nn, pred_ml, y = main()
# verify that NN works as well as ML
n = size(pred_nn,1)
#plot(1:n, [pred_nn pred_ml pred_nn - pred_ml], labels=["neural net forecast" "ML forecast" "difference in forecasts"])
plot(1:n, [pred_nn y], labels=["neural net forecast" "actual"])