# Testing only!
# Ref: https://discourse.julialang.org/t/lstm-training-for-a-sequence-of-multiple-features-using-a-batch-size-30/63238/2

using Pipe
using CSV: read
using Dates: Date
using DataFrames: DataFrame, groupby, combine, dropmissing!
#include("Flux3dArray.jl")

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
X = Array{Float32}(data[1:(end-1),:])
Y = Array{Float32}(data[2:end,1])

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
#inputs = Flux.flatten(xtrain) #[xtrain[1,t,:] for t in 1:14]
#inputs0 = [xtrain[1,1:2,t] for t = 1:4000]
#outputs0 = [[ytrain[1,1,t]] for t = 1:4000]

# batching...
xytrain = DataLoader((xtrain, ytrain), batchsize=30, shuffle=true)
xload = DataLoader(xtrain, batchsize=30, shuffle=true)
yload = DataLoader(ytrain, batchsize=30, shuffle=true)
size.(first(xytrain)) # check array size

# Training the models
using Flux
using Flux.Data: DataLoader
using Flux: params
#using Flux.Optimise: update!
#using Base.Iterators
#using StatsBase: mean
using Flux: mse

Flux.reset!(m)
m = Chain(
    LSTM(2, 10),
    Dropout(0.5),
    Dense(10, 1,σ)
    )
#m = Chain(LSTM(2, 5), Dense(5, 3, tanh), Dense(3, 1))
#evaluating prediction for the input sequence
inputs = [xtrain[t,:,:] for t=1]
inputs = reshape(xtrain,())
outputs = [ytrain[t,:,:] for t=1]
m.(inputs)
function eval_model(x)
    Flux.reset!(m)
    inputs = [xtrain[t,:,:] for t=1]
    output = m.(inputs)
end
L(x, y) = mse(eval_model(x), [y[t,:,:] for t=1][1])
L(xtrain, ytrain)

ps = Flux.params(m)
opt= Adam(1e-3)
Flux.train!(L, ps, xytrain, opt)   #zip(xload,yload)

# trying custom methods
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

# ----------------------------------------------------
## trial 2
Flux.reset!(m)
m = Chain(
    LSTM(2, 10),
    Dropout(0.5),
    Dense(10, 1,σ)
    )
#m = Chain(LSTM(2, 5), Dense(5, 3, tanh), Dense(3, 1))
#evaluating prediction for the input sequence
rnn2tabular(X) = permutedims(cat(X..., dims=3), [3, 1, 2])
inputs = rnn2tabular(xtrain)

m.(inputs)
function eval_model(x)
    Flux.reset!(m)
    inputs = [xtrain[t,:,:] for t=1]
    output = m.(inputs)
end
L(x, y) = mse(eval_model(x), [y[t,:,:] for t=1][1])
L(xtrain, ytrain)

ps = Flux.params(m)
opt= Adam(1e-3)
Flux.train!(L, ps, xytrain, opt)   #zip(xload,yload)

# trying custom methods
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


# using test data, predict using optimized m
pred_lstm = [m(i) for i in xtest[1:20]]
y_actual = [i for i in ytest[1:20]]


# take only certain stock vector
yhat = [pred_lstm[i][1] for i in 1:length(pred_lstm)]
y = [y_actual[i][1] for i in 1:length(y_actual)]
yhatcum = cumsum(yhat)
ycum = cumsum(y)

# Plots
using PlotlyJS
x = 1:length(yhat)
t1 = scatter(x=x, y=yhat, mode="lines", name="y prediction", opacity=0.5)
t2 = scatter(x=x, y=y, mode="lines", name="y actual", opacity=0.5)
plot([t1,t2])

# plotting cumulative return
t3 = scatter(x=x, y=yhatcum, mode="lines", name="prediction", opacity=0.7)
t4 = scatter(x=x, y=ycum, mode="lines", name="actual", opacity=0.7)
plot([t3,t4])

## -------------------------------------------------------
# Multivariate LSTM

using Flux.Optimise: update!
# Generate data
data_x = rand(Float32, 3, 11, 1325)
data_y = rand(Float32, 1, 1, 1325)
rnd_loader = DataLoader((data_x, data_y), shuffle = true)

# Define model
Flux.reset!(m)
model = Chain(
    LSTM(3, 3),
    Dense(3,1),
    Flux.flatten, #Reshape the LSTM output (1,11,1325) into (11,1325)
    Dense(11, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1))

# Define loss function
Loss(x, y) = Flux.mse((x), y)
function loss_lstm(x, y)
    Flux.reset!(model)
    yhat_reshaped = reshape(model(x), 1, :, 1) #Predict y and reshape to (1,1325,1)
    Loss(yhat_reshaped, permutedims(y[:,:,:],[2, 3, 1])) 
end
loss_lstm(data_x, data_y) #Test if loss function works

opt = ADAM(0.001) #Set optimiser learning rate

# Train model on 20 epochs
for epoch = 1:20
    @show epoch
    for d in rnd_loader
      gs = gradient(Flux.params(model)) do
        l = loss_lstm(d...)
      end
      update!(opt, Flux.params(model), gs)
    end
end

# problems when trying to predicts
yhat = [model(permutedims(data_x[:,:,i])) for i = 1:1325]



## ------------------------------------------------------
function loss(x, y)
    Flux.reset!(m)
    sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
end

x = [xtrain[1,1:2,t] for t = 1:4000]
x[1]
#evaluating prediction for the input sequence
function loss(x,y)
    Flux.reset!(m)
    x = [xtrain[1,1:2,t] for t = 1:4000]
    y = [ytrain[1,1,t] for t = 1:4000]
    yhat = m(inputs)
    sum(mse(m(xi), yi) for (xi, yi) in (x, y))
end
#L(x, y) = mse(eval_model(x), y)