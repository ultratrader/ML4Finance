using Pipe
using Dates: Date
using LibPQ, Tables
using DataFrames: DataFrame, groupby, combine, dropmissing!
using ShiftedArrays: lag
include("/home/najeeb/Projects/Ned/DataStreams/src/DataStreams.jl")
using .DataStreams: get_psql
include("Flux3dArray.jl")


# Data Preparations, always munging and cleansing before processed
df1 = get_psql("SELECT date,close,volume FROM idlfidxcdmax WHERE scode='BBCA.JK'","twsdb")
df1 = @pipe df1|>
        sort!(_, :date)|>
        combine(_,
            :date => :date,
            :close => (y -> y./lag(y,1).-1) => :rt,
            :volume => (v -> (v .- minimum(v)) ./ (maximum(v) .- minimum(v))) => :volz    
        )
df2 = get_psql("SELECT date,close,volume FROM idlfidxcdmax WHERE scode='BMRI.JK'","twsdb")
df2 = @pipe df2|>
        filter(:date =>>=(Date("2004-06-08")),_)|>
        sort!(_, :date)|>
        combine(_,
            :date => :date,
            :close => (y -> y./lag(y,1).-1) => :rt,
            :volume => (v -> (v .- minimum(v)) ./ (maximum(v) .- minimum(v))) => :volz    
        )
df = vcat(df1,df2)
sort!(df, :date) # sorting by date before grouped
dropmissing!(df)
gdf = groupby(df, :date)  # grouping by date to convert each time step to Array(2,2)

# transform data to Flux RNN format
# drop date by not selecting it
# create function to convert gdf to array of features x batches(entity)
# 1 convert gdf returned by DataFrames.jl
function gdftoarray(gdf)
    v = []
    for i in eachindex(gdf)
        m = Array{Float32}(gdf[i][!,2:end]) # drop date
        push!(v, m')
    end
    return v
end

# 2 select only certain column
function gdftoarray!(gdf, colindex::Int64)
    v = []
    for i in eachindex(gdf)
        m = Array{Float32}(gdf[i][!,colindex])
        push!(v, m')
    end
    return v
end

X = gdftoarray(gdf)
Y = gdftoarray!(gdf,2)
X = X[1:(end-1)] #drop last row
Y = Y[2:end]

# prepare training data x,y for loss computations
function split2traintest(v, cutoff::Int64)
    train_vec = v[1:cutoff]
    test_vec = v[(cutoff+1):end]
    return train_vec, test_vec
end
# split data into training and testing
xtrain, xtest = split2traintest(X,4000)
ytrain, ytest = split2traintest(Y,4000)

# prepare data x,y for training model by slicing into sequences
## 200 sequences, 20 time steps in each sequence 
## each time step contains:
## 2 input features x 2 entities(ticker) and 
## 1 output x 2 entities(ticker)
Xₜ = slice_vec(xtrain, 20)  # 200 sequences
Yₜ = slice_vec(ytrain, 20)  # 200 sequences
Xₜ[1]       # 1 sequences, each = 20 time step
Xₜ[1][1]    # 2x2 Array for each time step
# Training the models
using Flux
using Flux: params
using Flux.Optimise: update!
using Base.Iterators
using StatsBase: mean

# Method 1
using Flux: mse
function loss(x, y)
    sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
end

data = zip(Xₜ,Yₜ)

Flux.reset!(m)
m = Chain(LSTM(2, 5), Dense(5, 3, tanh), Dense(3, 1))
ps = Flux.params(m)
opt= Adam(1e-3)
epochs = 100
for i = 1:epochs
    Flux.train!(loss, ps, data, opt) # simple but not flexibles
end

# using test data, predict using optimized m
pred_lstm = [m(i) for i in xtest]
y_actual = [i for i in ytest]
mean_se = mean(abs2.(y_actual .- pred_lstm)) # check if mse == minimum(e)

# take only certain stock vector
yhat = [pred_lstm[i][2] for i in 1:length(pred_lstm)]
y = [y_actual[i][2] for i in 1:length(y_actual)]
yhatcum = cumsum(yhat)
ycum = cumsum(y)

# Plots
using PlotlyJS
date0 = 1:length(xtest)
t1 = scatter(x=date0, y=yhat, mode="lines", name="y prediction", opacity=0.5)
t2 = scatter(x=date0, y=y, mode="lines", name="y actual", opacity=0.5)
plot([t1,t2])

# plotting cumulative return
t3 = scatter(x=date0, y=yhatcum, mode="lines", name="prediction", opacity=0.7)
t4 = scatter(x=date0, y=ycum, mode="lines", name="actual", opacity=0.7)
plot([t3,t4])


# ----------------------------------------------------------------------
# method 2
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
m = Chain(LSTM(2, 5), Dense(5, 3, tanh), Dense(3, 1))

# set up the training
# train while out-of-sample improves, saving best model.
# stop when the out-of-sample has increased too many times   
function trains(model, in, out, epochs::Int64) ## max number of training loops
    bestmodel = model;
    numincreases = 0;
    maxnumincreases = Int64(0.1*epochs);
    ps = Flux.params(model);
    opt = ADAM();
    e = [];
    for i = 1:epochs
        gs = gradient(ps) do
            loss(in,out)
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
m,e = trains(m,xtrain,ytrain,500)
minimum(e)


# using test data, predict using optimized m
pred_lstm = predict(xt)
y_actual = [i[1] for i in yt]
mean_se = mean(abs2.(y_actual .- pred_lstm)) # check if mse == minimum(e)
cumrt_lstm = cumsum(pred_lstm)
cumrt = cumsum(y_actual)

# Plots
using PlotlyJS
date0 = date[length(date)-79:end]
t1 = scatter(x=date0, y=pred_lstm, mode="lines", name="y prediction", opacity=0.5)
t2 = scatter(x=date0, y=y_actual, mode="lines", name="y actual", opacity=0.5)
plot([t1,t2])

# plotting cumulative return
t3 = scatter(x=date0, y=cumrt_lstm, mode="lines", name="prediction", opacity=0.7)
t4 = scatter(x=date0, y=cumrt, mode="lines", name="actual", opacity=0.7)
plot([t3,t4])