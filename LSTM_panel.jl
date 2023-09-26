# Important!
# LSTM_panel.md contains related error other important issues

#using Pipe
using CSV: read
using Dates: Date
using DataFrames: DataFrame, groupby, combine, dropmissing!
include("Flux3dArray.jl")


# Data Preparations, always munging and cleansing before processed
df = read("data/paneldata.csv", DataFrame)

function lags(v, n::Int64)
    l = v[1:length(v)-n]
    m = [missing for i = 1:n]
    new_v = vcat(m,l)
    return new_v
end


data = @pipe df|>
        sort(_, :date)|>
        filter(:date =>>=(Date("2005-01-01")),_)|>
        groupby(_,:scode)|> #group by ticker
        combine(_,
            :date => :date,
            :close => (y -> y./lags(y,1).-1) => :rt,
            :volume => (v -> (v .- minimum(v)) ./ (maximum(v) .- minimum(v))) => :volz    
        )

sort!(data, :date) # sorting by date before grouped
dropmissing!(data)
gdf = groupby(data, :date)  # grouping by date to convert each time step to Array(2,2)

# transform data to Flux RNN format
# drop date by not selecting it
# create function to convert gdf to array of features x batches(entity)
# convert gdf returned by DataFrames.jl
function gdftoarray(gdf, colindex::Int64)
    v = []
    for i in eachindex(gdf)
        m = Array{Float32}(gdf[i][!,colindex:end]) # drop date
        push!(v, permutedims(m)) # use Base.permutedims for general data manipulations
    end
    return v
end

# 2 select only a selected column
function gdftoarray!(gdf, colindex::Int64)
    v = []
    for i in eachindex(gdf)
        m = Array{Float32}(gdf[i][!,colindex])
        push!(v, permutedims(m))
    end
    return v
end

X = gdftoarray(gdf,3)
Y = gdftoarray!(gdf,3)
# assume that Y(output) is a function of first lag of X(input features)
X = X[1:(end-1)] #drop last row
Y = Y[2:end] #drop first row

# prepare training data x,y for loss computations
function split_vec(v, cutoff::Int64)
    train_vec = v[1:cutoff]
    test_vec = v[(cutoff+1):end]
    return train_vec, test_vec
end
# split data into training and testing
xtrain, xtest = split_vec(X,1000)
ytrain, ytest = split_vec(Y,1000)

# prepare data x,y for training model by slicing into sequences
## 200 sequences, 20 time steps in each sequence 
## each time step contains:
## 2 input features x 3 entities(ticker) and 
## 1 output x 3 entities(ticker)
Xₜ = vec2sequence(xtrain, 20)  # 200 sequences
Yₜ = vec2sequence(ytrain, 20)  # 200 sequences
Xₜ[1]       # 1 sequences, each = 20 time step
Xₜ[1][1]    # 2x3 Array for each time step
# Training the models
using Flux
using Flux: params
using Flux.Optimise: update!
#using Base.Iterators
#using StatsBase: mean
using Flux: mse
function loss(x, y)
    Flux.reset!(m)
    sum(mse(m(xi), yi) for (xi, yi) in zip(x, y))
end

d = zip(Xₜ,Yₜ)
#d = zip(xtrain, ytrain)

Flux.reset!(m)
m = Chain(LSTM(2, 5), Dense(5, 3, tanh), Dense(3, 1))
ps = Flux.params(m)
opt= Adam(1e-3)
Flux.train!(loss, ps, d, opt)
epochs = 10
for i = 1:epochs
    Flux.train!(loss, ps, d, opt) # simple but not flexibles
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