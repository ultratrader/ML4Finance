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
            #[:low, :high] => ((l,h) -> h .- l) => :dayrange,
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
        push!(v, m')
    end
    return v
end

# 2 select only a selected column
function gdftoarray!(gdf, colindex::Int64)
    v = []
    for i in eachindex(gdf)
        m = Array{Float32}(gdf[i][!,colindex])
        push!(v, m')
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