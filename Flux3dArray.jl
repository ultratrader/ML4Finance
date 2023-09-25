"""
IMPORTANT!
A potential source of ambiguity with RNN in Flux can come
 from the different data layout compared to some common 
 frameworks where data is typically a 3 dimensional array: 
 (features, seq length, samples). In Flux, those 3 
 dimensions are provided through a vector of seq length 
 containing a matrix with size(features, samples).
"""

"""
    tabular2rnn(X)
Converts tabular data `X` into an RNN sequence format. 
`X` should have format T × K × M, where T is the number of time steps, K is the number 
of features, and M is the number of batches.
"""
tabular2rnn(X::AbstractArray{Float32, 3}) = [X[t, :, :] for t ∈ 1:size(X, 1)]

"""
    rnn2tabular(X)
Converts RNN sequence format `X` into tabular data.
"""
rnn2tabular(X::Vector{Matrix{Float32}}) = permutedims(cat(X..., dims=3), [3, 1, 2])

"""
    split_uniseries(series, timestep)
function to split a univariate series into AR(p) problems
returning x,y vector{vector}
"""
function split_uniseries(series::AbstractArray, tstep::Int64)
    X = []
    Y = []
    for i in 1:length(series)
        last = i + tstep
        if last > length(series)
            break
        else
            seq_x, seq_y = series[i:(last-1)], [series[last]]
            push!(X, seq_x)
            push!(Y, seq_y)
        end
    end
    return X, Y
end

"""
    slice_vec(v, seq_len)
function to slice a vector into multiple sequences thus will
have structures Vector{Vector{Vector}} with 
Batch(B) x Sequences(S) x Features(F)
"""
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