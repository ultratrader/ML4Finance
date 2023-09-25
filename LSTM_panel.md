### Notes to LSTM for Panel(tidy) data
##### Issues
1. features array cannot contain missing value that will lead to change in input or output size and throw "dimension mismatch error" when calculating mse using Flux.mse.
2. the parameter is not stable, it exhibit different parameter every time the model trained, even using the same data that indicate the model does not reach convergence.
3. the resulting forecast, may be only in certain configurations, exhibits persistences in prediction  