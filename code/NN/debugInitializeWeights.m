function W = debugInitializeWeights(fan_out, fan_in)
%   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
%   of a layer with fan_in incoming connections and fan_out outgoing 
%   connections using a fixed set of values
%
%   W is set to a matrix of size(1 + fan_in, fan_out) as
%   the first row of W handles the "bias" terms
%

% Setting W to zeros
W = zeros(fan_out, 1 + fan_in);

% Initializing W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end
