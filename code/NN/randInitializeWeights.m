function W = randInitializeWeights(L_in,L_out)
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   W is set to a matrix of size(L_out, 1 + L_in) as
%   the first row of W handles the "bias" terms
%

W = zeros(L_out, 1 + L_in);

% Initialize W randomly so that we break the symmetry while
% training the neural network.


epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

% =========================================================================

end
