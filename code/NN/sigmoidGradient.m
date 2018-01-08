function g = sigmoidGradient(z)
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z.

g = zeros(size(z));

% Compute the gradient of the sigmoid function evaluated at
% each value of z (z can be a matrix, vector or scalar).


m=size(z,1);
n=size(z,2);
t=ones(m,n);
g=sigmoid(z).*(t-sigmoid(z));

% =============================================================

end
