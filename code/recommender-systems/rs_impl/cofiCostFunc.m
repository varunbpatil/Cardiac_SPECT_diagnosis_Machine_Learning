function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

%   [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), num_users, num_features);

            
% Computing cost function J and gradients

% initialize the return variables to all zeros
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


A = ((X*Theta') - (Y)).^2;
A = A .* R;
s = sum(sum(A));
J = s/2;
D = Theta .^ 2;
E = X .^ 2;
J = J + (lambda/2)*sum(sum(D)) + (lambda/2)*sum(sum(E));	% regularized cost function


% regularized gradients
for i=1:num_movies
	[idx, idy] = find(R(i,:)==1);
	X_grad(i,:) = (((X(i,:)*Theta(idy,:)') - Y(i,idy)) * Theta(idy,:)) + (lambda*X(i,:));
endfor

for i=1:num_users
	[idx, idy] = find(R(:,i)==1);
	Theta_grad(i,:) = (((X(idx,:)*Theta(i,:)')-Y(idx,i))' * X(idx,:)) + (lambda*Theta(i,:));
endfor

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
