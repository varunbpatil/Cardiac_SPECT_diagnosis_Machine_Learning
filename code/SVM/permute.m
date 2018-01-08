function [C, sigma] = permute(X, y, Xval, yval)
%   [C, sigma] = PERMUTE(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.1;


% returns the optimal C and sigma
% learning parameters found using the cross validation set.


c = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
s = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
min_error = 99999;
k=1;

for i = 1:size(c)
for j = 1:size(s)
	fprintf('\n\nC = %f \t Sigma = %f', c(i), s(j));
	model = svmTrain(X, y, c(i), @(x1, x2) gaussianKernel(x1, x2, s(j)));	
	predictions = svmPredict(model, Xval);
	z = mean(double(predictions ~= yval));
	fprintf('Error on CV set = %f\n\n', z);
	if(z<min_error)
		min_error_1=i;
		min_error_2=j;
		min_error=z;
	endif
	
endfor
endfor

C = c(min_error_1);
sigma = s(min_error_2);
fprintf('\n\nBest values(least CV set error):\tC = %f\tSigma = %f\tError = %f\n\n\n', C, sigma, min_error);
% =========================================================================

end
