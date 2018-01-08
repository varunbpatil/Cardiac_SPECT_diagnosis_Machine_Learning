function [mu, sig, theta] = cm(im)
image = imread(im); % returns M X N X 3 matrix in "image"
[M, N] = size(image)
% calculate mu for each of the 3 planes
sum = [0; 0; 0];
for i=1 to M
	for j=1 to N
		sum(1) = sum(1) + image(i,j,1);	
		sum(2) = sum(2) + image(i,j,2);	
		sum(3) = sum(3) + image(i,j,3);	
	endfor
endfor
mu = [0; 0; 0];
mu(1) = sum(1) / (M * N);
mu(2) = sum(2) / (M * N);
mu(3) = sum(3) / (M * N);

% calculate sig for each of the 3 planes
sum1 = [0; 0; 0];
for i=1 to M
	for j=1 to N
		sum1(1) = sum1(1) + (image(i,j,1) - mu(1))^2;	
		sum1(2) = sum1(2) + (image(i,j,2) - mu(2))^2;	
		sum1(3) = sum1(3) + (image(i,j,3) - mu(3))^2;	
	endfor
endfor
sig = [0; 0; 0];
sig(1) = sqrt(sum1(1) / (M * N));
sig(2) = sqrt(sum1(2) / (M * N));
sig(3) = sqrt(sum1(3) / (M * N));

% calculate theta for each of the 3 planes
sum2 = [0; 0; 0];
for i=1 to M
	for j=1 to N
		sum2(1) = sum2(1) + (image(i,j,1) - mu(1))^3;	
		sum2(2) = sum2(2) + (image(i,j,2) - mu(2))^3;	
		sum2(3) = sum2(3) + (image(i,j,3) - mu(3))^3;	
	endfor
endfor
theta = [0; 0; 0];
theta(1) = nthroot((sum2(1) / (M * N)), 3);
theta(2) = nthroot((sum2(2) / (M * N)), 3);
theta(3) = nthroot((sum2(3) / (M * N)), 3);

end
