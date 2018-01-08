clear ; close all; clc

% Load the training dataset
M = csvread('SPECT.train');
[rows, cols] = size(M);
X = M(:, 2:cols);
y = M(:, 1);

% Load the test dataset
M = csvread('SPECT.test');
[rows, cols] = size(M);
% Xtest = M(:, 2:cols);
% Ytest = M(:, 1);

% Cross-validation set must be chosen randomly
% So, first shuffle the rows of SPECT.test and then choose the first 25 rows as the CV set
p = randperm(rows); % generate random numbers
M = M(p, :); % shuffle the rows 

Xcval = M(1:25, 2:cols); % this is the cross-validation set which is a part of the test set
ycval = M(1:25, 1);

Xtest = M(26:rows, 2:cols);
ytest = M(26:rows, 1);

fprintf('\nTraining Linear SVM ...\n')

% In practical implemetation, we should test all permutations for C and sigma 
% to get the best possible combinations of the two (for gaussian kernel).
% The function permute() does the above test
[C, sigma] = permute(X, y, Xcval, ycval);

C = 0.1; 
sigma = 0.1; % needed only for gaussian kernel

% you can use either a linear kernel or a non-linear(gaussian) kernel

% linear kernel 
model = svmTrain(X, y, C, @linearKernel);

% gaussian (non-linear) kernel
% model = svmTrain(X, y, C, @(x1,x2) gaussianKernel(x1, x2, sigma));

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n\n\n', mean(double(p == y)) * 100);



%% =================== Test Set Classification ================
% evaluating classifier on test set

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, Xtest);

fprintf('\nTest Accuracy: %f\n', mean(double(p == ytest)) * 100);
pause;


%%=====================sensitivity=============================
% sensitivity is the percentage of positive examples in the 
% cross validation set that are recognized correctly


m = size(Xtest,1);
correct = 0;
total = 0;
for i=1:m
	if ytest(i,1) == 1
		if p(i,1) == 1
			correct = correct + 1;
			total = total + 1;
		else
			total = total + 1;
		endif
	endif
endfor
fprintf('\nSesitivity =\t%f\n',(correct/total)*100);
pause;

%% =============specificity=====================================
% specifity is the percentage of negative samples that are 
% recognized correctly

correct = 0;
total = 0;
for i=1:m
	if ytest(i,1) == 0
		if p(i,1) == 0
			correct = correct + 1;
			total = total + 1;
		else
			total = total + 1;
		endif
	endif
endfor
fprintf('\nSpecificity =\t%f\n',(correct/total)*100);
pause;


%% ============================================================

fprintf('\n\nPredicting on user data...\n\n');
M = csvread('userdata');
[rows, cols] = size(M);
X = M(:, 2:cols);
p = svmPredict(model, X);
fprintf('\n\nClasses of user data are as follows:\t');
for i=1:rows
    if p(i,1) == 1
        fprintf('%d\t',1);
    else
        fprintf('%d\t',0);
    endif
endfor
fprintf('\n\n');
pause;
