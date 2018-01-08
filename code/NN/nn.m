%% Initialization
clear ; close all; clc

%% Setup the parameters of the neural network
input_layer_size  = 22;   % there are 22 features 
hidden_layer_size = 50;   % 25 hidden units
num_labels = 2;           % there are only two categories to classify 

%% =========== Part 1: Loading data ============

% Load Training Data
fprintf('Loading data ...\n')
M = csvread('SPECT.train');
[rows, cols] = size(M);
X = M(:, 2:cols);
y = M(:, 1);
fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 6: Initializing Parameters ================
%  implementing a function to initialize the weights of the neural network randomly
%  (randInitializeWeights.m)

fprintf('\nRandomly initializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 7: Implement Backpropagation ===============

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 8: Implement Regularization ===============
%  Implement regularization with the cost and gradient.

fprintf('\nChecking Backpropagation (with Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 8: Training NN ===================

fprintf('\nTraining Neural Network... \n')

% MaxIter can be altered to see how more training helps 
options = optimset('MaxIter', 400);

% different values for lambda can be set here, 
lambda = 1;

% "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Implement Prediction (training set) =================
% predict and test accuracy of classifier on training set
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


%% ================= Implement Prediction (test set)=================
% predict and test accuracy of classifier on training set

% load the test set
M = csvread('SPECT.test');
[rows, cols] = size(M);
X = M(:, 2:cols);
y = M(:, 1);

pred = predict(Theta1, Theta2, X);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y)) * 100);

