function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad are "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshaping nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup variables
m = size(X, 1);
         
%============================================================
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feedforward the neural network and return the cost in the
% variable J. 
%
% Implementing the backpropagation algorithm to compute the gradients
% Theta1_grad and Theta2_grad. returning the partial derivatives of
% the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% Theta2_grad, respectively.
%
%
% Note: The vector y passed into the function is a vector of labels
% containing values from 1..K. We map this vector into a 
% binary vector of 1's and 0's to be used with the neural network
% cost function.
%
%
% Implementing regularization with the cost function and gradients.
%


a=ones(m,1);
X=[a X];
X1=X*Theta1';
X2=sigmoid(X1);
X2=[a X2];
X3=X2*Theta2';
X4=sigmoid(X3);

Y1=zeros(m,num_labels);
for i=1:m
	j=y(i,1);
	if j==0
		Y1(i,1)=1;
	endif
	if j==1
		Y1(i,2)=1;
	endif
endfor

T=ones(m,num_labels);
Z1=Y1.*log(X4);
Z2=(T-Y1).*log(T-X4);
Z=Z1+Z2;

Z3=zeros(m,1);
for i=1:num_labels
	Z3=Z3+Z(:,i);
endfor
p=0;
for i=1:m
	p=p+Z3(i,1);
endfor

J=(-p)/m;


th1=Theta1(:,2:input_layer_size+1);
th2=Theta2(:,2:hidden_layer_size+1);
th11=th1.^2;
th22=th2.^2;

Z3=zeros(hidden_layer_size,1);
Z4=zeros(num_labels,1);

for i=1:input_layer_size
	Z3=Z3+th11(:,i);
endfor
p=0;
for i=1:hidden_layer_size
	p=p+Z3(i,1);
endfor
th111=(p*lambda)/(2*m);


for i=1:hidden_layer_size
	Z4=Z4+th22(:,i);
endfor
p=0;
for i=1:num_labels
	p=p+Z4(i,1);
endfor
th222=(p*lambda)/(2*m);

J=J+th111+th222;




X1=[a X1];
DELTA1=zeros(size(Theta1,1),size(Theta1,2));
DELTA2=zeros(size(Theta2,1),size(Theta2,2));
for i=1:m
	delta3=X4'(:,i)-Y1'(:,i);
	delta2=(Theta2'*delta3).*sigmoidGradient(X1'(:,i));
	DELTA1=DELTA1+(delta2(2:hidden_layer_size+1,1)*X(i,:));
	DELTA2=DELTA2+(delta3*X2(i,:));
endfor



Theta1_grad = (1/m)*DELTA1;
Theta2_grad = (1/m)*DELTA2;
p1=zeros(size(Theta1,1),1);
p2=zeros(size(Theta2,1),1);


v1=(lambda/m)*Theta1(:,2:input_layer_size+1);
v2=(lambda/m)*Theta2(:,2:hidden_layer_size+1);

r1=[p1 v1];
r2=[p2 v2];


Theta1_grad = Theta1_grad + r1;
Theta2_grad = Theta2_grad + r2;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
