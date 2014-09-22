function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add bias neural
% Size of X is m X 401
X = [ones(m, 1), X];

% Compute second layer of neural network.

% Size of a2 is m X 25.
% Size of X is m X 401.
% Size of Theta1 is 25 * 401
z2 = X * Theta1';
a2 = sigmoid(z2);
% Add bias neural
a2 = [ones(m, 1), a2];
% Now size of a2 is m X 26

% Compute third(final) layer of neural network.
% Now size of a2 is m X 26
% Size of Theta2 is 10 * 26
z3 = a2 * Theta2';
a3 = sigmoid(z3);
% Size of a3 is m X 10

% Compute cost for all classes and sum them together.
for i = 1:num_labels
	J +=  mean((-((y == i) .* log(a3(:,i)))) - (1 - (y == i)) .* log(1 - a3(:,i)));
end

% To reduce the computional cost of lambda / (2 * m), use a tmp sum to get the
% sum of regularization, and do lambda / (2 * m) only once, then add it to J.
sumTmp = 0;
for i = 1:hidden_layer_size
	for j = 2:input_layer_size + 1
		sumTmp += Theta1(i, j)^2;
	end
end

for i = 1:num_labels
	for j = 2:hidden_layer_size + 1
		sumTmp += Theta2(i, j)^2;
	end
end

J += sumTmp * lambda / (2 * m);

% Backpropabation algorithm.
big_delta1 = zeros(size(Theta1));
big_delta2 = zeros(size(Theta2));
for i = 1:m
	% Forward propabation.
	z2 = Theta1 * X(i,:)';
	%Now Size of z2 is 25 X 1
	a2 = sigmoid(z2);
	a2 = [1; a2];
	%Now Size of a2 is 26 X 1
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);
	%Now Size of a3 is 10 X 1
	
	% Backward probabation.
	y_i = zeros(num_labels, 1);
	y_i(y(i)) = 1;
	% little_delta3 10 X 1.
	little_delta3 = a3 - y_i;
	% Theta2 10 X 26
	% Since biased node is not used, exclude it using 2:end.
	% little_delta2 25 X 1.
	little_delta2 = (Theta2' * little_delta3)(2:end) .* sigmoidGradient(z2);

	% little_delta3 10 X 1.
	% a2 26 X 1
	big_delta2 += little_delta3 * a2';
	% Size of little_delta2 is 26 X 1.
	% Size of X(i,:) is 1 X 401
	big_delta1 += little_delta2 * X(i,:);
end

% Compute deriative of bias theta exclusively.
Theta1_grad(:,1) = big_delta1(:,1) / m;
Theta2_grad(:,1) = big_delta2(:,1) / m;

% Compute other thetas.
Theta1_grad(:, 2:end) = (big_delta1(:, 2:end) + lambda * Theta1(:, 2:end)) / m;
Theta2_grad(:, 2:end) = (big_delta2(:, 2:end) + lambda * Theta2(:, 2:end)) / m;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
