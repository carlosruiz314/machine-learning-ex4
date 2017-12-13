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

% Add bias term to input x
X_train = [ones(m,1) X];

% Initialize capital delta error matrices:
Delta1 = zeros(hidden_layer_size, input_layer_size + 1);
Delta2 = zeros(num_labels, hidden_layer_size + 1);

% Forward propagation. Store the different activation values for back-propagation
a2 = [ones(m,1) sigmoid(X_train * Theta1')];
pred = sigmoid(a2 * Theta2');


% Loop through training examples and classes to calculate the neural network's cost at the current values of Theta1 and Theta2
for i=1:m
yVal = [];
	for k=1:num_labels
		% Create a binary vector for all the class labels
		yVal = [yVal; (k==y(i))];
	end
	J = J + (-1/m * (yVal' * log(pred(i,:))' + (1-yVal)' * log(1-pred(i,:))'));

	% Backpropagation:
	% Calculate delta error vectors:
	delta3 = pred(i,:)' - yVal;
	delta2 = (Theta2(:,2:end)' * delta3) .* a2(i,2:end)' .* (1-a2(i,2:end)');
	% Calculate capital delta error matrices:
	Delta1 += delta2 * X_train(i,:);
	Delta2 += delta3 * a2(i,:);
end

% Add the regularization term to the cost function
J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Store the gradients for each layer:
Theta1_grad = 1/m * (Delta1 + [zeros(hidden_layer_size,1) lambda*Theta1(:,2:end)]);
Theta2_grad = 1/m * (Delta2 + [zeros(num_labels,1) lambda*Theta2(:,2:end)]);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
