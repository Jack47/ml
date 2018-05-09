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

% three layer
% Theta1 25x401
% Theta2 10x26
a1 = [ones(m, 1) X]; % [ m, 401]
z2 = a1*Theta1'; % [m, 25]
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2*Theta2'; % [m, 10]
a3 = sigmoid(z3);

sums = 0;
for k = 1:num_labels,
  yk = y == k; % [m, 1]
  hk = a3(:, k); % [m, 1]
  % just like logistic regression, and sum all
  % the cost in different classification
  sums += yk'*log(hk) + (1-yk)'*log(1-hk);
end

% exclude first column (bais unit)
theta1 = Theta1(:, 2:end);
theta2 = Theta2(:, 2:end);
regularized = lambda/(2*m) * (sum(sum(theta1.^2)) + sum(sum(theta2.^2)));
J = -1/m * sums + regularized;
% -------------------------------------------------------------

% =========================================================================

% Theta1 25x401
% Theta2 10x26
% So Delta* is the same dimension with coordinating Theta
Delta1 = zeros(hidden_layer_size, input_layer_size+1); % [25, 401]
Delta2 = zeros(num_labels, hidden_layer_size+1); % [10, 26]

for t = 1:m,
  % compute for t-th example
  % add bais unit
  a1 = [1 X(t, :)]; % 1x401
  z2 = a1*Theta1'; % [1 25]
  % add bais unit
  a2 = [1 sigmoid(z2)]; % [1 26]
  z3 = a2*Theta2'; % [1 10]
  a3 = sigmoid(z3); % [1 10]
  % compute delta3
  yt = zeros(1, num_labels); % [1, 10]
  yt(y(t)) = 1; % only column t is 1

  delta3 = a3-yt; % [1 10]
  % z2 is  [1 25], so delta3*Theta2 should be [1 25]
  delta2 = delta3*Theta2(:, 2:end).*sigmoidGradient(z2); % [1 25]

  Delta2 = Delta2 + delta3'*a2; % [10, 26]
  Delta1 = Delta1 + delta2'*a1; % [25, 401]
end

Theta1_grad = Delta1/m + lambda/m * Theta1; % j equal to zero
Theta1_grad(:, 1) = Delta1(:,1)/m; % j equal to zero

Theta2_grad = Delta2/m + lambda/m * Theta2;
Theta2_grad(:, 1) = Delta2(:, 1)/m;
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
