function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
% y is a vector of order [mx1]=[12x1]
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
% grad is a vector of theta parameters
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X is a matrix of order [mx2] where the first column is ones
% theta is a vector of order [2x1]
% order of h_x becomes [mx1]
h_x = X * theta;

% scalar cost function without reqularization
J1 = (1/(2*m)) * sum((h_x - y).^2);

% scalar reqularization_parameter for the cost function
regularization_parameter = (lambda/(2*m)) * sum(theta(2:end).^2);

% scalar regularized cost function
J = J1 + regularization_parameter;

% h_x, y is a vector of order [mx1]
% X is a matrix of order [mx2] where the first column is ones
% X(:,1) gives the first column of ones of order [mx1]
% grad(1) is the first element of grad vector where value is not reqularized
grad(1) = (1/m) * X(:,1)'*(h_x - y);

% grad(2:end) calculates the rest of the values of grad vector
% X(:,2:end) is a matrix of order [mx1], first column of X is avoided
% (X(:,2:end)'*(h - y) gives a vector of order [27x1]
% theta(2:end) is a vector of order [27x1]
% grad(2:end) is a vector of order [27x1]
grad(2:end) = (1/m) * (X(:,2:end)'*(h_x - y)) + (lambda/m)*theta(2:end);



% =========================================================================

grad = grad(:);

end
