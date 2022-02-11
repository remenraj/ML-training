function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% X is a matrix of order [mx3] where the first column is ones

% theta is a vector of order [3x1] where 3 is the number of columns in X (after 
% adding column of ones)

% hypothesis function of order [mx1]
h = sigmoid(X * theta); 



% y is a vector of order [mx1]
% log(h) takes log of every element in h
% y'*log(h) is scalar value
% (1-y), (1-h) subtracts 1 from every element in y, h respectively
% (1-y)'*log(1-h) is a scalar value
% scalar cost function in logistic regression
J = (1 / m) * ((-y' * log(h)) - (1 - y)' * log(1 - h)); % scalar


% X' is a matrix of order [3xm]
% (h-y) is a vector of order [mx1]
% grad/gradient is a vector of theta or order [3x1]
grad = (1 / m) * (X' * (h - y));



% =============================================================

end
