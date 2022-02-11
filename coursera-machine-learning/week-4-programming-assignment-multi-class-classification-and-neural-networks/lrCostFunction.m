function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% regularization is done to address the issue of overfitting
% if lambda is too high, it may smooth out the function too much and cause
% underfitting.
% if lambda is loo low, regularization has no effect on the parameters and 
% overfitting is caused.


% X is a matrix of order [mx28] where the first column is ones. After feature 
% mapping the number of features increased from 2 to 27. i.e., mapped the 
% features into all polynomial terms of x1 and x2 up to the sixth power.

% theta is a vector of order [28x1] where 28 is the number of columns in X (after 
% feature mapping, Note: while feature mapping column of ones is added to X)

% hypothesis function of order [mx1]
h = sigmoid(X * theta); 



%%%%% scalar regularized cost function in logistic regression %%%%%%%%
% y is a vector of order [mx1]
% log(h) takes log of every element in h
% y'*log(h) is scalar value
% (1-y), (1-h) subtracts 1 from every element in y, h respectively
% (1-y)'*log(1-h) is a scalar value
J1 = (1/m) * ((-y' * log(h)) - (1 - y)'*log(1 - h));

% sum(theta(2:end).^2) takes the square of each elements in the vector except first
% and then the function sum, takes sum of all elements except first row(theta0)
% theta0 is not penalised or regularized. It gives a scalar value.
% .^2 takes the square of each element in the vector
% so the whole term square and sums starting from the second element till last
regularization_parameter = (lambda/(2*m)) * sum(theta(2:end).^2);

% J is the scalar reqularized cost function
J = J1 + regularization_parameter;


%%%%%% gradient vector of theta %%%%%%%
% h, y is a vector of order [mx1]
% first the term (h-y) is calculated and then it is multiplied by the first
% column of X which contains only ones.
% X is a vector of order [mx1] where is a unit vector
% grad is a vector of order [28x1]
% here the equation gives a scalar value
% grad(1) is the first element of grad vector where value is not reqularized
grad(1) = (1/m) * (X(:,1)'*(h - y));


% grad(2:end) calculates the rest of the values of grad vector
% X(:,2:end) is a matrix of order [mx27], first column of X is avoided
% (X(:,2:end)'*(h - y) gives a vector of order [27x1]
% theta(2:end) is a vector of order [27x1]
% grad(2:end) is a vector of order [27x1]
grad(2:end) = (1/m)* (X(:,2:end)'*(h - y)) + (lambda/m)*theta(2:end);



% =============================================================

grad = grad(:);

end
