function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% pinv() is used returns the Moore-Penrose pseudo inverse of a matrix using 
% Singular value. The inv() function returns the inverse of the matrix. 
% The pinv() function is useful when your matrix is non-invertible(singular matrix) 
% or Determinant of that Matrix =0.


% X is a matrix of order [mx3] where the first column is ones
% y is a vector of order [mx1]

% pinv(X'*X) is a matrix of order [3x3]
% X'*y is a vector of order [3x1]

% theta is a vector of order [3x1] 
theta = pinv(X' * X) * X' * y

% -------------------------------------------------------------


% ============================================================

end
