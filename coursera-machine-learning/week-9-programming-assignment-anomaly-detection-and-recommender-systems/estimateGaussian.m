function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
% here m = 1000, number of examples
% here n = 11, number of features
[m, n] = size(X);

% You should return these values correctly
% mu and sigma2(variance) are vector of order [n x 1] = [11x1]
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


mu = ( (1/m) * sum(X) )';

sigma2 = ( (1/m) * sum((X - mu').^2) )';


% =============================================================


end
