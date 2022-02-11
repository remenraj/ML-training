function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;

% size(X, 2) gives the number of columns
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% feature scaling is used to speed up gradient descent by having each of the 
% input values in roughly the same range.

% feature scaling involved dividing the input values by the range(max - min)
% i.e, dividing by (max-min) of the values

% mean normalization involves subtracting the average value for an input variable
% from the values for that input variable resulting in a new average value for 
% the input variable of just zero.

% X is a matrix of order [mx2]

% mu is a matrix of order [1x2]
% mean(X) computes mean columnwise and forms a matrix of order [1xcolumns of X]
mu = mean(X);

% sigma is a matrix of order [1x2]
% std(X) computes mean columnwise and forms a matrix of order [1xcolumns of X]
sigma = std(X);

% feature normalization
% (X - mu) subtracts all the rows of X with mu
% ./ divides every element of the matrix by sigma
X_norm = (X - mu) ./ sigma;



% ============================================================

end
