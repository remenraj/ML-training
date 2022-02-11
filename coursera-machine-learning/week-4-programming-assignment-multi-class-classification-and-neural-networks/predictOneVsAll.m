function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

% m is the number of training set. Here m = 5000
m = size(X, 1);
% num_labels = No. of output classifier (Here, it is 10)
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 

% p is a vector of [mx1]=[5000x1]
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
% before order of X was [mxn] here [5000x400]
% now the order of X becomes [mx(n+1)] here [5000x401]
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% X is a matrix of order [mx(n+1)]=[5000x401] where the first column is ones
% all_theta is a matrix of order [num_labels x (input layer size + 1)]=[10x401]

% all_theta = 10 x 401 = num_labels x (input_layer_size+1) 
%                      == num_labels x (no_of_features+1)

% predict is a matrix of order [mx(num_labels)]=[5000x10]
predict = sigmoid(X*all_theta'); 


% max(predict, [], 2) will give two values: max value in the row and its index

% prob: probability of predicted output. maximum value/probability is stored in prob
% p: predicted output (index). index of maximum value is stored in p

% returns maximum element in each row  ==
% == max. probability and its index for each input image
% prob is a scalar index
% p is a vector of order [mx1]
[prob, p] = max(predict, [], 2);




% =========================================================================


end
