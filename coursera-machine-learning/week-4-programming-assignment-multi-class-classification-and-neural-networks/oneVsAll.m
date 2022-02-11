function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
% m is the number of training set. Here m = 5000
m = size(X, 1);
% n is the total number of pixels. 20x20= 400
n = size(X, 2);

% You need to return the following variables correctly 
% here num_labels = 10. There are 10 classes here (0-9)
% and n = 400
% order of all_theta is [10x401]

% each row of all_theta corresponds to the learned logistic regression  
% parameters for one class.
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% Variable 'X' contains data in dimension (5000 * 400). 
% 5000 = Total no. of training examples, 
% 400 = 400 pixels / training sample (digit image) = Total number of features
% After adding unit vector to X, order of X becomes 5000 x 401

% y (5000*1) is an array of labels i.e. it contains actual 
% digit names (y==c) will return a vector with values 0 or 1. 
% 1 at places where y==c
% a = [1 2 3 4 5 6 7 8 9 10]   b = 3
% a == b gives [0 0 1 0 0 0 0 0 0 0]


% 't' is passed as dummy parameter which is initialized with 'initial_theta' first
% then subsequent values are choosen by fmincg 
% [Note: Its not a builtin function like fminunc]


# Set Initial theta
% initial_theta is a vector of order [401x1]
initial_theta = zeros(n + 1, 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Run fmincg to ottain the optimal theta
% This function will return theta and the cost

% fmincg will consider all training data having label c
% (1-10 note0 is mapped to 10) and find the optimal theta vector for it 
% (Classifying white pixels with gray pixels). 
% same process is repeated for other classes

% each row of all_theta corresponds to the learned logistic regression  
% parameters for one class.
% You can do this with a “for”-loop
% from 1 to K(num_labels), training each classifier independently.

% When training the classifier for class k ? {1, ..., K}, you will want a m-
% dimensional vector of labels y, where yj ? 0, 1 indicates whether the j-th
% training instance belongs to class k (yj = 1), or if it belongs to a different
% class (yj = 0). You may find logical arrays helpful for this task.

for c = 1:num_labels
  % all_theta(c,:) replaces the c'th row of all_theta
  all_theta(c,:) = ...
           fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                   initial_theta, options);


end
% =========================================================================


end
