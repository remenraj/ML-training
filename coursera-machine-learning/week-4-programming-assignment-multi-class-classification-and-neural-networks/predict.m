function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
% here num_labels = 10
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% order of X is [mxn] = [5000x400]
% adding column of ones to X. order of a1 becomes [mx(n+1)]=[5000x401]
a1 = [ones(m, 1) X];

% order of a1 is [mx(n+1)]=[5000x401]
% order of Theta1 is [hidden layer size x (n+1)]=[25x401]
% Theta1:
% 1st row indicates: theta corresponding to all nodes from layer1 connecting to for 1st node of layer2
% 2nd row indicates: theta corresponding to all nodes from layer1 connecting to for 2nd node of layer2
% and
% 1st Column indicates: theta corresponding to node1 from layer1 to all nodes in layer2
% 2nd Column indicates: theta corresponding to node2 from layer1 to all nodes in layer2

% a1* Theta1' is a matrix of order [m x hidden layer size]=[5000x25]
z2 = a1 * Theta1';
a2 = sigmoid(z2);

% adding column of ones to a2
% order of a2 becomes [m x (hidden layer size + 1)]=[5000x26]
a2 = [ones(m, 1) a2];

% order of a2 is [m x (hidden layer size + 1)]=[5000x26]
% order of Theta2 is [total output classifiers x (hidden layer size + 1)]=[10x26]

% Theta2:
% 1st row indicates: theta corresponding to all nodes from layer2 connecting to for 1st node of layer3
% 2nd row indicates: theta corresponding to all nodes from layer2 connecting to for 2nd node of layer3
% and
% 1st Column indicates: theta corresponding to node1 from layer2 to all nodes in layer3
% 2nd Column indicates: theta corresponding to node2 from layer2 to all nodes in layer3
  
% order of a3 is [m x number of classifiers]=[5000x10]
z3 = a2 * Theta2';
a3 = sigmoid(z3);


% max(t2, [], 2) will give two values: max value in the row and its index

% prob: probability of predicted output. maximum value/probability is stored in prob
% p: predicted output (index). index of maximum value is stored in p

% returns maximum element in each row  ==
% == max. probability and its index for each input image
% prob is a scalar index
% p is a vector of order [mx1]
[prob, p] = max(a3, [], 2);


% =========================================================================


end
