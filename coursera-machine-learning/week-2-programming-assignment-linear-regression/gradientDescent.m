function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % alpha is the learning rate constant
    % order of X is [mx2] where the first column is ones
    % y is a vector of order [mx1]
        
    % X*theta gives a vector of order [mx1]
    % X*(X*theta - y) gives a vector of order [2x1]
    % theta is a vector of order [2x1]
    theta = theta - alpha * (1 / m) * (X') * (X * theta - y);



    % ============================================================

    % Save the cost J in every iteration    
    % Using J_history we can plot the 3d graph of various J values with theta
    J_history(iter) = computeCost(X, y, theta);

end

end
