function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

% prediction_error is a matrix used to store all the error combinations 
% arising with different values of C and sigma
% It has 3 columns and length(C_vec) * length(sigma_vec) rows
% First element in a row is the error after prediction with particular C and sigma
% Second element in a row is C and third element is sigma
prediction_error = zeros(length(C_vec) * length(sigma_vec), 3);

% initiating the row number to be 1
row_number = 1;

for i = 1 : length(C_vec)
  % current C value from C_vec
  C_test = C_vec(i);
  
  for j = 1 : length(sigma_vec)
    % current sigma value from sigma_vec
    sigma_test = sigma_vec(j);
    
    % training the model with current C and sigma values
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    
    % predecting with the trained model
    predictions = svmPredict(model, Xval);
    
    % finding the error between predictions and yval
    predicted_error = mean(double(predictions ~= yval));
    
    % replacing the current row of predicted_error with the predicted error and its corresponding C, sigma 
    prediction_error(row_number,:) = [predicted_error, C_test, sigma_test];
    
    # increasing the row number by 1
    row_number = row_number + 1;
  end
end

% Sorting prediction_error in ascending order of error (first column)
sorted_result = sortrows(prediction_error, 1);
  
% C and sigma corresponding to min(prediction_error)
C = sorted_result(1,2);
sigma = sorted_result(1,3);



% =========================================================================

end
