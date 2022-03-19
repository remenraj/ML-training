function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
% here centroids is a matrix of order [3x2], meaning there are 3 centroids and 
% each row is the location of the centroid
% centroids = K x no. of features = 3 x 2
% size(centroids,1) = 3
K = size(centroids, 1);

% You need to return the following variables correctly.
% X is a matrix of order [300x2]
% idx is a vector of order [300x1]
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% looping through examples
for i = 1 : size(X,1)
  
  % creating a temporary vector to store the distance from the current example to the centroid
  temp = zeros(K,1);
  
  % looping through clusters
  for j = 1:K
    
      temp(j) = sqrt(sum( (X(i,:) - centroids(j,:) ).^2 ));
      
  end
  
  % saving the shortest distance to the idx vector
  [~,idx(i)] = min(temp);

end


% =============================================================

end

