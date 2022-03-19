function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
% here K = 1
% X is a matrix of order [mxn] = [50x2]
% Z is a vector of order [mx1] = [50x1]
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%


% U is a matrix of order [nxn] = [2x2]. It is eigenvector.
% K is scalar.

U_reduce = U(:,[1:K]);   % [n x K] = [2x1]
Z = X * U_reduce;        % [m x k] = [50x1]


% =============================================================

end
