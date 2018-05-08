function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J =  (1/m) * sum(-y' * log(sigmoid(X*theta)) - (1 - y)'*log(1 - sigmoid(X*theta)));

%size(X,2) = 3

% At first step sigmoid(X*theta) returns a 100 x 1 vector such that all elements
% has value 0.5, since theta has only zeros in each column. When we substract y from sigmoid(X*theta)
% we obtain a 100 x 1 vector such that each element has vlaue +- 0.5 depends on the entries of y.
%
% repmat Form a block matrix of size M by N, with a copy of matrix A as each element, i.e.
% we obtain a 100 x 3 Matrix, where each row is a copy of sigmoid(X*theta) - y.


grad = (1/m) * sum(X.* repmat((sigmoid(X*theta) - y), 1, size(X,2)));






% =============================================================

end
