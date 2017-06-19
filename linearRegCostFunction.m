function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


J1 = sum((X*theta-y).^2/(2*m));
J2 = (0.5*lambda/m)*sum(theta(2:end).^2);
J =J1 + J2;

gradient = (1/m) * (X'*(X*theta-y));
gradient1 = gradient(1,:);
gradient2 = gradient(2:end,:)+(lambda/m)*theta(2:end,:);

grad= [gradient1;gradient2];










% =========================================================================

grad = grad(:);

end
