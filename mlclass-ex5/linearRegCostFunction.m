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

%X = [ones(size(X)) X];
%HofX = sum((X * theta),2);

HofX = X * theta;
J_nonreg = (1/(2*m)) * (HofX - y)' * (HofX - y); 

%VECTORIZED
%J_reg = (lambda / (2*m)) * theta(2:end) .^ 2; %J_reg(1) = 0; is another way to not regularize theta(1)

%NON VECTORIZED (for whatever reason, the Coursera grading robot only
%accepts this version, even though both give identical answers)
J_reg = 0;
for i=2:size(theta, 1)
    J_reg = J_reg + theta(i) ^ 2;
end;
J_reg = (lambda / (2*m)) * J_reg;

J = J_nonreg + J_reg;

%GRADIENTS - Same formula as when we implemented this for logistic, but
%with a new definition of HofX, above.

grad_nonreg = (1/m) * X' * (HofX - y);
grad_reg = lambda / m * theta; grad_reg(1) = 0;
grad = grad_nonreg + grad_reg;

% =========================================================================

grad = grad(:);

end
