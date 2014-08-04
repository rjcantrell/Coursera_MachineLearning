function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    
    % NON VECTORIZED VERSION
    %rateT1 = 0;
    %rateT2 = 0;
    %for i=1:m
    %    rateT1 = rateT1 + (theta(1)*X(i,1) + theta(2)*X(i,2) - y(i))*X(i,1);
    %    rateT2 = rateT2 + (theta(1)*X(i,1) + theta(2)*X(i,2) - y(i))*X(i,2);
    %end;
    %tempT1 = theta(1) - alpha * (1/m) * rateT1;
    %tempT2 = theta(2) - alpha * (1/m) * rateT2;
    %theta = [tempT1; tempT2];
    
    % VECTORIZED VERSION
    theta = theta' - alpha * (1/m) * ((X * theta) - y)' * X;
    theta = [theta(1); theta(2)];
    %fprintf('Iteration #%d: thetas %4.4f %4.4f has cost of %4.4f\n', iter, theta(1), theta(2), computeCost(X,y,theta));
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
