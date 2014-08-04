function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
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

possible_Cs = [0.01 0.03 0.1 0.3 1 3 10 30];
possible_sigmas = [0.01 0.03 0.1 0.3 1 3 10 30];
errors = zeros(length(possible_Cs), length(possible_sigmas));

for i=1:length(possible_Cs)
    for j=1:length(possible_sigmas)
        
        % determine model based on C(i) using symTrain.m
        model = svmTrain(X, y, possible_Cs(i), @(x1, x2) gaussianKernel(x1, x2, possible_sigmas(j)));
        
        % determine prediction using model as input to svmPredict.m
        predictions = svmPredict(model, Xval);
        
        % calculate error - mean(double(predictions ~= yval)
        errors(i,j) = mean(double(predictions ~= yval));
    end
end

% find indices of minimum value in our box o' errors
[I,J] = find(errors == min(errors(:)) )
C = possible_Cs(I);
sigma = possible_sigmas(J);

% =========================================================================

end
