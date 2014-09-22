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
potentialC = [0.01, 0.03, 0.1, 1, 3, 10 30];
potentialSigma = [0.01 0.03 0.1, 1, 3, 10, 30];

potentialCNum = length(potentialC);
potentialSigmaNum = length(potentialSigma);

misclassifications = zeros(potentialCNum, potentialSigmaNum);

for i = 1:potentialCNum
	for j = 1:potentialSigmaNum
		model = svmTrain(X, y, potentialC(i),...
						@(x1, x2) gaussianKernel(x1, x2, potentialSigma(j)));
		predictions = svmPredict(model, Xval);
		misclassifications(i, j) = mean(double(predictions ~= yval));
	end
end

% Plot a contour figure to visualize the changing trend of errors.
% Use log10 of potentialC and potentialSigma to make the change more
% well-distributed.

logPotentialC = log10(potentialC);
logPotentialSigma = log10(potentialSigma);

[c h] = contourf(logPotentialC, logPotentialSigma, misclassifications, 20);
clabel(c, h, 'fontsize', 20);

% Return the C and sigma value with the least errors.
potentialCIndex = 1;
potentialSigmaIndex = 1;
minElement = misclassifications(potentialCIndex, potentialSigmaIndex);
for i = 1:size(misclassifications, 1)
	for j = 1:size(misclassifications, 2)
		if misclassifications(i, j) < minElement
			minElement = misclassifications(i, j);
			potentialCIndex = i;
			potentialSigmaIndex = j;
		end
	end
end

C = potentialC(potentialCIndex);
sigma = potentialSigma(potentialSigmaIndex);

% =========================================================================

end
