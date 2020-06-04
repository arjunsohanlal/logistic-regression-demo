function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Evaluating hypothesis vector for 'm' examples
h = sigmoid(X * theta);		% m x 1

% Evaluating error terms for 'm' examples
errorList = h - y;			% m x 1

% Evaluating cost function
J = (1/m) * (((-y)'*log(h)) - ((1-y)'*log(1-h))) + ((lambda/(2*m)) * (theta(2:end)' * theta(2:end)));

% Evaluating gradients/partial derivatives
grad = (1/m) * X' * errorList;

% Adding regularization terms to gradient
reg_grad = [0; (lambda/m) * theta(2:end)];
grad = grad + reg_grad;

end
