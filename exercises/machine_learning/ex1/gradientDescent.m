
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
%repeat until convergence: {
%   ��_0 := ��_0 - ��*(1/m)*sum(h_��(x_i)-y_i)
%   ��_1 := ��_1 - ��*(1/m)*sum((h_��(x_i)-y_i)x_i)
%}
%
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % X is a n��2 matrix; 
    % theta is a 2��1 matrix;
    % h is a n��2*2��1=n��1 matrix
    h = X * theta; 
    % y is a n��1 matrix
    % h-y=n��1 matrix
    % (h-y)'=1��n matrix
    % X(:,j) is a n��1 matrix
    % (h-y)'*X(:,j)=1��n*n��1 = a realNumber = sum((h_��(x_i)-y_i)x_i)
    theta(1) = theta(1) - alpha * (1 / m) * (h - y)'*X(:,1); % X(:, 1) = [1,1???1]
    theta(2) = theta(2) - alpha * (1 / m) * (h - y)'*X(:,2);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
