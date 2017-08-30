function [theta, J_history] = gradientDescent(x, y, theta, alpha, num_iters)
% Performs gradient descent to find theta, also keeps a matrix of the cost for each iteration

m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

h = x * theta;
theta = theta - alpha * (1/m) * transpose(x) * (h - y);

    % Save the cost J in every iteration
    J_history(iter) = computeCost(x, y, theta);

end

end
