function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples

J = 1/(2 * m) * sum(((X * theta) .- y).^2);

% Compute Cost for single varaible:
% J = 1/(2 * m) * sum(((X * theta) - y) .^ 2);
% only difference here is element-wise subtraction of Y

end
