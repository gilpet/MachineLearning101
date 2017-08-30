fprintf('Loading Data ...\n')
data = load('filename.txt');
x = data(:, 1);
y = data(:, 2);
m = length(y); % number of examples

fprintf('Plotting Data ...\n')
plotData(x, y);

fprintf('Press enter to continue.\n');
pause;

% Compute Cost & Gradient Descent

x = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters at 0

% Some gradient descent settings
iterations = 1000;
alpha = 0.01;

fprintf('\nTesting cost function ...\n')
J = computeCost(x, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);

J = computeCost(x, y, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);

fprintf('Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')
theta = gradientDescent(x, y, theta, alpha, iterations);
fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);

fprintf('Press enter to continue.\n');
pause;

fprintf('Plotting Data ...\n')
hold on; % keep previous plot visible
plot(x(:,2), x*theta, '-')

predict1 = [1, 3.5] *theta;
fprintf('For x = 3.5, we predict a y value of %f\n',...
    predict1);
predict2 = [1, 7] * theta;
fprintf('For x = 7, we predict a y value of %f\n',...
    predict2);
