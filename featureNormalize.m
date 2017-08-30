function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X); % 1xn
sigma = std(X); % 1xn
ones_vector = ones(size(X), 1);
mu_matrix = ones_vector * mu;
std_matrix = ones_vector * sigma;
X_norm = X - mu_matrix;
X_norm = X_norm ./ std_matrix;

end
