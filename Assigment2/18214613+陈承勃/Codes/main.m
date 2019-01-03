function main(datapath, K, randomstate, Max_iteration, eps)
% datapath: path to import dataset
% K: int, number of classes
% randomstate: int number for random state
% Max_iteration: Maximum number for iteration
% eps: parameter to control early stop

%%% Import dataset
data = csvread(datapath);
[N, M] = size(data);

%%% Visualization of the dataset
figure(1)
plot(data(:,1), data(:,2), '.')
title('Scatter plot of the dataset')
xlabel('x1'), ylabel('x2')

%%% Training Run GMM_EM model
[pi_w, mu, sigma, gammas, lnL] = GMM_EM(data, K, randomstate, Max_iteration, eps);

%%% Print learned paramters
pi_w
mu
sigma

%%% Visualization
%% In-complete log-likelihood
figure(2)
plot(lnL)
title('In-complete log-likelihood during training')
xlabel('Iteration'), ylabel('log-likelihood')

%% Visualization of the responsibilities, i.e., \gamma(z_{nk})
n_sample = 500;
% setting random state
rand('state', 123);
sample_ids = randperm(N, n_sample);
sample_points = data(sample_ids, :);
sample_gammas = gammas(sample_ids, :);
% assign each sample to a class
[max_gammas ,labels] = max(sample_gammas, [], 2);
% setting colors for classes
colormap([1, 0, 0; 
      0, 1, 0; 
      0, 0, 1;
      0, 1, 1]); 
figure(3)
scatter(sample_points(:,1), sample_points(:,2), 20, labels, 'filled') 
title('Scatter plot of class assignments')
xlabel('x1'), ylabel('x2')