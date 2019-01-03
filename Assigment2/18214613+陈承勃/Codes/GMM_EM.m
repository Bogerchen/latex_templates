function [pi_w, mu, sigma, gammas, lnL] = GMM_EM(data, K, randomstate, Max_iteration, eps)
% data: NxM  where M denotes the dimension of each sample, N denotes sample size
% K: int, number of classes
% randomstate: int number for random state
% Max_iteration: Maximum number for iteration
% eps: parameter to control early stop
% ---returns---
% pi_w: \pi_k, Kx1 class weights  where K denotes number of classes
% mu: MxK mean for each class  
% sigma: MxM covariances for each class
% gammas: NxK responsibility for each sample from each class 
% lnL: vector, in-complete log-likelihood during training

[N, M] = size(data);
%%% Initialization
%% Initialize pi_w
rand('state', randomstate);
pi_w = rand(K, 1); 
pi_w = pi_w ./ sum(pi_w);  % Ensure \sum{pi_w} = 1
%% Initialize mu
rand('state', randomstate);
mu = randn(M, K);
%%Initialize sigma
rand('state', randomstate);
props = randsample(0.5:0.1:2.5, K);  % 1xK
sigma = zeros(M, M, K);
for k=1:K
    sigma(:,:,k) = repmat(props(1,k), M) .* eye(M);  % MxMxK all covariances are proportional to identity matrix in the case of singular covariance
end

%%% Initial value of in-complete data log-likehood
probs = density(data, pi_w, mu, sigma);  % NxK
[wt_probs, lnL_old] = incompl_lnL(probs, pi_w);
lnL(1) = lnL_old;

%%% EM iteration
iter = 1;
while (iter < Max_iteration)
    %% E-step
    gammas = compute_gammas(probs, wt_probs, pi_w);  % NxK

    %% M-step
    Nk = sum(gammas, 1);  % 1xK
    % update mu
    mu = update_mu(data, Nk, gammas);  % MxK
    % update sigma
    sigma = update_sigma(data, Nk, gammas, mu); 
    % update pi_w
    pi_w = Nk' ./ N;  % Kx1

    %% Evaluate the in-complete log-likelihood
    probs = density(data, pi_w, mu, sigma);  % NxK
    [wt_probs, lnL_new] = incompl_lnL(probs, pi_w);
    % check for convergence
    if (lnL_new - lnL_old < eps)
        lnL(iter+1) = lnL_new;
        break;
    else
        lnL(iter+1) = lnL_new;
        lnL_old = lnL_new;
        iter = iter + 1;
    end
end

end