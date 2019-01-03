function sigma = update_sigma(X, Nk, gammas, mu)
% X: NxM  where N denotes the number of observations
% Nk: 1xK
% gammas: NxK
% mu: MxK
% ---return---
% sigma: MxMxK
[N, M] = size(X);
K = size(Nk, 2);

sigma = cat(3, eye(M), eye(M), eye(M), eye(M));
gammas = gammas ./ repmat(Nk, N, 1);  % NxK
for k=1:K
    mu_k = repmat(mu(:,k)', N, 1);  % NxM
    gammas_k = repmat(gammas(:,k)', M, 1);  % MxN
    sigma(:,:,k) = gammas_k .* (X - mu_k)' * (X - mu_k);  % MxM
end

end