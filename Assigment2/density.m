function probs = density(X, pi_w, mu, sigma)
% X: NxM  where N denotes the number of observations
% pi_w: Kx1
% mu: MxK
% sigma: MxMxK
% ---return---
% probs: NxK
K = size(pi_w, 1);
[N, M] = size(X);

probs = zeros(K, N);     
for k=1:K
    probs(k,:) = mvnpdf(X, mu(:,k)', sigma(:,:,k));  % vector of dimension Nx1
end
probs = probs';

end