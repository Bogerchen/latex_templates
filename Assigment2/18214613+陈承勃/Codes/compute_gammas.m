function gammas = compute_gammas(probs, wt_probs, pi_w)
% probs: NxK
% wt_probs: Nx1
% pi_w: Kx1
% ---return---
% gammas: NxK
[N, K] = size(probs);
pi_w = pi_w';
probs = probs .* repmat(pi_w, N, 1);  % NxK
gammas = probs ./ repmat(wt_probs', K, 1)'; % NxK

end
