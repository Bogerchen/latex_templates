function mu = update_mu(X, Nk, gammas)
% X: NxM  where N denotes the number of observations
% Nk: 1xK
% gammas: NxK
% ---return---
% mu: MxK
N = size(gammas, 1);

gammas = gammas ./ repmat(Nk, N, 1);  % NxK
mu = X' * gammas;  % MxK

end
