function [wt_probs, value] = incompl_lnL(probs, pi_w)
% probs: Nxk
% pi_w: Kx1
% ---return---
% wt_probs: Nx1

wt_probs = probs * pi_w;  % Nx1
value = sum(log(wt_probs));
end
    