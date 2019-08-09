% -----------------------------------------------------------------
% Author's page:
% https://sites.google.com/view/alessandrovilla/
% -----------------------------------------------------------------
function MSE = MSEPolicy(ctilde,phic,policy,k_grd,z_grd)
MSE=0;
for k_iter=1:length(k_grd)
    for z_iter=1:length(z_grd)
        MSE=MSE+(ctilde(k_grd(k_iter),z_grd(z_iter),phic)-policy(k_grd(k_iter),z_grd(z_iter)))^2;
    end
end
end

