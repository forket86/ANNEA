function [s, prob_tauchen] = tauchendisc(mu,sigma_eps,rho,m,ns)

x1 = mu - m*sigma_eps/(sqrt(1-rho^2));
xn = mu + m*sigma_eps/(sqrt(1-rho^2));
grid = linspace(x1,xn,ns)';
w = grid(3)-grid(2);


for i=1:length(grid)
    for j=2:length(grid)-1
        prob_tauchen(i,j) = normcdf((grid(j)+w/2-(1-rho)*mu-rho*grid(i))/sigma_eps,0,1) - normcdf((grid(j)-w/2-(1-rho)*mu-rho*grid(i))/sigma_eps,0,1);
    end
end

for i=1:length(grid)
    prob_tauchen(i,1) =  normcdf((grid(1)+w/2-(1-rho)*mu-rho*grid(i))/sigma_eps,0,1);
    prob_tauchen(i,length(grid)) = 1 - normcdf((grid(end)-w/2-(1-rho)*mu-rho*grid(i))/sigma_eps ,0,1);
end
%s=grid;
s=exp(grid);