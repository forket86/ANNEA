% -----------------------------------------------------------------
% Source (Please cite the paper if you use this algorithm in other applications):
% A Machine Learning Projection Method for Macro-finance Models
% https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3209934
%
% Authors:
% https://www.alessandrotenzinvilla.com/
% https://sites.google.com/view/vytautas-valaitis/home
%
% Purpose of the code:
% Show as illustrative example how to solve the neoclassical investment model
% using an ANN-based Expectations Algorithm and compare its solution to projection and PEA
%
% -----------------------------------------------------------------

clear all;
clc;
close all;
opt_fminunc = optimoptions('fminunc','Display','none');

% The number of neurons for the neural network is selected with seed equal
% to 10. If you change the see, you would need to run the neural selection
% code with the specified seed.
rng(10);

addpath('./Utils/');
addpath('./Functions/');

%% Income markov chain generation
T=5000;
z = zeros(1,T);
precision=1e-4;

% Number of states
sigma_eps=sqrt(0.0005); rho=0.8; tauchen_mu=0.0;
% Tauchen discretization
n=15; m=2.5;
[z_grd, P_z] = tauchendisc(tauchen_mu,sigma_eps,rho,m,n);

e2_sim=sigma_eps*randn(1,T);
z(1) = tauchen_mu/(1-rho);
for t=2:T
    z(t) = tauchen_mu + rho*(z(t-1))+ e2_sim(t);
end
z  = exp(z);
Z_mean=mean(z);

%% Parameters
% Discount factor, capital share, depreciation
beta=0.95; alpha=0.36; delta=0.1;

% Utility
u=@(c) log(c); du=@(c) c.^(-1); duinv=@(u) u.^(-1); ddu=@(c) -c.^(-2);

% Parameters for the projection method
explore=0.25;% Dev from SS for grid
order=3; % Polynomial order
cross=true; % Cross-terms
K=n;% #nodes in the grids


%% Steady-State values
KSS=(alpha*beta*Z_mean/(1-beta*(1-delta)))^(1/(1-alpha)); LOW_K=KSS*(1-explore); HIGH_K=KSS*(1+explore);
CSS=Z_mean*KSS^alpha+(1-delta)*KSS-KSS; LOW_C=CSS*(1-explore); HIGH_C=CSS*(1+explore);

%% SS initialization
kss = @(z) [(1-beta*(1-delta))./(beta*alpha.*z)].^(1/(alpha-1));
css = @(k,z) z.*(k.^alpha)-delta*k;

k_init = kss(z);
c_init = css(k_init,z);


%% Policy Projection
k_grd=chebySpaceAle(LOW_K,HIGH_K,K)';
ctilde=@(k,z,phic) chebyAle(order,[k z],phic,[LOW_K z_grd(1)],[HIGH_K z_grd(end)],LOW_C,HIGH_C,cross);
phic=zeros(order+1,3);
[phic MSEInit flag] = fminunc(@(phic) MSEPolicy(ctilde,phic,css,k_grd,z_grd),phic,opt_fminunc);
MSEInit=MSEInit/(length(k_grd)*length(z_grd));

display(['Num. of c params: ' num2str(prod(size(phic))-1)])

phicinit=phic;
obj = @(phic)  Euler(beta,alpha,delta,du,ctilde,phic,k_grd,z_grd,P_z);
[phic MSEEuler flag] = fminunc(@(phic) sum(sum(obj(phic).^2)),phic,opt_fminunc);
MSEEuler=MSEEuler/(length(k_grd)*length(z_grd));

k_Proj(1)=KSS;
for t=1:T
    c_Proj(t)=ctilde(k_Proj(t),z(t),phic);
    k_Proj(t+1)=z(t).*k_Proj(t).^alpha+(1-delta)*k_Proj(t)-c_Proj(t);
end
k_Proj(end)=[];

%% PEA
Expectation=@(k,z,phiE) exp(phiE(1)+phiE(2)*log(k)+phiE(3)*log(z));
phiE=zeros(1,3);

ESS=du(CSS).*(alpha.*Z_mean.*KSS.^(alpha-1)+1-delta);
phiE(1)=log(ESS);
RHS=du(c_init(2:end)).*(alpha.*z(2:end).*k_init(2:end).^(alpha-1)+1-delta);
[phiE MSEPea flag] = fminunc(@(phiE) sum((Expectation(k_init(1:end-1)',z(1:end-1)',phiE)-RHS').^2),phiE,opt_fminunc);
MSEPea=MSEPea/(T-1);

dampening = 0.1;
maxit = 1e5; k_old = k_init;
for iter=1:maxit
    k_PEA(1)=KSS;
    for t=1:T
        ERHS_PEA(t)=Expectation(k_PEA(t),z(t),phiE);
        c_PEA(t)=duinv(beta*ERHS_PEA(t));
        k_PEA(t+1)=z(t).*k_PEA(t).^alpha+(1-delta)*k_PEA(t)-c_PEA(t);
    end
    k_PEA(end)=[];
    RHS=du(c_PEA(2:end)).*(alpha.*z(2:end).*k_PEA(2:end).^(alpha-1)+1-delta);
    [phiENew MSEPea flag] = fminunc(@(phiE) sum((Expectation(k_PEA(1:end-1)',z(1:end-1)',phiE)-RHS').^2),phiE,opt_fminunc);
    
    for k_iter=1:length(k_grd)
        for z_iter=1:length(z_grd)
            ExpeF(k_iter,z_iter)=Expectation(k_grd(k_iter),z_grd(z_iter),phiE);
        end
    end
    
    MSEPea(iter)=MSEPea/(T-1);
    predictionMSEPea(iter)=sum((Expectation(k_PEA(1:end-1)',z(1:end-1)',phiE)-RHS').^2)/(T-1);
    phiE=phiENew*dampening+(1-dampening)*phiE;
    % Check convergence
    crit(iter) = max(abs(k_old-k_PEA));
    if iter>5 && crit(iter)<4e-4
        break;
    end
    k_old = k_PEA;
end

%% ANN-based Expectations Algorithm
RHS=du(c_init(2:end)).*(alpha.*z(2:end).*k_init(2:end).^(alpha-1)+1-delta);
net = feedforwardnet(12);
net.trainParam.showWindow=0; %Default: 1
[net,tr] = train(net,[k_init(1:end-1);z(1:end-1)],RHS);

maxit = 1e4; k_old = k_init; crit = [];
for iter=1:maxit
    
    netparams = get_netparams(net);
    
    k_ANNPEA(1)=KSS;
    tobefilter=1;
    for t=1:T
        ERHS_ANNPEA(t)=SimNetwork([k_ANNPEA(t);z(t)],netparams);
        c_ANNPEA(t)=duinv(beta*ERHS_ANNPEA(t));
        k_ANNPEA(t+1)=z(t).*k_ANNPEA(t).^alpha+(1-delta)*k_ANNPEA(t)-c_ANNPEA(t);
    end
    k_ANNPEA(end)=[];
    RHS=du(c_ANNPEA(2:end)).*(alpha.*z(2:end).*k_ANNPEA(2:end).^(alpha-1)+1-delta);
    [net,y,e] = train(net,[k_ANNPEA(1:end-1);z(1:end-1)],RHS);

    % Check convergence
    crit(iter) = max(abs(k_old-k_ANNPEA));
    if iter>5 && crit(iter)<4e-4
        break;
    end
    k_old = k_ANNPEA;
    
    % Mse(e)
    predictionMSEANN(iter)=sum((SimNetwork([k_ANNPEA(1:end-1);z(1:end-1)],netparams)'-RHS').^2)/(T-1);
    % ANN Weights Convergence
    distance(iter) = norm(net.IW{1}-netparams.IW) + norm(net.LW{2,1}-netparams.LW) + norm(net.b{1}-netparams.B1) + norm(net.b{2}-netparams.b2);
    
    
end

initPeriod=10;
endPeriod=500;
hline = 1.5;

figure
plot(c_Proj(initPeriod:endPeriod),'-.','color',[0, 0.4470, 0.7410], 'linewidth', hline-1)
hold on
plot(c_PEA(initPeriod:endPeriod),'--','color',[0.6350 0.0780 0.1840], 'linewidth', hline-1)
hold on
plot(c_ANNPEA(initPeriod:endPeriod),'k','linewidth', hline)
xlabel('Time')
ylabel('Consumption')
legend('Projection','PEA','ANN-based Expectation Algorithm')
legend boxoff
grid on

figure
plot(k_Proj(initPeriod:endPeriod),'-.','color',[0, 0.4470, 0.7410], 'linewidth', hline-1)
hold on
plot(k_PEA(initPeriod:endPeriod),'--','color',[0.6350 0.0780 0.1840], 'linewidth', hline-1)
hold on
plot(k_ANNPEA(initPeriod:endPeriod),'k','linewidth', hline)
xlabel('Time')
ylabel('Capital')
legend('Projection','PEA','ANN-based Expectation Algorithm')
legend boxoff
grid on

figure
plot(predictionMSEANN(1:end),'linewidth',hline)
title('ANN-based Expectation Algorithm  - Prediction Error')
ylabel('Error')
xlabel('Iteration')
grid on

figure
plot(distance(1:end),'linewidth',hline)
title('ANN-based Expectation Algorithm - Change of Network Weights')
ylabel('Distance')
xlabel('Iteration')
grid on

function netparams = get_netparams(net)

netparams.ymaxin=net.inputs{1}.processSettings{1}.ymax;
netparams.yminin=net.inputs{1}.processSettings{1}.ymin;
netparams.xmaxin=net.inputs{1}.processSettings{1}.xmax;
netparams.xminin=net.inputs{1}.processSettings{1}.xmin;

netparams.ymaxout=net.outputs{2}.processSettings{1}.ymax;
netparams.yminout=net.outputs{2}.processSettings{1}.ymin;
netparams.xmaxout=net.outputs{2}.processSettings{1}.xmax;
netparams.xminout=net.outputs{2}.processSettings{1}.xmin;

netparams.IW = net.IW{1,1};
netparams.B1 = net.b{1};
netparams.LW = net.LW{2,1};
netparams.b2 = net.b{2};

end


