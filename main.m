% -----------------------------------------------------------------
% Source (Please cite the paper if you use this algorithm in other applications):
% Machine Learning Projection Methods for Macro-finance Models
% https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3209934
%
% Author's page:
% https://sites.google.com/view/alessandrovilla/
%
% Purpose of the code:
% Show as illustrative example how to solve the neoclassical investment model 
% using an ANN-based expectations algorithm and compare its solution to the 
% normal PEA (compare also to the analytical solution)
% 
% Inputs:
% Specify model parameters and algorithm precision
% 
% Outputs:
% if debugVerbose=1 graph that show the convergence path
% Graph that compare the different dynamics
% -----------------------------------------------------------------

clear all;
clc;

rng(10);

addpath('./Utils/');
addpath('./Functions/');

%% Income markov chain generation
infinityHorizon=1000;
z = zeros(1,infinityHorizon);
precision=1e-4;

%Number of states
n=10;
sigma_eps=0.25;
rho=0.8;
m=2; % num of standard deviations
tauchen_mu=0.0;
%Tauchen discretization
[z_grd, P_z] = tauchendisc(tauchen_mu,sigma_eps,rho,m,n);
%Show prediction error neural network step by step
debugVerbose=true;

for i =1:length(z_grd)
    for s = 1:length(z_grd)
        thresholds(i,s) = sum(P_z(i,1:s));
    end
end
z(1) = z_grd(round(end/2));
for t=1:infinityHorizon
    i = dsearchn(z_grd,z(t));
    U=rand(1);
    compare=thresholds(i,1);
    x=1;
    for x=1:n
        if x==1 & U<thresholds(i,1)
            z(t+1) = z_grd(x);
            break;
        end
        
        if U<thresholds(i,x) & U>thresholds(i,x-1) & x>1
            z(t+1) = z_grd(x);
            break;
        end
        
    end
end
z(end)=[];
Z_mean=mean(z);

%% Parameters
beta=0.95;%discount factor
alpha=0.36;%capital share
explore=0.3;% %dev from SS for grid
delta=1.00;

%Approximation parameters
order=3;
cross=true;
K=n;% #nodes in the grids

u=@(c) log(c);
du=@(c) c.^(-1);
duinv=@(u) u.^(-1);
ddu=@(c) -c.^(-2);

%% Steady-State values
KSS=(alpha*beta*Z_mean/(1-beta*(1-delta)))^(1/(1-alpha)); LOW_K=KSS*(1-explore); HIGH_K=KSS*(1+explore);
CSS=Z_mean*KSS^alpha+(1-delta)*KSS-KSS; LOW_C=CSS*(1-explore); HIGH_C=CSS*(1+explore);

%% Analytical solution: benchmark
%The analytical solution is good when delta=1, utility is ln and production
%is CD
Kprime_analytical=@(k,z) beta.*alpha.*z.*k.^alpha;
C_analytical=@(k,z) (1-delta).*k+z.*k.^alpha-Kprime_analytical(k,z);

k_Analytical(1)=KSS;
for t=1:infinityHorizon
    c_Analytical(t)=C_analytical(k_Analytical(t),z(t));
    k_Analytical(t+1)=z(t).*k_Analytical(t).^alpha+(1-delta)*k_Analytical(t)-c_Analytical(t);
end
k_Analytical(end)=[];

%% Policy Projection
k_grd=chebySpaceAle(LOW_K,HIGH_K,K)';
ctilde=@(k,z,phic) chebyAle(order,[k z],phic,[LOW_K z_grd(1)],[HIGH_K z_grd(end)],LOW_C,HIGH_C,cross);
phic=zeros(order+1,3);
[phic MSEInit flag] = fminunc(@(phic) MSEPolicy(ctilde,phic,C_analytical,k_grd,z_grd),phic);
MSEInit=MSEInit/(length(k_grd)*length(z_grd));

display(['Num. of c params: ' num2str(prod(size(phic))-1)])

phicinit=phic;
obj = @(phic)  Euler(beta,alpha,delta,du,ctilde,phic,k_grd,z_grd,P_z);
[phic MSEEuler flag] = fminunc(@(phic) sum(sum(obj(phic).^2)),phic);
MSEEuler=MSEEuler/(length(k_grd)*length(z_grd));
MSEOut = MSEPolicy(ctilde,phic,C_analytical,k_grd,z_grd);
MSEOut = MSEOut/(length(k_grd)*length(z_grd));

k_Proj(1)=KSS;
for t=1:infinityHorizon
    c_Proj(t)=ctilde(k_Proj(t),z(t),phic);
    k_Proj(t+1)=z(t).*k_Proj(t).^alpha+(1-delta)*k_Proj(t)-c_Proj(t);
end
k_Proj(end)=[];

%% PEA
Expectation=@(k,z,phiE) exp(phiE(1)+phiE(2)*log(k)+phiE(3)*log(z));
phiE=zeros(1,3);

ESS=du(CSS).*(alpha.*Z_mean.*KSS.^(alpha-1)+1-delta);
phiE(1)=log(ESS);
RHS=du(c_Proj(2:end)).*(alpha.*z(2:end).*k_Proj(2:end).^(alpha-1)+1-delta);
[phiE MSEPea flag] = fminunc(@(phiE) sum((Expectation(k_Proj(1:end-1)',z(1:end-1)',phiE)-RHS').^2),phiE);
MSEPea=MSEPea/(infinityHorizon-1);

dampening=0.1;
for iter=1:1e+10
    k_PEA(1)=KSS;
    for t=1:infinityHorizon
        ERHS_PEA(t)=Expectation(k_PEA(t),z(t),phiE);
        c_PEA(t)=duinv(beta*ERHS_PEA(t));
        k_PEA(t+1)=z(t).*k_PEA(t).^alpha+(1-delta)*k_PEA(t)-c_PEA(t);
    end
    k_PEA(end)=[];
    RHS=du(c_PEA(2:end)).*(alpha.*z(2:end).*k_PEA(2:end).^(alpha-1)+1-delta);
    [phiENew MSEPea flag] = fminunc(@(phiE) sum((Expectation(k_PEA(1:end-1)',z(1:end-1)',phiE)-RHS').^2),phiE);
    
    for k_iter=1:length(k_grd)
        for z_iter=1:length(z_grd)
            ExpeF(k_iter,z_iter)=Expectation(k_grd(k_iter),z_grd(z_iter),phiE);
        end
    end
    
    MSEPea(iter)=MSEPea/(infinityHorizon-1);
    predictionMSEPea(iter)=sum((Expectation(k_PEA(1:end-1)',z(1:end-1)',phiE)-RHS').^2)/(infinityHorizon-1);
    phiE=phiENew*dampening+(1-dampening)*phiE;
    if iter>1 && (predictionMSEPea(iter)-predictionMSEPea(iter-1))<precision
        break;
    end
end

%% ANN-based Expectations Algorithm
neurons=6;
net = feedforwardnet(neurons);
net.trainFcn = 'trainlm'; %traingda traingdx good alternative
net.trainParam.goal=0.002; %Default: 0
net.trainParam.mu=0.001;
net.trainParam.mu_dec=0.1;
net.trainParam.mu_inc=10;
net.trainParam.min_grad = 1e-5; %Default: 1e-5
net.trainParam.epochs = 1000; %Default: 1000
net.trainParam.showWindow=0; %Default: 1
net.layers{1}.transferFcn = 'tansig';
RHS=du(c_Proj(2:end)).*(alpha.*z(2:end).*k_Proj(2:end).^(alpha-1)+1-delta);
[net,tr] = train(net,[k_Proj(1:end-1);z(1:end-1)],RHS);

for iter=1:1e+10
    
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
    
    if debugVerbose && iter>1
        for k_iter=1:length(k_grd)
            for z_iter=1:length(z_grd)
                ExpeF(k_iter,z_iter)=SimNetwork([k_grd(k_iter);z_grd(z_iter)],netparams);
            end
        end
        
        figure(1)
        subplot(2,1,1)
        yyaxis left
        plot(predictionMSEANN,'Color', [0,0.4470,0.7410],'linewidth',1.5)
        hold on
        yyaxis right
        plot(abs(diff(predictionMSEANN)),'--','Color',[0.8500,0.3250,0.0980],'linewidth',1.5)
        xlabel('Iteration')
        legend('Residuals','Convergence of Residuals')
        subplot(2,1,2)
        contourf(z_grd,k_grd,ExpeF,'ShowText','on');
        xlabel('k')
        ylabel('z')
        title('Forecasted Expectation')
        drawnow
    end
    
    k_ANNPEA(1)=KSS;
    tobefilter=1;
    for t=1:infinityHorizon
        ERHS_ANNPEA(t)=SimNetwork([k_ANNPEA(t);z(t)],netparams);
        c_ANNPEA(t)=duinv(beta*ERHS_ANNPEA(t));
        k_ANNPEA(t+1)=z(t).*k_ANNPEA(t).^alpha+(1-delta)*k_ANNPEA(t)-c_ANNPEA(t);
    end
    k_ANNPEA(end)=[];
    RHS=du(c_ANNPEA(2:end)).*(alpha.*z(2:end).*k_ANNPEA(2:end).^(alpha-1)+1-delta);
    for i=1:20
        [net,y,e] = adapt(net,[k_ANNPEA(1:end-1);z(1:end-1)],RHS);
    end
    %mse(e)
    predictionMSEANN(iter)=sum((SimNetwork([k_ANNPEA(1:end-1);z(1:end-1)],netparams)'-RHS').^2)/(infinityHorizon-1);
    
    % difference with analytical
    ANN_minus_analytical(iter)=sum((k_ANNPEA'-k_Analytical').^2)/(infinityHorizon-1);
    %
    if iter>1 && abs(predictionMSEANN(iter)-predictionMSEANN(iter-1))<1e-4
        break;
    end
    k_ANNPEA_store(iter,:) = k_ANNPEA;
end

initPeriod=10;
endPeriod=30;
hline = 3;
subplot(3,1,1)
plot(z(initPeriod:endPeriod),'k', 'linewidth', hline)
xlabel('time')
ylabel('z')

subplot(3,1,2)
plot(c_Analytical(initPeriod:endPeriod),'k', 'linewidth', hline)
hold on
plot(c_Proj(initPeriod:endPeriod),'r-.', 'linewidth', hline-1)
hold on
plot(c_PEA(initPeriod:endPeriod),'mo', 'linewidth', hline-1)
hold on
plot(c_ANNPEA(initPeriod:endPeriod),'b-.', 'linewidth', hline-1)
xlabel('time')
ylabel('Consumption')
legend('Analytical','Projection','PEA','ANN-PEA')

subplot(3,1,3)
plot(k_Analytical(initPeriod:endPeriod),'k', 'linewidth', hline)
hold on
plot(k_Proj(initPeriod:endPeriod),'r-.', 'linewidth', hline-1)
hold on
plot(k_PEA(initPeriod:endPeriod),'mo', 'linewidth', hline-1)
hold on
plot(k_ANNPEA(initPeriod:endPeriod),'b-.', 'linewidth', hline-1)
xlabel('time')
ylabel('Capital')
legend('Analytical','Projection','PEA','ANN-Expectations Algorithm')


%% analytical and ANN Expectation Algorithm only
figure(2)
initPeriod=10;
endPeriod=50;
hline = 2;
subplot(3,1,1)
plot(z(initPeriod:endPeriod),'k', 'linewidth', hline)
xlabel('time')
ylabel('z')

subplot(3,1,2)
plot(c_Analytical(initPeriod:endPeriod),'k', 'linewidth', hline)
hold on
plot(c_ANNPEA(initPeriod:endPeriod),'b-.', 'linewidth', hline-1)
xlabel('time')
ylabel('Consumption')
legend('Analytical','ANN-Expectations Algorithm')

subplot(3,1,3)
plot(k_Analytical(initPeriod:endPeriod),'k', 'linewidth', hline)
hold on
plot(k_ANNPEA(initPeriod:endPeriod),'b-.', 'linewidth', hline-1)
xlabel('time')
ylabel('Capital')
legend('Analytical','ANN-Expectations Algorithm')

subplot(1,2,1)
plot(predictionMSEANN, 'b-.','linewidth', hline-1)
xlabel('iteration')
ylabel('MSE')
legend('prediction error')

subplot(1,2,2)
plot(predictionMSEANN, 'b-.','linewidth', hline-1)
xlabel('iteration')
ylabel('MSE')
legend('prediction error')


