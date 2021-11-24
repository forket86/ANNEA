clear all;
clc;

rng(10);

addpath('./Utils/');
addpath('./Functions/');

%% Income markov chain generation
T=5000;   % with more data things become more stable but was harder to see the V-shape pattern actually
z = zeros(1,T);
precision=1e-4;

% Number of states
sigma_eps=sqrt(0.0005);
rho=0.8; tauchen_mu=0.0;

% Simulate the AR(1) process
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

%% Steady-State values

KSS=(alpha*beta*Z_mean/(1-beta*(1-delta)))^(1/(1-alpha));
CSS=Z_mean*KSS^alpha+(1-delta)*KSS-KSS;
% c_init = CSS*z/1.5;
% k_init = KSS*z/1.5;
kss = @(z) [(1-beta*(1-delta))./(beta*alpha.*z)].^(1/(alpha-1));
css = @(k,z) z.*(k.^alpha)-delta*k;

k_init = kss(z);  c_init = css(k_init,z);
RHS=du(c_init(2:end)).*(alpha.*z(2:end).*k_init(2:end).^(alpha-1)+1-delta);

%input = [k_init(1:end-1)*mean(z);z(1:end-1)];
input = [repmat(mean(z),1,T-1);z(1:end-1)];
output = RHS;

neurons_grd=1:35;
for i =1:length(neurons_grd)
    
    %     rng(10) ideally would need to fix the seed because I think the
    %     initial network weights are random so need to make sure they are the
    %     same. Fixing the seed and the number of neurons guarrantees that the
    %     initial network weights are the same. in the paper we didn't do it
    %     because we only realized after.
    
    neurons=neurons_grd(i);
    
    net = feedforwardnet(neurons);
    net.trainFcn = 'trainlm'; %traingda traingdx good alternative
    net.trainParam.goal=0;
    net.trainParam.mu=0.01;
    net.trainParam.mu_dec=0.01;
    net.trainParam.mu_inc=10;
    net.trainParam.min_grad = 1e-7;
    net.trainParam.epochs = 1000;
    net.trainParam.showWindow=0; %Default: 1
    net.layers{1}.transferFcn = 'tansig';
    
    
    [net,tr] = train(net ,input,output);
    
    perf(i)  = tr.best_perf;%: 0.0074
    vperf(i) = tr.best_vperf;%: 0.0080
    tperf(i) = tr.best_tperf;%: 0.0088
    
end

blue = [0,90,191]/255;
red = [196,0,100]/255;

mvaw_vperf=10;
mvaw_perf=10;
xf=[neurons_grd(1) neurons_grd(end)];

linewidth=1.5;
linewidthmvaw=2.5;

figure
hold on
plot(neurons_grd,movmean(perf,mvaw_perf),'linewidth',linewidthmvaw,'Color', blue)
plot(neurons_grd,movmean(vperf,mvaw_vperf),'--','linewidth',linewidthmvaw,'Color', red)
h=scatter(neurons_grd,perf,'linewidth',linewidth)
h.MarkerEdgeColor = blue;
h=scatter(neurons_grd,vperf,'linewidth',linewidth)
h.MarkerEdgeColor = red;
xlabel('Number of Hidden Units','fontsize',14)
ylabel('RMSE','fontsize',14)
xlim(xf)
ylim([8*1e-4 10*1e-4]);
grid on
x = [11 13 13 11];
y = [8*1e-4 8*1e-4 10*1e-4 10*1e-4];
p=patch(x,y,'k','EdgeColor','none');
set(p,'FaceAlpha',0.1);
%print('-depsc','neurons_select_illustrative')