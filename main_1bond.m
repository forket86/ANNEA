% -----------------------------------------------------------------
% Source (Please cite the paper if you use this algorithm in other applications):
% Machine Learning Projection Methods for Macro-finance Models
% https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3209934
%
% Author's pages:
% https://sites.google.com/view/alessandrovilla/
% https://sites.google.com/view/vytautas-valaitis/home
%
% Purpose of the code:
% Solve the Ramsey taxation problem with one non-state-contingent bond
% using the ANN-based expectations algorithm
%
% -----------------------------------------------------------------

clear all;
clc;

rng(10);

addpath('./Utils/');

%% Government expenditure AR(1) parameters
infinityHorizon = 5000;
const           = 0.004161764705882;
variance        = 1.109550560082420e-05;

phi1=0.95; % Persistence of the g process
phi2=0;
model = arima('Constant',const,'AR',{phi1,phi2},'Variance',variance);
g=simulate(model,infinityHorizon)';

g=g(1:infinityHorizon);
gMean = mean(g);
gmean = mean(g)*(1-phi1);

g = max(min(g,0.35*0.33),0.15*0.33);  % restrict g to be between 15% and 35% of the GDP.


%% algorithm parameters

cutInit = 15;
cutEnd  = 0;
burnin=2;
startingbound=0.01;

BoundStep = 0.005;
outofbounfloops=0;
numboutofbound=700;
tolerance = 1e-11; tolerance_mean = 1e-4;

%% Variable declaration

beta=0.96;
A=1;
Bmax=(1/3)*1.5;
Bmin=-(1/3)*1.5;
maxIterations=1000;
numIterationToFinish=maxIterations;

errorHist1=zeros(1,maxIterations);
errorHist2=zeros(1,maxIterations);
errorHist3=zeros(1,maxIterations);

warning('off')

opt = optimset('Display','off');
phi=90;


%% Model parameters and Functions
gamma=1.5;
eta=1.8;%4;
B =  2.872727272727273; %;2.7310;

u=@(c) c.^(1-gamma)./(1-gamma);
du=@(c) c.^(-gamma);
ddu=@(c) -gamma*c.^(-gamma-1);

v=@(l) B*l.^(1-eta)./(1-eta);
dv=@(l) B*l.^(-eta);
ddv=@(l) -eta*B*l.^(-eta-1);


%% Initialization sequence for the neural network
c=zeros(1,infinityHorizon);
% lambda - recursive multiplyer.
lambda = zeros(1,infinityHorizon);
b = zeros(1,infinityHorizon);
zeta_low = zeros(1,infinityHorizon);
zeta_up  = zeros(1,infinityHorizon);

b(1+1:end) = 2*(movmean(g(1+1:end),40)/2-mean(g/2));

Blower = -startingbound;
Bupper =  startingbound;

meanc = 1-mean(g)-2/3;
for t=1+1:infinityHorizon
    income= @(ct) -b(t-1)*du(meanc*0.9+0.1*ct)-g(t)*du(ct)+(du(ct)-dv(A-ct-g(t)))*(g(t)+ct)+b(t)*beta*du(meanc);
    [c(t), fvalBudget]=fsolve(income,(A-g(t))/3,opt);
    
    lambda(t) = (dv(A-c(t)-g(t))-ddu(c(t))*(lambda(t-1))*b(t-1)-du(c(t)))/(ddu(c(t))*c(t)+du(c(t))+ddv(A-c(t)-g(t))*(c(t)+g(t))-dv(A-c(t)-g(t))-ddu(c(t))*b(t-1));
    
end


for t=1+1:infinityHorizon-1
    if b(t)<Blower
        zeta_low(t) = phi*(Blower-b(t))+log(1+phi*(Blower-b(t)));   %    +beta^N*du(c(t+N))/du(c(t))*lambda(t) - beta^N*du(c(t+N))/du(c(t))*lambda(t+1);
    elseif b(t)>Bupper
        zeta_up(t)  = phi*(b(t)-Bupper)+log(1+phi*(b(t)-Bupper)); % +beta^N*du(c(t+N))/du(c(t))*lambda(t+1) - beta^N*du(c(t+N))/du(c(t))*lambda(t);
    else
        zeta_up(t) = 0; zeta_low(t) = 0;
    end
    
end

output1=du(c(burnin+1:end)).*lambda(burnin+1:end-1+1);
output2=du(c(burnin+1:end));


%% Create and initialize a single hidden layer neural network
neurons = 10;

net = feedforwardnet(neurons);
net.trainFcn = 'traingda'; %traingda traingdx good alternative
net.trainParam.lr=0.01;
net.trainParam.showWindow=0; %Default: 1
net.layers{1}.transferFcn = 'tansig';


% Create matrix of inputs to train the neural network
string1='';
string1='X= [g(burnin:end-1)';
for z=1:1
    string1=[string1 ';lambda(burnin-' int2str(z) ' :end-1-' int2str(z) ');b(burnin-' int2str(z) ':end-1-' int2str(z) ');zeta_up(burnin-' int2str(z) ':end-1- ' int2str(z) ');zeta_low(burnin-' int2str(z) ':end-1- ' int2str(z) ')'];
end
string1=[string1 '];'];
eval(string1);

% Use the initialization sequence to initially train the neural network
[net,tr] = train(net ,X(:,cutInit:end-cutEnd),[output1(cutInit:end-cutEnd);output2(cutInit:end-cutEnd);]);

string='';
string='state=@(gp,Lambda,b,zeta_up,zeta_low,t) [gp';
for i=1:1
    string=[string ';Lambda(t-' int2str(i) ');b(t-' int2str(i) ');zeta_up(t-' int2str(i) ');zeta_low(t-' int2str(i) ')'];
end
string=[string '];'];
eval(string);




%% Simulate

Bound(1) = startingbound;
i=1;

disp(['Starting Bound: ' num2str(Bound(i))])
FirstReverse=false;

while (Bound(i)<Bmax || outofbounfloops<numboutofbound)
    
    % Retrieve current neural network parameters
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
    
    % Set Maliar bounds
    Blower = -Bound(i);
    Bupper =  Bound(i);
    
    % Initialize simulation sequence
    lambda=zeros(1,infinityHorizon);
    c=zeros(1,infinityHorizon);
    b=zeros(1,infinityHorizon);
    zeta_up = zeros(1,infinityHorizon);
    zeta_low = zeros(1,infinityHorizon);
    
    % Simulate the sequence by solving the FOCs, given the neural network prediction
    for t=1+1:infinityHorizon
        
        stateCurr=state(g(t),lambda,b,zeta_up,zeta_low,t);
        
        temp_output = SimNetwork(stateCurr,netparams);
        
        Edulambda=temp_output(1);
        Edu=temp_output(2);
        
        Edulambda_rec(t)  = Edulambda;
        Edu_rec(t)        = Edu;
        
        % FOC wrt b_{t+1}
        lambda(t)=Edulambda/Edu;
        lambda(t) = min(lambda(t),10);
        
        
        L = @(ct) -[lambda(t)*(ddu(ct)*ct+du(ct)+ddv(A-ct-g(t))*(ct+g(t))-dv(A-ct-g(t))-ddu(ct)*b(t-1)) - (dv(A-ct-g(t))-ddu(ct)*(lambda(t-1))*b(t-1)-du(ct))].^2;
        
        [c(t), fval(t)] = golden(L,0.22,0.29);
        
        
        % Government Budget constraint
        
        b(t)=(du(c(t))*b(t-1)+g(t)*du(c(t))-(du(c(t))-dv(A-c(t)-g(t)))*(g(t)+c(t)))/(beta*Edu);
        
        if b(t)<max(Bmin, Blower)
            b(t)=max(Bmin, Blower);
            % Government Budget constraint
            
            BC=@(ct) -[b(t)-(b(t-1)*beta*du(ct)-g(t)*du(ct)+(du(ct)-dv(A-ct-g(t))).*(g(t)+ct))/(beta*Edu)].^2;
            
            [c(t), fval(t)] = golden(BC,0.22,0.30);
            
            
            % FOC wrt c_{t}
            lambda(t) = (dv(A-c(t)-g(t))-ddu(c(t))*(lambda(t-1))*b(t-1)-du(c(t)))/(ddu(c(t))*c(t)+du(c(t))+ddv(A-c(t)-g(t))*(c(t)+g(t))-dv(A-c(t)-g(t))-ddu(c(t))*b(t-1));
            
            zeta_low(t) = -beta*Edu*lambda(t) + beta*Edulambda;
        elseif b(t)>min(Bmax, Bupper)
            b(t)=min(Bmax, Bupper);
            % Government Budget constraint
            BC=@(ct) -[b(t)-(b(t-1)*beta*du(ct)-g(t)*du(ct)+(du(ct)-dv(A-ct-g(t))).*(g(t)+ct))/(beta*Edu)].^2;
            
            [c(t), fval(t)] = golden(BC,0.22,0.30);
            
            lambda(t) = (dv(A-c(t)-g(t))-ddu(c(t))*(lambda(t-1))*b(t-1)-du(c(t)))/(ddu(c(t))*c(t)+du(c(t))+ddv(A-c(t)-g(t))*(c(t)+g(t))-dv(A-c(t)-g(t))-ddu(c(t))*b(t-1));
            
            zeta_up(t) = -beta*Edulambda + beta*Edu*lambda(t);
        end
    end
    i
    
    %Prepare data for regression. In the regression use X's only starting
    %from N+1. so the first forecasted period t+N will be N+N+1
    
    info(:,i) = mean(fval')';
    info(:,i)
    
    if i>2 && (max(abs(info(:,i)))>tolerance_mean || min(isreal(info(:,i)))==0)
        if FirstReverse==false
            i=i-1;
            net=netStore{i};
            Bound(i)=(9*Bound(i)+1*Bound(i-1))/10;
            FirstReverse=true;
            disp(['Errors are high. Reverting to previous net with ' num2str(Bound(i))])
            continue;
        else
            Bound(i)=(9*Bound(i)+1*Bound(i-1))/10;
            disp(['Errors are high. Eroding the bound to ' num2str(Bound(i))])
            continue;
        end
    end
    
    
    string1='';
    
    string1='X= [g(burnin:end-1)';
    for z=1:1
        string1=[string1 ';lambda(burnin-' int2str(z) ' :end-1-' int2str(z) ');b(burnin-' int2str(z) ':end-1-' int2str(z) ');zeta_up(burnin-' int2str(z) ':end-1- ' int2str(z) ');zeta_low(burnin-' int2str(z) ':end-1-' int2str(z) ')'];
    end
    string1=[string1 '];'];
    
    eval(string1);
    
    output1=du(c(burnin+1:end)).*lambda(burnin+1:end-1+1);
    output2=du(c(burnin+1:end));
    
    %Check error and update weights
    for t=burnin:infinityHorizon
        errorHist1_temp(t-burnin+1) = Edulambda_rec(t);
        errorHist2_temp(t-burnin+1) = Edu_rec(t);
    end
    
    netStore{i} = net;
    
    
    prevnet = net;
    
    [net,tr] = train(net,X(:,cutInit:end-cutEnd),[output1(cutInit:end-cutEnd);output2(cutInit:end-cutEnd)]);
    
    distance(i) = norm(net.IW{1}-prevnet.IW{1}) + norm(net.LW{2,1}-prevnet.LW{2,1}) + norm(net.b{1}-prevnet.b{1}) + norm(net.b{2}-prevnet.b{2});
    
    
    errorHist1(i)=sum(abs(errorHist1_temp(cutInit:end-1-cutEnd) - output1(cutInit:end-cutEnd)))/(infinityHorizon-cutInit-cutEnd);
    errorHist2(i)=sum(abs(errorHist2_temp(cutInit:end-1-cutEnd) - output2(cutInit:end-cutEnd)))/(infinityHorizon-cutInit-cutEnd);
    
    
    if i>3
        dist_b(i) = sum(abs(b-b_old));
        errorhist(i) = errorHist1(i) + errorHist2(i);
        
        if (abs(errorhist(i)-errorhist(i-1))<1e-6 && max(dist_b(i-2:i))<1e-5 && Bound(i)==Bmax)
            numIterationToFinish=i;
            break;
        end
    end
    
    
    b_old = b;
    
    cStore(i,:) = c;
    bStore(i,:) = b;
    
    i=i+1;
    
    if FirstReverse==false
        Bound(i)=Bound(i-1)+BoundStep;
    else
        Bound(i)=Bound(i-1)+(Bound(i-1)-Bound(i-2));
    end
    
    if Bound(i)>=Bmax && outofbounfloops<numboutofbound
        Bound(i)=Bmax;
        outofbounfloops=outofbounfloops+1;
    end
    disp(['Iteration: ' num2str(i) ' New Bound: ' num2str(Bound(i))])
    FirstReverse=false;
    
end

%% Plot Final Result
displayPeriods=900;

labor=c(1:displayPeriods)+g(1:displayPeriods);
leisure=A-labor;
tau=1-dv(leisure)./du(c(1:displayPeriods));

start = 2;
X=start:displayPeriods;


EstimateSymbol='k-';
RealizedSymbol='b'; %k

figure
subplot(2,1,1)
plot(X,errorHist1_temp(start:displayPeriods),EstimateSymbol,X,output1(start:displayPeriods),RealizedSymbol)
title('Fitting for $E_t[u_{c,t+1}\lambda_{t+1}]$','Interpreter','Latex')
legend('Neural-estimate','Realized')
xlabel('Time (t)')

subplot(2,1,2)
plot(X,errorHist2_temp(start:displayPeriods),EstimateSymbol,X,output2(start:displayPeriods),RealizedSymbol)
title('Fitting for $E_t[u_{c,t+1}]$','Interpreter','Latex')
legend('Neural-estimate','Realized')
xlabel('Time (t)')

hline = 2;
figure
plot(X,b(1+1:displayPeriods),'b-', 'linewidth', hline,'Color', [0.0, 90.0/256, 191.0/256])
hold on
plot(X,g(1+1:displayPeriods),'r', 'linewidth', hline,'Color', [196/256, 0.0, 100.0/256])
legend('Debt: b_t','Gvt expenditure','Interpreter','Latex')
xlabel('Time (t)')









