%%%% Fast Polynomial Kernel Learning for Massive Data Classification %%%%%%
%%%% Authors: Jinshan Zeng, Minrun Wu, Shaobo Lin, and Ding-Xuan Zhou %%%%%
%%%% Edited by Jinshan Zeng on Feb 01, 2019. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [beta,trainerr,testerr,traintime,testtime]=FPC(xtr,ytr,xte,yte,s,cx)
%%% Input %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% xtr         -- inputs of training samples
%%% ytr         -- labels of training samples
%%% xte         -- inputs of testing samples
%%% yte         -- labels of testing samples
%%% s           -- algorithmic parameter: the order of the used polynomial (generally, an integer less than 10)
%%% cx          -- algorithmic parameter: center points in X-space (generally, taking the center points as the first nc columns of the polynomial kernel matrix)
%%% gamma       -- augmented lagrangian parameter (default:1)
%%% alpha       -- proximal parameter (defualt: 1), mainly to overcome the ill-condedness of the induced kernel matrix and improve the numerical stability
%%% MaxIter     -- the default number of maximal iterations: 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Output %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% beta        -- the parameters of model
%%% testerr     -- test error (misclassification rate)
%%% trainerr    -- training error (misclassification rate)
%%% testtime    -- test time (seconds)
%%% traintime   -- training time (seconds)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



start_cpu_time = cputime;
% the default parameters for FPC
alpha = 1; % proximal parameter
gamma = 1; % the augmented parameter
MaxIter = 5; % the maximal iterations

ATrain = PolynomialKerMat(xtr,cx,s); % the associated matrix via polynomial kernel using training samples
ATest = PolynomialKerMat(xte,cx,s); % the associated matrix via polynomial kernel using test samples
[m,n]=size(ATrain);

% calculate the inverse and restore
% tempA = (gamma*(ATrain'*ATrain)+alpha*eye(n))\eye(n);
% some faster ways for implementation
tempA = decomposition(gamma*(ATrain'*ATrain)+alpha*eye(n)); % way 1
% or by cholesky
% tempA = decomposition(gamma*(ATrain'*ATrain)+alpha*eye(n),'chol'); % way 2

% initialization
u0=zeros(n,1);
v0=ytr;
w0=zeros(m,1);

iter = 1;
while iter<=MaxIter
%     ut = tempA*(alpha*u0+ATrain'*(gamma*v0-w0)); % update ut via original way
    ut = tempA\(alpha*u0+ATrain'*(gamma*v0-w0)); % update ut via faster way
    vt = hinge_prox(ytr,ATrain*ut+gamma^(-1)*w0,m*gamma); % update vt
    wt = w0+gamma*(ATrain*ut-vt); % update multiplier wt
    
    u0 = ut;
    v0 = vt;
    w0 = wt;
    iter = iter +1;
end
end_cpu_time = cputime;
traintime = end_cpu_time - start_cpu_time; % calculating the training time

beta = u0;
trainerr = sum((ytr~=mysign(ATrain*u0)))/size(ytr,1); % calculating test error

start_test_time = cputime;
testerr = sum((yte~=mysign(ATest*u0)))/size(yte,1); % calculating test error
end_test_time = cputime;
testtime = end_test_time - start_test_time;
end


%%% Constructing a matrix given the center points of kernel x, coefficients c and the order of polynomial kernel s.
function A = PolynomialKerMat(x,c,s)
% c: Polynomial kernel coefficient
% s: the order of Polynomial kernel
A = (ones(size(x,1),size(c,1))+x*c').^s;
end

function tempsign = mysign(u)
tempsign = (u>=0)-(u<0);
end

function z = hinge_prox(a,b,gamma)
% hinge_prox(a,b,gamma) = argmin_u max{0,1-a*u} + gamma/2 (u-b)^2
% a: m*1 vector
% b: m*1 vector
% gamma>0: parameter
% z: output of the proximal of hinge
% m = size(a,1);
tol = 1e-10;
z = b.*(a==0)+(b+gamma^(-1)*a).*(a~=0&a.*b<=1-gamma^(-1)*a.*a)+...
    (a~=0&a.*b>1-gamma^(-1)*a.*a&a.*b<1).*((a+tol).^(-1))+b.*(a~=0&a.*b>=1);
end
