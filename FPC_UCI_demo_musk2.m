%%%FPC for UCI data: musk2
close all;clear all; clc;
warning off all;
% UCI dataset experiments

%% Load dataset and data processing
load('clean2.mat');
% normalization: input [0,1], output {-1,1}
dataset = clean2; % file name of the dataset
% total number of samples 6598*167
% 2-class classification
% the last column is the target
% 50% for training, 25% for validation, 25% for test
% TrainDataSize =  3299;
% ValidDataSize = 1649
% TestDataSize = 1650;

% no. of negative labels: 5581
% no. of poisitve labels: 1017

%% data normalization:
%%%% input [0,1]
for i=1:size(dataset,2)-1
    if max(dataset(:,i))~=min(dataset(:,i))
        %          dataset(:,i)=((dataset(:,i)-min(dataset(:,i)))/(max(dataset(:,i))-min(dataset(:,i))))*2-1;
        dataset(:,i)=((dataset(:,i)-min(dataset(:,i)))/(max(dataset(:,i))-min(dataset(:,i))));
    else
        dataset(:,i)=0;
    end
end

% output {-1,1}
T=dataset(:,size(dataset,2));
if max(T)~=min(T)
    T=2*(T-min(T))/(max(T)-min(T))-1;
else
    T=ones(size(T))*0.5;
end
dataset(:,size(dataset,2))=T;

%% Generate the training and test samples
[DataSize,d] = size(dataset);
m0 = 3299; % number of training samples
m1 = 1649; % number of validation samples
m2 = DataSize-m0-m1; % Number of Test samples

trail = 20; % 20 trails
trainerr = zeros(trail,1); % recording training error
testerr = zeros(trail,1); % recording test error
traintime = zeros(trail,1); % recording training time
testtime = zeros(trail,1); % recording test time
best_s = zeros(trail,1); % recording best polynomial degree s
best_nc = zeros(trail,1); % recording best number of centers, i.e. nc = (s+d,s)

for i=1:trail
    %% Generate the training and test samples
    tempI = randperm(DataSize);
    TrainI = tempI(1:m0); % index set of training samples
    ValidI = tempI(m0+1:m0+m1); % index set of validation samples
    TestI = tempI(m0+m1+1:DataSize); % index set of testing samples
    % Training samples
    xtr = dataset(TrainI,1:d-1);
    ytr = dataset(TrainI,d);
    % Validation samples
    xvalid = dataset(ValidI,1:d-1);
    yvalid = dataset(ValidI,d);
    % Test samples
    xte = dataset(TestI,1:d-1);
    yte = dataset(TestI,d);
    
    max_s = min(ceil((m0/log(m0))^(1/(d-1))),10); % the maximal bound for s in theory, max_s = 2
    
    %% training the best parameters via validation    
    s = (1:max_s)'; % the candidates of s
    ValidErr = zeros(max_s,1); % validation error for each s
    temp_TrainTime = zeros(max_s,1); % training time for each s
    for j=1:max_s
        nc = min(nchoosek(s(j)+d-1,d-1),m0); % number of centers
        cx = xtr(1:nc,:); % strategy 1
        
        [~,~,ValidErr(j),temp_TrainTime(j),~] = FPC(xtr,ytr,xvalid,yvalid,s(j),cx);
        fprintf('trail:%d, s:%d,validerr:%f\n',i, s(j), ValidErr(j));
    end
    traintime(i) = sum(temp_TrainTime); % total training time
    
    bestj = find(ValidErr==min(ValidErr));
    best_s(i) = s(bestj(1)); % the best degree of polynomial parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% training and testing using the best parameter obtained via validation
    best_nc(i) = min(nchoosek(best_s(i)+d-1,d-1),m0); % the best number of centers
    best_cx = xtr(1:best_nc(i),:); %the best centers
    [beta,trainerr(i),testerr(i),~,testtime(i)]=FPC(xtr,ytr,xte,yte,best_s(i),best_cx);    
    
    fprintf('trail:%d, trainerr:%f, testerr:%f, traintime:%f, testtime:%f\n', ...
        i, trainerr(i), testerr(i), traintime(i), testtime(i));
end
fprintf('testerr:%f,trainerr:%f\n',(1-mean(testerr)),(1-mean(trainerr)));

fprintf('stdoftesterr:%f\n',std(testerr));

fprintf('traintime:%f, testtime:%f\n',mean(traintime),mean(testtime));

fprintf('best_s:%f, best_nc:%f\n',mean(best_s), mean(best_nc));