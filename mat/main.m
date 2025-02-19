%% Measure time and clear workspace variables 
tic;
clearvars;
clc;

%% Data required for Optimization Problem
Tsamp = 1/1.4; % sampling time of one minute
% Load real measurement
% netLoadStruct = load('data/flymat.mat').flyDat(1, 1:100) .* 50;

numberOfClasses = 4;
numberOfTimeSamples = 60;

kh = 0.5;
ki = 0.3;
kp = 0.1;

A = [-kh,   0,   0, 0;
      kh, -ki,   0, 0;
       0,  ki, -kp, 0;
       0,   0,  kp, 0];
B = [0, 0, 0, 0; 
     0, 1, 0, 0;
     0, 0, 0, 0;
     0, 0, 0, 0;];
Q = [1, 0, 0, 0; 
     0, 1, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, 0;];

S = [1, 0, 0, 0; 
     0, 1, 0, 0;
     0, 0, 1, 0;
     0, 0, 0, 1;];

%% Create Optimization Problem
prob = optimproblem;

%% Optimization Variables
x = optimvar('x',numberOfClasses,numberOfTimeSamples);
% Optimial Dosage
d = optimvar('d',numberOfClasses,numberOfTimeSamples); 
a = optimvar('a',numberOfClasses,numberOfTimeSamples); 
%% Constraints

InitCons = x(:, 1) == [1, 0, 0, 0]';
prob.Constraints.InitCons = InitCons;

% SoC Dynamics 
NextStateConsX = optimconstr(numberOfClasses, numberOfTimeSamples-1);

stateCost = optimexpr(1, numberOfTimeSamples);

% this for loop constrains next state dynamics and gen ramping
% it also computes stateCost at each time instance
for k = 1:numberOfTimeSamples-1
    NextStateConsX(:,k) = x(:, k+1) == x(:, k) + A * x(:,k) * Tsamp + B * d(:, k);
    stateCost(k) = x(:, k)' * Q * x(:, k) + (a(:, k))' * S * x(:, k + 1);
end

% State cost at final time 
% Q may change for the final time
stateCost(numberOfTimeSamples) = x(:,numberOfTimeSamples)' * Q * x(:,numberOfTimeSamples);
% assign NextStateCons array
prob.Constraints.NextStateConsX = NextStateConsX; 


%% Objective
% matrix operations are unfortunately restricted on optimization variables

% define cost and objective
cost = sum(stateCost);
prob.Objective = cost;

%% Solve the optimization problem
% optimalDD: structure with optimal trajectories
% minCost: minimum value of cost functional (scalar)
% exitflag and output contain solver information
% lambda: structure containing lagrange multipliers for constraints

% sol0.x = zeros(numberOfClasses,numberOfTimeSamples);
% sol0.d = zeros(numberOfClasses,numberOfTimeSamples);
% sol0.a = zeros(1,numberOfTimeSamples);
% [optimalDD, minCost, exitflag, output, lambda] = solve(prob, sol0);

[optimalDD, minCost, exitflag, output, lambda] = solve(prob);



