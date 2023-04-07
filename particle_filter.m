clc
clear
close all 

%% 1. Create CUDAKernel object.
k_particle_filter = parallel.gpu.CUDAKernel('./particle_filter.ptx', './particle_filter.cu')


%% 2. Set object properties.
threadsize = 32;
gridsize = 32; %256
k_particle_filter.GridSize = [gridsize 1 1];
k_particle_filter.ThreadBlockSize = [threadsize 1 1];


%% 3. Generate random array
d = gpuDevice; % get the Device
gpurng(0, 'Threefry'); % set the rng seed and type


data = load("pendulum.mat");
y = data.y;
x = data.x;

dt = 0.01;
q_c = 0.15;
g = 9.81;

rSigma = 0.05;
rSigma2 = rSigma * rSigma;

% Eq.(6) lecture 6
Q = [ q_c * dt^3 / 3, q_c * dt^2 / 2;...
      q_c * dt^2 / 2, q_c * dt];

% Cholesky decomposition, needed for drawing samples from N( 0, Q)
% You can also compute this analytically if it is difficult to input to the
% GPU devices
L_device = gpuArray(single(chol( Q )));

N = length( y );
J = threadsize*gridsize;

x1_mu = 0.5;
x1_sigma = 1;

x2_mu = 0;
x2_sigma = 1;

X1 = zeros( J, N,"single");
X2 = zeros( J, N,"single");


% Step 1 of the bootstrap filter
x1 = single(x2_mu + x1_sigma * randn( J, 1));
x2 = single(x1_mu + x2_sigma * randn( J, 1));

X1(:,1) = x1;
X2(:,1) = x2;

x1_prev = x1;
x2_prev = x2;

X_device = zeros(J,N,"single");
W_device = zeros(J,N,"single"); 
y_device = gpuArray(single(y));


[X1,X2]=feval(k_particle_filter,X_device,X_device,dt,N,J,L_device,x1,x2,W_device,y_device,X_device,gpuArray(single(rSigma)))

%
% After this point we are just plotting the results

x1_groundTruth = x( 1, :);
x2_groundTruth = x( 2, :);

p = [ 0.025, 0.50, 0.975];

tArea = 1:N;
tArea = [ tArea, fliplr( tArea )];

% confidence intervals
x1_CI = quantile( X1, p);
x2_CI = quantile( X2, p);

x1_median = x1_CI( 2, :);
x1_lowerBound = x1_CI( 1, :);
x1_upperBound = x1_CI( 3, :);

x1Area = [ x1_lowerBound, fliplr( x1_upperBound )];

x2_median = x2_CI( 2, :);
x2_lowerBound = x2_CI( 1, :);
x2_upperBound = x2_CI( 3, :);

x2Area = [ x2_lowerBound, fliplr( x2_upperBound )];

figure();
tiledlayout("flow");

nexttile();
hold on

h = plot( x1_groundTruth );
h.LineWidth = 2;

h = plot( x1_median );
h.LineWidth = 2;

hFill = fill( tArea, x1Area, h.Color);
hFill.FaceAlpha = 0.25;
hFill.LineStyle = "none";


nexttile();
hold on

h = plot( x2_groundTruth );
h.LineWidth = 2;

h = plot( x2_median );
h.LineWidth = 2;

hFill = fill( tArea, x2Area, h.Color);
hFill.FaceAlpha = 0.25;
hFill.LineStyle = "none";

