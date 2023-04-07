
clear
close all

dt = 0.01;
q_c = 0.1;
g = 9.81;

rSigma = 0.05;
Q = [ q_c * dt^3 / 3, q_c * dt^2 / 2;...
      q_c * dt^2 / 2, q_c * dt];
  
L = chol( Q );

N = 300;

x = zeros( 2, N);
x( 1, 1) = 0;
x( 2, 1) = 1;

for n = 2:N
    
    % Generate the next states with Eq.(5)
    xPrev = x( :, n - 1);
    
    x1_prev = xPrev(1);
    x2_prev = xPrev(2);
    
    w_n = randn( 2, 1);
    q_n = L * w_n;
    
    x1_next = x1_prev + x2_prev * dt + q_n(1);
    x2_next = x2_prev - g * sin( x1_prev ) * dt + q_n(2);
    
    xNext = [ x1_next; x2_next];
    x( :, n) = xNext;
end

x1 = x( 1, :);
x2 = x( 2, :);

y = sin( x1 ) + rSigma * randn( 1, N);

figure();
tiledlayout("flow");

nexttile();
h = plot( y );
h.LineWidth = 2;

h = title("$ y $");
h.Interpreter = "latex";

ax = gca();
ax.FontSize = 25;

nexttile();
h = plot( x1 );
h.LineWidth = 2;

h = title("$ x_1 $");
h.Interpreter = "latex";

ax = gca();
ax.FontSize = 25;

nexttile();
h = plot( x2 );
h.LineWidth = 2;

h = title("$ x_2 $");
h.Interpreter = "latex";

ax = gca();
ax.FontSize = 25;

save("pendulum.mat", "y", "x");
















