% Symbolic solutions of the Static Output Feeedback Equations for the
% Linearized system

% Author: Murad Abu-Khalaf, MIT CSAIL.


syms p11 p12 p13 p21 p22 p23 p31 p32 p33
P = [p11 p12 p13;
     p21 p22 p23; 
     p31 p32 p33];

p21=p12;
p31=p13;
p32=p23;

P = subs(P);

syms alpha1 m1 alpha2 m2 c
assume(c,'real')

A = [-alpha1/m1 0 0;1 0 -1;0 0 -alpha2/m2];
B = [0; 0; 1/m2];
C = [0 c 0];

syms G1 G2 G3 
G = [G1 G2 G3];

V = eye(3) - C'*inv(C*C')*C;

Ric = transpose(A)*P + P*A - P*B*transpose(B)*P + transpose(C)*C + transpose(G)*G
Proj = V*(P*A+transpose(A)*P)*V

% Conditions that guarantee that V*(P*A+transpose(A)*P)*V = 0
p12=alpha1/m1*p11;
p13=-(alpha1/m1*p11+alpha2/m2*p33)/(alpha1/m1+alpha2/m2); 
p23 =-alpha2/m2*p33;

% Verify that this indeeds solve the kernel equation
simplify(subs(Proj))

% get all 6 quadratic equations
simplify(subs(Ric))

% Solve symbolically the 6 quadratic equations by first setting G2 = 0
% (guess using numerical solutions), then solve for p33, then G3, then p22,
% then p11, then G1.

p11 = (abs(c)*alpha2+c^2*(m2/alpha2)-c^2*(m1/alpha1)*(m2/alpha2)/((m2/alpha2)+(m1/alpha1))) / (abs(c)*alpha1/(alpha1*m2+alpha2*m1)+alpha1^2/m1^2 )
p22 = abs(c)*alpha2 + c^2*m2/alpha2
p33 = abs(c)*m2^2/alpha2

p12 = alpha1/m1*p11
p13 = -((alpha1/m1*p11) + (alpha2/m2*p33)) / (alpha1/m1 + alpha2/m2)
p23 = -alpha2/m2*p33

G1 = -(alpha1*m2*p11+alpha2*m1*p33)/(m2*(alpha1*m2+alpha2*m1))
G2 = 0
G3 = abs(c)*m2/alpha2

% Verify that the the equations are solved for this symbolic answer.
simplify(subs(Ric))
simplify(subs(Proj))