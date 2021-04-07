% Numerical solutions of the Static Output Feeedback Equations for the
% Linearized system

% Author: Murad Abu-Khalaf, MIT CSAIL.

m1=2; m2=7; alpha1=3; alpha2=5; c=-1.2;
A = [-alpha1/m1 0 0;1 0 -1;0 0 -alpha2/m2];
B = [0; 0; 1/m2];
C = [0 c 0];

V = eye(3) - C'*inv(C*C')*C;

G = [0;0;0];
Q = C'*C+G'*G;
P = icare(A,B,Q);

for i=1:20
    G = -B'*P*C'*inv(C*C')*C + B'*P;
    Q = C'*C+G'*G;
    P = icare(A,B,Q);
end


V*(P*A+transpose(A)*P)*V
transpose(A)*P + P*A - P*B*transpose(B)*P + transpose(C)*C + transpose(G)*G

P
G

% Compare numerical soution with symbolic one

% [                                       G1^2 - (alpha1*m2*p11 + alpha2*m1*p33)^2/(m2^2*(alpha1*m2 + alpha2*m1)^2), p22 + G1*G2 - (alpha1^2*p11)/m1^2 - (alpha2*p33*(alpha1*m2*p11 + alpha2*m1*p33))/(m2^3*(alpha1*m2 + alpha2*m1)), (G1*G3*alpha1*m2^3 + G1*G3*alpha2*m1*m2^2 + alpha1*p11*m2*p33 + alpha2*m1*p33^2)/(m2^2*(alpha1*m2 + alpha2*m1))]
% [ p22 + G1*G2 - (alpha1^2*p11)/m1^2 - (alpha2*p33*(alpha1*m2*p11 + alpha2*m1*p33))/(m2^3*(alpha1*m2 + alpha2*m1)),                                                                              G2^2 + c^2 - (alpha2^2*p33^2)/m2^4,                                                         G2*G3 - p22 + (alpha2^2*p33)/m2^2 + (alpha2*p33^2)/m2^3]
% [ (G1*G3*alpha1*m2^3 + G1*G3*alpha2*m1*m2^2 + alpha1*p11*m2*p33 + alpha2*m1*p33^2)/(m2^2*(alpha1*m2 + alpha2*m1)),                                                         G2*G3 - p22 + (alpha2^2*p33)/m2^2 + (alpha2*p33^2)/m2^3,                                                                                               G3^2 - p33^2/m2^2]
 
p11 = (abs(c)*alpha2+c^2*(m2/alpha2)-c^2*(m1/alpha1)*(m2/alpha2)/((m2/alpha2)+(m1/alpha1))) / (abs(c)*alpha1/(alpha1*m2+alpha2*m1)+alpha1^2/m1^2 )
p22 = abs(c)*alpha2 + c^2*m2/alpha2
p33 = abs(c)*m2^2/alpha2

p12 = alpha1/m1*p11
p13 = -((alpha1/m1*p11) + (alpha2/m2*p33)) / (alpha1/m1 + alpha2/m2)
p23 = -alpha2/m2*p33

G1 = -(alpha1*m2*p11+alpha2*m1*p33)/(m2*(alpha1*m2+alpha2*m1))
G2 = 0
G3 = abs(c)*m2/alpha2


% Note that multiple solutions are possible. Try these numerical solutions
% p11 = 1;
% p22 = 2;
% p33 = 1;
% m1=1; m2=1; alpha1=1; alpha2=1; c =1;
% G1=-1;G2=0;G3=1; 
% %G1=1;G2=0;G3=-1; % a second valid solution for G