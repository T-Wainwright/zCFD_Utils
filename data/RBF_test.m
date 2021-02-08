clear all
close all

A_as = readmatrix('A_as.csv');
M_ss = readmatrix('M_ss.csv');
P_s = readmatrix('P_s.csv');

M_inv = pinv(M_ss);

M_p = pinv(P_s*M_inv*P_s');

TOP = M_p*P_s*M_inv;
BOTTOM = M_inv - M_inv*P_s'*M_p*P_s*M_inv;

Combined = [TOP;BOTTOM];

H = A_as*Combined;

Css = [zeros(4,4),P_s;P_s',M_ss];

H2 = A_as * pinv(Css);