"""
#################### Three Axle Symbolic Derivation ####################

    Author: 
        Tahn Thawainin, AU GAVLAB
        pzt0029@auburn.edu

    Description: 
        Symbolic derivation of the 3 axle tractor-trailer bicycle model.

########################################################################
"""
#%%
import numpy as np
from sympy import *
from algebra_with_sympy import *
init_printing()

# ----- Declare Symbols ----- #
# vehicle parameters
a, b1, b2, c, m1, J1, C1, C2, C3 = symbols('a b1 b2 c m1 J1 C1 C2 C3')

# time varying terms
t = symbols('t')
y = Function('y')(t)
psi = Function('psi')(t)
vy = Function('vy')(t)
vx = Function('vx')(t)
delta = Function('delta')(t)

psi_dot = diff(psi,t)
psi_ddot = diff(psi,t,t)
y_dot = diff(y,t)
y_ddot = diff(y,t,t)
vy_dot = diff(vy,t)

# dot derivative latex expressions (for equation visualization)
dpsi = Function('\\dot{\\psi}')(t)
ddpsi = Function('\\ddot{\\psi}')(t)
dvy = Function('\\dot{vy}')(t)

# ----- Tire Slip and Force Expressions ----- #
# slip expressions (SAA)
alpha1 = delta - ((vy + a*psi_dot) / vx)
alpha2 = -((vy - b1*psi_dot) / vx)
alpha3 = -((vy - b2*psi_dot) / vx)

F1 = C1*alpha1
F2 = C2*alpha2
F3 = C3*alpha3

# ----- Generalized Forces ----- #
Q_y = F1*cos(delta) + F2 + F3
Q_psi = a*F1*cos(delta) - b1*F2 - b2*F3

# ----- Lagrangian Derivation ----- #
T = (1/2)*m1*(vx**2 + y_dot**2) + (1/2)*J1*psi_dot**2
V = 0
L = T - V

# first EOM
d1_y = diff(L,y_dot)
d2_y = diff(L,y)
EOM1 = Eqn(diff(d1_y,t) - d2_y, Q_y)
EOM1 = EOM1.subs(y_ddot,vy_dot + vx*psi_dot)
EOM1 = expand(EOM1)
EOM1 = EOM1.subs(vy_dot,dvy)
EOM1 = EOM1.subs(psi_dot,dpsi)
EOM1 = collect(EOM1,[vy,dpsi])

# second EOM
d1_psi = diff(L,psi_dot)
d2_psi = diff(L,psi)
EOM2 = Eqn(diff(d1_psi,t) - d2_psi, Q_psi)
EOM2 = expand(EOM2)
EOM2 = EOM2.subs(psi_ddot,ddpsi)
EOM2 = EOM2.subs(psi_dot,dpsi)
EOM2 = collect(EOM2, [vy,dpsi])

