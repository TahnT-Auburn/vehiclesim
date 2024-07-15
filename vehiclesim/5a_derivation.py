"""
#################### Three Axle Symbolic Derivation ####################

    Author: 
        Tahn Thawainin, AU GAVLAB
        pzt0029@auburn.edu

    Description: 
        Symbolic derivation of the 5 axle tractor-trailer bicycle model.

########################################################################
"""
#%%
import numpy as np
from sympy import *
from algebra_with_sympy import *
init_printing()

print(f"Symbolic 5 Axle Bicycle Model Derivation")

# ----- Declare Revision -----#
revision = 4
print(f"Revision: {revision}\n")

# ----- Declare Symbols ----- #
# vehicle parmaters
a, b1, b2, c, d, f1, f2, m1, m2, J1, J2, C1, C2, C3, C4, C5 = symbols('a b1 b2 c d f1 f2 m1 m2 J1 J2 C1 C2 C3 C4 C5')

# time varying terms
t = symbols('t')
y = Function('y')(t)
psi = Function('psi')(t)
gamma_ = Function('gamma_')(t)
vy = Function('vy')(t)
vx = Function('vx')(t)
delta = Function('delta')(t)

psi_dot = diff(psi,t)
psi_ddot = diff(psi,t,t)
gamma_dot = diff(gamma_,t)
gamma_ddot = diff(gamma_,t,t)
y_dot = diff(y,t)
y_ddot = diff(y,t,t)
vy_dot = diff(vy,t)

# dot derivative latex expressions (for equation visualization)
dpsi = Function('\\dot{\\psi}')(t)
ddpsi = Function('\\ddot{\\psi}')(t)
dgamma = Function('\\dot{\\gamma}')(t)
ddgamma = Function('\\ddot{\\gamma}')(t)
dvy = Function('\\dot{vy}')(t)

# ----- Trailer Velocity Expressions ------ #
vx2 = vx
vy2 = y_dot - (c + d*cos(gamma_))*psi_dot - d*cos(gamma_)*gamma_dot

# ----- Tire Slip and Force Expressions ----- #
# slip expressions (SAA)
alpha1 = delta - ((vy + a*psi_dot) / vx)
alpha2 = -((vy - b1*psi_dot) / vx)
alpha3 = -((vy - b2*psi_dot) / vx)
alpha4 = gamma_ - ((vy - (c + f1*cos(gamma_))*psi_dot - f1*gamma_dot*cos(gamma_)) / vx2)
alpha5 = gamma_ - ((vy - (c + f2*cos(gamma_))*psi_dot - f2*gamma_dot*cos(gamma_)) / vx2)

F1 = C1*alpha1
F2 = C2*alpha2
F3 = C3*alpha3
F4 = C4*alpha4
F5 = C5*alpha5

# ----- Generalized Forces ----- #
Q_y = F1*cos(delta) + F2 + F3 + F4*cos(gamma_) + F5*cos(gamma_)
Q_psi = a*F1*cos(delta) - b1*F2 - b2*F3 - (c*cos(gamma_) + f1)*F4 - (c*cos(gamma_) + f2)*F5
Q_gamma = -f1*F4 - f2*F5

# ----- Lagrangian Derivation ----- #
T = (1/2)*m1*(vx**2 + y_dot**2) + (1/2)*J1*psi_dot**2 + (1/2)*m2*(vx2**2 + vy2**2) + (1/2)*J2*(psi_dot + gamma_dot)**2
V = 0
L = T - V

# first EOM
d1_y = diff(L,y_dot)
d2_y = diff(L,y)
EOM1 = Eqn(diff(d1_y,t) - d2_y, Q_y)
EOM1 = EOM1.subs(y_ddot,vy_dot + vx*psi_dot)
EOM1 = expand(EOM1)
EOM1 = EOM1.subs(y_dot,vy)
EOM1 = EOM1.subs(vy_dot,dvy)
EOM1 = EOM1.subs(psi_ddot,ddpsi)
EOM1 = EOM1.subs(psi_dot,dpsi)
EOM1 = EOM1.subs(gamma_ddot,ddgamma)
EOM1 = EOM1.subs(gamma_dot,dgamma)
EOM1 = collect(EOM1,[dvy, dpsi, ddpsi, dgamma, ddgamma, vy, psi, gamma_])

# second EOM
d1_psi = diff(L,psi_dot)
d2_psi = diff(L,psi)
EOM2 = Eqn(diff(d1_psi,t) - d2_psi, Q_psi)
EOM2 = EOM2.subs(y_ddot,vy_dot + vx*psi_dot)
EOM2 = expand(EOM2)
EOM2 = EOM2.subs(y_dot,vy)
EOM2 = EOM2.subs(vy_dot,dvy)
EOM2 = EOM2.subs(psi_ddot,ddpsi)
EOM2 = EOM2.subs(psi_dot,dpsi)
EOM2 = EOM2.subs(gamma_ddot,ddgamma)
EOM2 = EOM2.subs(gamma_dot,dgamma)
EOM2 = collect(EOM2,[dvy, dpsi, ddpsi, dgamma, ddgamma, vy, psi, gamma_])

# third EOM
d1_gamma = diff(L,gamma_dot)
d2_gamma = diff(L,gamma_)
EOM3 = Eqn(diff(d1_gamma,t) - d2_gamma, Q_gamma)
EOM3 = EOM3.subs(y_ddot,vy_dot + vx*psi_dot)
EOM3 = expand(EOM3)
EOM3 = EOM3.subs(y_dot,vy)
EOM3 = EOM3.subs(vy_dot,dvy)
EOM3 = EOM3.subs(psi_ddot,ddpsi)
EOM3 = EOM3.subs(psi_dot,dpsi)
EOM3 = EOM3.subs(gamma_ddot,ddgamma)
EOM3 = EOM3.subs(gamma_dot,dgamma)
EOM3 = collect(EOM3,[dvy, dpsi, ddpsi, dgamma, ddgamma, vy, psi, gamma_])

# ----- Revision 4 ----- #
# cos(gamma) = 1, sin(gamma) = 0
if (revision == 4):
    EOM1 = EOM1.subs(cos(gamma_),1)
    EOM1 = EOM1.subs(cos(gamma_)**2,1)
    EOM1 = EOM1.subs(sin(gamma_),0)
    EOM1 = EOM1.subs(sin(gamma_)**2,0)

    EOM2 = EOM2.subs(cos(gamma_),1)
    EOM2 = EOM2.subs(cos(gamma_)**2,1)
    EOM2 = EOM2.subs(sin(gamma_),0)
    EOM2 = EOM2.subs(sin(gamma_)**2,0)
    
    EOM3 = EOM3.subs(cos(gamma_),1)
    EOM3 = EOM3.subs(cos(gamma_)**2,1)
    EOM3 = EOM3.subs(sin(gamma_),0)
    EOM3 = EOM3.subs(sin(gamma_)**2,0)

# ----- Revision 6 ----- #
# sin(gamma) = 0
if (revision == 6):
    EOM1 = EOM1.subs(sin(gamma_),0)
    EOM1 = EOM1.subs(sin(gamma_)**2,0)

    EOM2 = EOM2.subs(sin(gamma_),0)
    EOM2 = EOM2.subs(sin(gamma_)**2,0)
    
    EOM3 = EOM3.subs(sin(gamma_),0)
    EOM3 = EOM3.subs(sin(gamma_)**2,0)

# ----- Linear State Space Representation ----- #
if (revision != 0):
    M = Matrix([[EOM1.lhs.coeff(dvy), EOM1.lhs.coeff(ddpsi), EOM1.lhs.coeff(dpsi), EOM1.lhs.coeff(ddgamma), EOM1.lhs.coeff(dgamma)], \
                [EOM2.lhs.coeff(dvy), EOM2.lhs.coeff(ddpsi), EOM2.lhs.coeff(dpsi), EOM2.lhs.coeff(ddgamma), EOM2.lhs.coeff(dgamma)], \
                [0, 0, 1, 0, 0], \
                [EOM3.lhs.coeff(dvy), EOM3.lhs.coeff(ddpsi), EOM3.lhs.coeff(dpsi), EOM3.lhs.coeff(ddgamma), EOM3.lhs.coeff(dgamma)], \
                [0, 0, 0, 0, 1]])
    
    K = Matrix([[EOM1.rhs.coeff(vy), EOM1.rhs.coeff(dpsi), EOM1.rhs.coeff(psi), EOM1.rhs.coeff(dgamma), EOM1.rhs.coeff(gamma_)], \
                [EOM2.rhs.coeff(vy), EOM2.rhs.coeff(dpsi), EOM2.rhs.coeff(psi), EOM2.rhs.coeff(dgamma), EOM2.rhs.coeff(gamma_)], \
                [0, 1, 0, 0, 0], \
                [EOM3.rhs.coeff(vy), EOM3.rhs.coeff(dpsi), EOM3.rhs.coeff(psi), EOM3.rhs.coeff(dgamma), EOM3.rhs.coeff(gamma_)], \
                [0, 0, 0, 1, 0]])
    
    F = Matrix([[EOM1.rhs.coeff(delta)], \
               [EOM2.rhs.coeff(delta)], \
                [0],
                [EOM3.rhs.coeff(delta)], \
                [0]])
    
    M = simplify(nsimplify(M))
    K = simplify(nsimplify(K))
    F = simplify(nsimplify(F))

    A = M.inv()*K
    B = M.inv()*F

    # ----- Transfer Functions ----- #
    # laplace operator
    s = symbols('s') 

    # # lateral velocity TF
    # H_vy = Matrix([1,0,0,0,0])*(s*eye(5) - A).inv()*B
    # # yaw rate TF
    # H_yr = Matrix([0,1,0,0,0])*(s*eye(5) - A).inv()*B
    # # hitch rate TF
    # H_hr = Matrix([0,0,0,1,0])*(s*eye(5) - A).inv()*B

    # ----- Evaluate ----- #
    # vehParams = {vx:5,\
    #              delta:0,\
    #              gamma_:0,\
    #              a:1.384,\
    #              b1:3.616,\
    #              b2:4.886,\
    #              c:4.251,\
    #              d:7.303,\
    #              f1:13.124,\
    #              f2:14.412,\
    #              m1:8633,\
    #              m2:4526,\
    #              J1:1.9658e+04,\
    #              J2:1.8001e+05,\
    #              C1:3.922730292168247e+05,\
    #              C2:1.817771488938941e+05,\
    #              C3:1.817771488938941e+05,\
    #              C4:1.206555470490843e+05,\
    #              C5:1.206555470490843e+05}
    
    # A_numeric = A.evalf(subs=vehParams)
    # B_numeric = B.evalf(subs=vehParams)
