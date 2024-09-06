"""
#################### Vehicle Roll Symbolic Derivation ####################

    Author: 
        Tahn Thawainin, AU GAVLAB
        pzt0029@auburn.edu

    Description: 
        Symbolic derivation of the vehicle roll model. Includes Nonlinear EOM
        and a state space representation for KF formulation

##########################################################################
"""

#%%
import numpy as np
from sympy import *
from algebra_with_sympy import *
init_printing()

print(f"Symbolic Vehicle Roll Model Derivation")

#----- Declare Symbols -----#

m_s, k, c, ts, Ixx, g = symbols('m_s k c ts Ixx g')

# time varying terms
t = symbols('t')
phi = Function('phi')(t)
hr = Function('hr')(t)
Ay = Function('Ay')(t)

phi_dot = diff(phi,t)
phi_ddot = diff(phi,t,t)
hr_dot = diff(hr,t)
hr_ddot = diff(hr,t,t)

# dot derivative latex expressions (for equation visulization)
dphi = Function('\\dot{\\phi}')(t)
ddphi = Function('\\ddot{\\phi}')(t)
dhr = Function('\\dot{hr}')(t)
ddhr = Function('\\ddot{hr}')(t)

#----- Nonlinear EOM -----#
NL_EOM  = Eqn((Ixx + m_s*hr**2)*phi_ddot, -(1/2)*c*ts**2*cos(phi)*phi_dot + (m_s*g*hr - (1/2)*k*ts**2)*sin(phi) + m_s*Ay*hr*cos(phi))
NL_EOM = expand(NL_EOM)
NL_EOM = NL_EOM.subs(phi_ddot,ddphi)
NL_EOM = NL_EOM.subs(phi_dot,dphi)
NL_EOM = collect(NL_EOM,[Ay,ddphi,dphi,phi])

#----- Jacobian KF Formulation -----#
dynamics = Eqn(phi_ddot, (1 / (Ixx + m_s*hr**2))*(-(1/2)*c*ts**2*cos(phi)*phi_dot + (m_s*g*hr - (1/2)*k*ts**2)*sin(phi) + m_s*Ay*hr*cos(phi)))

# state transition matrix
F = Matrix([[diff(dynamics.rhs,phi_dot), diff(dynamics.rhs,phi), diff(dynamics.rhs,hr_dot), diff(dynamics.rhs,hr)],
            [diff(phi_dot,phi_dot), diff(phi_dot,phi), diff(phi_dot,hr_dot), diff(phi_dot,hr)],
            [diff(hr_ddot,phi_dot), diff(hr_ddot,phi), diff(hr_ddot,hr_dot), diff(hr_ddot,hr)],
            [diff(hr_dot,phi_dot), diff(hr_dot,phi), diff(hr_dot,hr_dot), diff(hr_dot,hr)]])

F = F.subs(phi_dot,dphi)
