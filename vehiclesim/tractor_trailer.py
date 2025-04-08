"""
#################### Tractor Trailer Class ####################

    Author: 
        Tahn Thawainin, AU GAVLAB

    Description: 
        A class to handle tractor trailer simulations.
        Simulations include:
            * Lateral Vehicle Model
            * Lateral Tire Cornering Estimation
            * Lateral Vehicle State Estimation
            {add simulations}

    Dependencies:
        <numpy>
        <python_utilities>

################################################################
"""
import pandas as pd
import numpy as np
import math
from control.matlab import *
from python_utilities.parsers_class import Parsers

class TractorTrailer:

    def __init__(self, veh_config_file:str, config_type:str='5a', ts_data_file:str=None):
        """
        Class Description:
            A class to handle tractor trailer simulations.
        ------------
        Class Instance Input(s):
            * veh_config_file:  Vehicle yaml config file
                                type: <str>

            * config_type:  Vehicle configuration type (Options: '3a', '5a') [Default = '5a']
                            type: <str>

            * ts_data_file: TruckSim csv data file
                            type: <str>
        """

        #Input assertions
        assert type(veh_config_file) == str,\
            f"Input <ts_data> has invalid type. Expected <class 'str'> but recieved {type(veh_config_file)}"
        assert type(config_type) == str,\
            f"Input <ts_data> has invalid type. Expected <class 'str'> but recieved {type(config_type)}"
        assert type(ts_data_file) == str or ts_data_file == None,\
            f"Input <ts_data> has invalid type. Expected <class 'str'> but recieved {type(ts_data_file)}"
        
        #Class objects
        self.veh_config_file = veh_config_file
        self.config_type = config_type
        self.ts_data_file = ts_data_file
        
        #Generate vehicle configuration
        self.genVehParams()  

        #Generate TruckSim data
        if self.ts_data_file is not None:   
            self.genTsData()


    def genTsData(self, ts_data_file=None):
        """
        `Description`:
            Generates <self.ts_data> as a data frame from parsed TruckSim data
        ------------
        `Input(s)`:
            ts_data_file: str, optional.
                Directory to TruckSim data file.
        ------------
        `Output(s)`:
            None
        """

        if ts_data_file is None:
            ts_data_file = self.ts_data_file

        #Parse TruckSim csv data
        ts_data = Parsers().csvParser(ts_data_file)

        #Eliminate empty spaces
        ts_data.columns = pd.Index(pd.Series(ts_data.columns).apply(lambda x: x.strip())) #strip extra empty spaces

        #Generate self.ts_data
        self.ts_data = ts_data

        return ts_data
    
    def genVehParams(self):
        """
        `Description`:
            Generates <self.vp> as a dict/Box from parsed vehicle parameters
        ------------
        `Input(s)`:
            None
        ------------
        `Outputs(s)`:
            None
        """
        
        #Parse yaml
        vp = Parsers().yamlParser(self.veh_config_file, box=True)
        
        #Generate additional parameters
        if (self.config_type == '3a'):

            #Tractor mass
            vp.m_t1 = vp.m_s1 + vp.m_us_A1 + vp.m_us_A2 + vp.m_us_A3

            #Total mass
            vp.m_veh = vp.m_t1
            
            #Tractor inertias
            vp.j_xx1 = vp.m_s1*vp.Rx1**2    # roll inertia
            vp.j_yy1 = vp.m_s1*vp.Ry1**2    # pitch inertia
            vp.j_zz1 = vp.m_s1*vp.Rz1**2    # yaw inertia

        if (self.config_type == '5a'):
            
            #Tractor mass
            vp.m_t1 = vp.m_s1 + vp.m_us_A1 + vp.m_us_A2 + vp.m_us_A3
           
            #Tractor inertias
            vp.j_xx1 = vp.m_s1*vp.Rx1**2    # roll inertia
            vp.j_yy1 = vp.m_s1*vp.Ry1**2    # pitch inertia
            vp.j_zz1 = vp.m_s1*vp.Rz1**2    # yaw inertia
            # vp.j_zz1 = 19665
            #Trailer mass
            vp.m_t2 = vp.m_s2 + vp.m_us_A4 + vp.m_us_A5
            
            #Trailer inertias
            vp.j_xx2 = vp.m_s2*vp.Rx2**2    # roll inertia
            vp.j_yy2 = vp.m_s2*vp.Ry2**2    # pitch inertia
            vp.j_zz2 = vp.m_s2*vp.Rz2**2    # yaw inertia
            # vp.j_zz2 = 179992
            #Total mass
            vp.m_veh = vp.m_t1 + vp.m_t2

        #Generate self.vp
        self.vp = vp


    def latModel(self, steer_ang, Vx, dt):
        """
        Lateral Bicycle Model Simulation. 
        (Source: S.M. Wolfe, "Heavy Truck Modeling and Estimation for Vehicle-to-Vehicle Collision Avoidance Systems")
        
        Parameters:
            steer_ang:  steering angle [rad]
            Vx: longitudinal velocity [m/s]
            dt: sampling rate (for discretization)
        Returns:
            sysc:   continuous time state space model
            sysd:   discrete time state space model
        
        """
        # vehicle parameters
        vp = self.vp
        # cornering stiffness coefficients
        C1 = vp.cs[0]
        C2 = vp.cs[1]
        C3 = vp.cs[2]
        C4 = vp.cs[3]
        C5 = vp.cs[4]
        
        # inertial matrix
        M = np.array([
            [vp.m_t1 + vp.m_t2, -vp.m_t2*(vp.c + vp.d), Vx*(vp.m_t1 + vp.m_t2), -vp.m_t2*vp.d, 0],

            [-vp.m_t2*(vp.c + vp.d), vp.j_zz1 + vp.j_zz2 + vp.m_t2*(vp.c + vp.d)**2, \
            -vp.m_t2*Vx*(vp.c + vp.d), vp.j_zz2 + vp.m_t2*vp.d**2 + vp.m_t2*vp.c*vp.d, 0],

            [0, 0, 1, 0, 0],

            [-vp.m_t2*vp.d, vp.j_zz2 + vp.m_t2*vp.d**2 + vp.m_t2*vp.c*vp.d, -vp.m_t2*Vx*vp.d, \
            vp.j_zz2 + vp.m_t2*vp.d**2, 0],

            [0, 0, 0, 0, 1]
            ])
        
        # stiffness matrix
        k11 = (1/Vx)*(-C1*np.cos(steer_ang) - C2 - C3 - C4 - C5)
    
        k12 = (1/Vx)*(-C1*vp.a*np.cos(steer_ang) + C2*vp.b1 + C3*vp.b2 + C4*(vp.c + vp.f1) \
            + C5*(vp.c + vp.f2))
        
        k14 = (1/Vx)*(vp.f1*C4 + vp.f2*C5)
        
        k15 = C4 + C5
        
        k21 = (1/Vx)*(-C1*vp.a*np.cos(steer_ang) + C2*vp.b1 + C3*vp.b2 + C4*(vp.f1 + vp.c) \
            + C5*(vp.f2 + vp.c))
        
        k22 = (1/Vx)*(-C1*vp.a**2*np.cos(steer_ang) - C2*vp.b1**2 - C3*vp.b2**2 \
            - (vp.f1 + vp.c)*C4*(vp.c + vp.f1) \
            - (vp.f2 + vp.c)*C5*(vp.c + vp.f2))
        
        k24 = (1/Vx)*(-(vp.f1 + vp.c)*C4*vp.f1 \
            - (vp.f2 + vp.c)*C5*vp.f2)
        
        k25 = -(vp.f1 + vp.c)*C4 - (vp.f2 + vp.c)*C5
        
        k41 = (1/Vx)*(vp.f1*C4 + vp.f2*C5)
        
        k42 = (1/Vx)*(-vp.f1*C4*(vp.c + vp.f1) - vp.f2*C5*(vp.c + vp.f2))
        
        k44 = (1/Vx)*(-vp.f1**2*C4 - vp.f2**2*C5)
        
        k45 = -vp.f1*C4 - vp.f2*C5

        K = np.array([
            [k11, k12, 0, k14, k15],
            [k21, k22, 0, k24, k25],
            [0, 1, 0, 0, 0],
            [k41, k42, 0, k44, k45],
            [0, 0, 0, 1, 0]])

        # forcing matrix
        F = np.array([
            [np.cos(steer_ang)*C1],
            [vp.a*np.cos(steer_ang)*C1],
            [0],
            [0],
            [0]
        ])

        # continuous state space model
        Ac = np.linalg.inv(M) @ K   #State transition matrix
        Bc = np.linalg.inv(M) @ F   #Input matrix
        Cc = np.identity(5)         #Observation matrix
        Dc = 0                      #Measurement input matrix
        sysc = ss(Ac,Bc,Cc,Dc)
        
        # discrete state space model
        sysd = c2d(sysc, dt, 'zoh')

        return sysc, sysd
    
    def rollModel(self, unit, phi, phid, Ay, hr,):
        '''
        Tractor and trailer dynamic roll models.

        Parameters:
            unit: str, 'Tractor'/1 or 'Trailer'/2.
                Unit of interest.
            phi: float,
                Roll angle in rads.
            phid: float,
                Roll rate in rads.
            Ay: float,
                Lateral acceleration in m/s.
            hr: float,
                Instanteanous roll height in m.
        
        Returns:
            phidd: float,
                Roll acceleration from dynamic model.
        '''
        # vehicle parameters
        vp = self.vp
        if unit == 'Tractor' or unit == 1:
            j_xx = vp.j_xx1
            m_s = vp.m_s1
            ks = vp.ks1*3
            c = vp.c1*3
        elif unit == 'Trailer' or unit ==2:
            j_xx = vp.j_xx2
            m_s = vp.m_s2
            ks = vp.ks2*4
            c = vp.c2*4
        g = 9.81

        # simulate nonlinear EOM
        phidd = (1 / (j_xx + m_s*hr**2))*(m_s*Ay*hr*np.cos(phi) + m_s*g*hr*np.sin(phi) \
                  - (1/2)*ks*vp.ts**2*np.sin(phi) - (1/2)*c*vp.ts**2*np.cos(phi)*phid)
        
        return phidd
    
    def linRollModel(self, unit, hr, dt):
        # vehicle parameters
        vp = self.vp
        if unit == 'Tractor' or unit == 1:
            j_xx = vp.j_xx1
            m_s = vp.m_s1
            ks = vp.ks1*3
            c = vp.c1*3
        elif unit == 'Trailer' or unit ==2:
            j_xx = vp.j_xx2
            m_s = vp.m_s2
            ks = vp.ks2*4
            c = vp.c2*4
        g = 9.81

        # generate linearized system matrices
        Ac = np.array([[(-0.5*c*vp.ts**2) / (j_xx + m_s*hr**2), (m_s*g*hr - 0.5*ks*vp.ts**2) / (j_xx + m_s*hr**2)],
                       [1, 0]])
        Bc = np.array([[(m_s*hr) / (j_xx + m_s*hr**2)],
                       [0]])
        Cc = np.identity(2)
        Dc = 0

        sysc = ss(Ac,Bc,Cc,Dc)
        
        # discrete state space model
        sysd = c2d(sysc, dt, 'zoh')

        return sysc, sysd