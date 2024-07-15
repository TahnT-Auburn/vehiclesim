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

    def __init__(self, veh_config_file:str, config_type:str='5a', ts_data_file:str=''):
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
        assert type(ts_data_file) == str,\
            f"Input <ts_data> has invalid type. Expected <class 'str'> but recieved {type(ts_data_file)}"
        
        #Class objects
        self.veh_config_file = veh_config_file
        self.config_type = config_type
        self.ts_data_file = ts_data_file
        
        #Generate vehicle configuration
        self.genVehParams()  

        #Generate TruckSim data
        if (self.ts_data_file != ''):   
            self.genTsData()


    def genTsData(self):
        """
        `Description`:
            Generates <self.ts_data> as a data frame from parsed TruckSim data
        ------------
        `Input(s)`:
            None
        ------------
        `Output(s)`:
            None
        """

        #Parse TruckSim csv data
        ts_data = Parsers().csvParser(self.ts_data_file)

        #Eliminate empty spaces
        ts_data.columns = pd.Index(pd.Series(ts_data.columns).apply(lambda x: x.strip())) #strip extra empty spaces

        #Generate self.ts_data
        self.ts_data = ts_data


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

            #Trailer mass
            vp.m_t2 = vp.m_s2 + vp.m_us_A4 + vp.m_us_A5
            
            #Trailer inertias
            vp.j_xx2 = vp.m_s2*vp.Rx2**2    # roll inertia
            vp.j_yy2 = vp.m_s2*vp.Ry2**2    # pitch inertia
            vp.j_zz2 = vp.m_s2*vp.Rz2**2    # yaw inertia

            #Total mass
            vp.m_veh = vp.m_t1 + vp.m_t2

        #Generate self.vp
        self.vp = vp


    def latModel(self, steer_ang, Vx):
        """
        `Description`:
            Lateral Bicycle Model Simulation. 
            (Source: S.M. Wolfe, "Heavy Truck Modeling and Estimation for Vehicle-to-Vehicle Collision Avoidance Systems")
        ------------
        `Input(s)`:
            steer_ang:  steering angle [rad]
        ------------
        `Output(s)`:
            *{add outputs}
        
        """
        #Vehicle parameters
        vp = self.vp
        #Cornering stiffness coefficients
        C1 = vp.cs[0]
        C2 = vp.cs[1]
        C3 = vp.cs[2]
        C4 = vp.cs[3]
        C5 = vp.cs[4]
        
        #Inertial matrix
        M = np.array([
            [vp.m_t1 + vp.m_t2, -vp.m_t2*(vp.c + vp.d), Vx*(vp.m_t1 + vp.m_t2), -vp.m_t2*vp.d, 0],

            [-vp.m_t2*(vp.c + vp.d), vp.j_zz1 + vp.j_zz2 + vp.m_t2*(vp.c + vp.d)**2, \
            -vp.m_t2*Vx*(vp.c + vp.d), vp.j_zz2 + vp.m_t2*vp.d**2 + vp.m_t2*vp.c*vp.d, 0],

            [0, 0, 1, 0, 0],

            [-vp.m_t2*vp.d, vp.j_zz2 + vp.m_t2*vp.d**2 + vp.m_t2*vp.c*vp.d, -vp.m_t2*Vx*vp.d, \
            vp.j_zz2 + vp.m_t2*vp.d**2, 0],

            [0, 0, 0, 0, 1]
            ])
        
        #Stiffness matrix
        k11 = (1/Vx)*(-C1 - C2 - C3 - C4 - C5)
    
        k12 = (1/Vx)*(-C1*vp.a + C2*vp.b1 + C3*vp.b2 + C4*(vp.c + vp.f1) \
            + C5*(vp.c + vp.f2))
        
        k14 = (1/Vx)*(vp.f1*C4 + vp.f1*C5)
        
        k15 = C4 + C5
        
        k21 = (1/Vx)*(-C1*vp.a + C2*vp.b1 + C3*vp.b2 + C4*(vp.f1 + vp.c) \
            + C5*(vp.f2 + vp.c))
        
        k22 = (1/Vx)*(-C1*vp.a**2 - C2*vp.b1**2 - C3*vp.b2**2 \
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

        #Forcing matrix
        F = np.array([
            [np.cos(steer_ang)*C1],
            [vp.a*np.cos(steer_ang)*C1],
            [0],
            [0],
            [0]
        ])

        #Continuous state space model
        Ac = np.linalg.inv(M) @ K   #State transition matrix
        Bc = np.linalg.inv(M) @ F   #Input matrix
        Cc = np.identity(5)         #Observation matrix
        Dc = 0                      #Measurement input matrix

        sysc = ss(Ac,Bc,Cc,Dc)

        return sysc