'''
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
'''
import pandas as pd
import numpy as np
from python_utilities.parsers_class import Parsers

class TractorTrailer:

    def __init__(self, ts_data_file:str=''):
        '''
        Class Description:
            A class to handle tractor trailer simulations.
        Class Instance Input(s):
            * ts_data:  TruckSim csv data file
                        type: <str>
        '''

        #Input assertions
        assert type(ts_data_file) == str,\
            f"Input <ts_data> has invalid type. Expected <class 'str'> but recieved {type(ts_data_file)}"
        
        #Class objects
        self.ts_data_file = ts_data_file

        #Call class methods
        if (self.ts_data_file != ''):   #TruckSim data parser
            self.parseTsData()

    def parseTsData(self):
        '''
        Description:
            Trucksim data <type: csv> parser (if input is provided).
            Data <type: dict> is then declared as a class variable.
        Input(s):
            None
        Output(s):
            None
        '''
        #Parse TruckSim csv data
        ts_data = Parsers().csvParser(self.ts_data_file)
        ts_data.columns = pd.Index(pd.Series(ts_data.columns).apply(lambda x: x.strip())) #strip extra empty spaces
        self.ts_data = ts_data

    def latModel(self):
        '''
        Description:
            Lateral Bicycle Model Simulation. 
            (Source: S.M. Wolfe, "Heavy Truck Modeling and Estimation for Vehicle-to-Vehicle Collision Avoidance Systems")
        Input(s):
            *{add inputs}
        Output(s):
            *{add outputs}
        
        '''
        pass
