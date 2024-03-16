#%%
import numpy as np
import matplotlib.pyplot as plt

from vehiclesim.tractor_trailer import TractorTrailer
from python_utilities.plotters_class import Plotters

if __name__ == '__main__':
    
    #Call instance
    veh_config_file = 'veh_config/tractor_trailer/5a_config.yaml'
    ts_data_file = 'data/Run103.csv'
    test_instance = TractorTrailer(veh_config_file=veh_config_file, config_type='5a', ts_data_file=ts_data_file)

    #Vehicle config
    vp = test_instance.vp

    #TruckSim Data
    ts_data = test_instance.ts_data

    #Simulate open loop dynamics
    #Sims specs
    L = len(ts_data.Time)
    dt = 1/40

    #Inputs
    steer_ang = np.deg2rad((ts_data.Steer_L1 + ts_data.Steer_R1)/2)
    Vx = ts_data.Vx*(1e3/3600)

    #Initialize
    sysc = list()
    x = list()
    xdot = list()
    Vy = np.zeros(L)
    yaw_rate = np.zeros(L)
    yaw = np.zeros(L)
    hitch_rate = np.zeros(L)
    hitch = np.zeros(L)

    x_ = np.array([
        [0],
        [0],
        [0],
        [0],
        [0]])
    
    for i in range(L):

        sysc_ = test_instance.latModel(steer_ang=steer_ang[i], Vx=Vx[i], dt=dt)
        sysc.append(sysc_)

        u = steer_ang[i]
        xdot_ = sysc_.A*x_ + sysc_.B*u
        xdot.append(xdot_)

        x_ = x_ + xdot_*dt
        x.append(x_)

        Vy[i] = x_[0]
        yaw_rate[i] = x_[1]
        yaw[i] = x_[2]
        hitch_rate[i] = x_[3]
        hitch[i] = x_[4]

    #Plotting
    x = ts_data.Time
    y = ts_data.Yaw
    y2 = np.rad2deg(yaw)
    y3 = ts_data.Yaw
    plot = Plotters()
    plot.matPlot(sig_count='multi', signals=[[x,y], [x,y2]],\
            label=['Yaw Rate', x.name, y.name],
            legend=['TruckSim','Model'])