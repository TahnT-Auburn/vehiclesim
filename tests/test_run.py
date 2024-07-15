#%%
import numpy as np
import matplotlib.pyplot as plt

from vehiclesim.tractor_trailer import TractorTrailer
from python_utilities.plotters_class import Plotters
from vehiclesim.imu_sim import *

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
    
#     plot.matPlot(sig_count='multi', 
#             signals=[[x,ts_data.Ax], [x,ts_data.Ay], [x,ts_data.Az]],
#             label=['Linear Accel IMU (Tractor)', x.name, y.name],
#             legend=['Ax','Ay','Az'])
    
#     plot.matPlot(sig_count='multi', 
#             signals=[[x,ts_data.AVx], [x,ts_data.AVy], [x,ts_data.AVz]],
#             label=['Angular Velocity IMU (Tactor)', x.name, y.name],
#             legend=['Roll','Pitch','Yaw'])
    
#     plot.matPlot(sig_count='multi', 
#             signals=[[x,ts_data.Ax_2], [x,ts_data.Ay_2], [x,ts_data.Az_2]],
#             label=['Linear Accel IMU (Trailer)', x.name, y.name],
#             legend=['Ax','Ay','Az'])
    
#     plot.matPlot(sig_count='multi', 
#             signals=[[x,ts_data.AVx_2], [x,ts_data.AVy_2], [x,ts_data.AVz_2]],
#             label=['Angular Velocity IMU (Trailer)', x.name, y.name],
#             legend=['Roll','Pitch','Yaw'])

#----- IMU SIM TEST -----#

# 3DOF body-fixed linear accelerations (@ tractor CG) [m/s^2]
Ax_bf1 = ts_data.AxBf_SM*9.81
Ay_bf1 = ts_data.AyBf_SM*9.81
Az_bf1 = ts_data.AzBf_SM*9.81

# 3DOF body-fixed angular velocities (@ tractor CG) [rad]
AVx_bf1 = np.deg2rad(ts_data.AVx)
AVy_bf1 = np.deg2rad(ts_data.AVy)
AVz_bf1 = np.deg2rad(ts_data.AVz)

# 3DOF body-fixed linear accelerations (@ trailer CG) [m/s^2]
Ax_bf2 = ts_data.AxBf_SM2*9.81
Ay_bf2 = ts_data.AyBf_SM2*9.81
Az_bf2 = ts_data.AzBf_SM2*9.81

# 3DOF body-fixed angular velocities (@ trailer CG) [rad]
AVx_bf2 = np.deg2rad(ts_data.AVx_2)
AVy_bf2 = np.deg2rad(ts_data.AVy_2)
AVz_bf2 = np.deg2rad(ts_data.AVz_2)


imu_test = imu_sim(1,Ax_bf1,Ay_bf1,Az_bf1,AVx_bf1,AVy_bf1,AVz_bf1,L)

#Plotting
# fig, axs = plt.subplots(2)
# fig.suptitle("Simulated IMU")
# axs[0].plot(ts_data.T_event, imu_test)