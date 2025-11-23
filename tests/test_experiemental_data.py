#%%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('ipympl')
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyproj import Transformer
from pyproj import CRS
import folium

from vehiclesim.tractor_trailer import TractorTrailer
from postprocessing.standard_state_est_plotter import standard_state_est_plotter
from filter_tools.estimators import Estimators
from genNavMatrices import *

#%%
# Load data
CSV = 'D:\\Tahn\\6_19_25\\csv\\raw\\original\\01\\01.csv'   
df = pd.read_csv(CSV, dtype={'SUBSET':str})
# df = df[500:-1].reset_index()
veh_config_file = 'C:\\Users\\Tahn\\SoftDevel\\vehiclesim\\vehiclesim\\vehicle_configs\\5a_config.yaml'
tract_trail_model = TractorTrailer(veh_config_file, '5a')

L = len(df)

# define variables
t = df["t"]
dt = round(np.mean(np.diff(t)),3)
steer_can = df["steer_ang"]
vx_can = df["vx"]
vx_thresh = 0.1 # velocity threshold to apply zero velocity update
imu_accel_y = df["imu_accel_y"]
imu_gyro_z = df["imu_gyro_z"]
vy_etal = df["vy"]
yaw_etal = df["yaw"]
yaw_rate_etal = df["yaw_rate"]
hitch_etal = df["hitch"]
hitch_rate_etal = df["hitch_rate"]

#%%
# Helper functions
def enu2lla(enu, ref_lla):
    """
    Convert ENU coordinates to LLA coordinates.
    
    Parameters:
    enu: array-like, (3,)
        East, North, Up coordinates.
    ref_lla: array-like, (3,)
        Reference LLA points.
    Returns:
    lla: array-like, (3,)
        Resulting LLA points.
    """
    e = enu[0]
    n = enu[1]
    u = enu[2]

    ref_lat = ref_lla[0]
    ref_lon = ref_lla[1]
    ref_alt = ref_lla[2]

    # Define coordinate systems
    wgs84 = CRS.from_epsg(4326)  # WGS84 coordinate system (lat, lon)
    
    # Create ENU coordinate system at the reference point
    enu = CRS.from_string(
        f"+proj=tmerc +lat_0={ref_lat} +lon_0={ref_lon} +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs")
    
    # Create transformer
    transformer = Transformer.from_crs(enu, wgs84, always_xy=True)
    
    # Transform ENU to LLA
    lon, lat = transformer.transform(e, n)
    
    # Calculate altitude
    alt = ref_alt + u
    
    lla = np.array([lat, lon, alt])
    return lla

#%%
##### Implement full navigation kalman filter #####
def run_nav_kf():

    # storage list
    sysd_nav = []
    x_nav = []
    xdot_nav = []
    P_nav = []
    innov_nav = []
    K_nav = []

    # preallocate states
    X_nav = np.zeros(L)
    vx_nav = np.zeros(L)
    Y_nav = np.zeros(L)
    vy_nav = np.zeros(L)
    yaw_rate_nav = np.zeros(L)
    yaw_nav = np.zeros(L)
    hitch_rate_nav = np.zeros(L)
    hitch_nav = np.zeros(L)
    bias_ar_nav = np.zeros(L)

    # initialize
    x_ = np.array([[0],
                [vx_can[0]],
                [0],
                [vy_etal[0]],
                [yaw_rate_etal[0]],
                [yaw_etal[0]],
                [hitch_rate_etal[0]],
                [hitch_etal[0]],
                [0]])

    X_nav[0] = x_[0].item()
    vx_nav[0] = x_[1].item()
    Y_nav[0] = x_[2].item()
    vy_nav[0] = x_[3].item()
    yaw_rate_nav[0] = x_[4].item()
    yaw_nav[0] = x_[5].item()
    hitch_rate_nav[0] = x_[6].item()
    hitch_nav[0] = x_[7].item()
    bias_ar_nav[0] = x_[8].item()

    P_ = np.diag(np.array([1.10708923e+01, 9.90195136e-03, 1.37791859e+01, 1.76029376e+01,
        1.36849423e-03, 1.01000014e+02, 6.32907841e-01, 3.12263142e-01, 1.37416405e-03]))
    x_nav.append(x_)
    P_nav.append(P_)

    # process noise
    # Q = np.diag([0.1, 1, 0.1, 1, 0.001, 0.0001, 0.001, 0.001, 0.05, 0.001])
          # bias ar

    # Q = np.diag([2,                     # X
    #             0.001,                    # vx    
    #             2,                      # Y
    #             0.01,                    # vy
    #             0.01,                 # yaw rate
    #             0.001,                  # yaw
    #             0.001,                  # hitch_rate
    #             0.01,                  # hitch
    #             1e-6])              # bias ar

    # generate a KF instance
    kfnav = Estimators(n=9,m=2)

    for k in tqdm(range(0,L-1)):
        # process noise
        Q = np.diag([2,                     # X
                    0.001,                    # vx    
                    2,                      # Y
                    0.01,                    # vy
                    0.001,                 # yaw rate
                    0.01,                  # yaw
                    0.001,                  # hitch_rate
                    0.01,                  # hitch
                    1e-6])              # bias ar
        # measurement noise
        R = np.diag([1e-3, 1e-2])

        # === ZUPT ===
        zupt=False
        if vx_can[k+1] <= vx_thresh:
            zupt = True
            # x_ = np.array([
            #     [X_nav[k]],
            #     [0],
            #     [Y_nav[k]],
            #     [0],
            #     [0],
            #     [yaw_nav[k]],
            #     [0],
            #     [hitch_nav[k]],
            #     [0]
            # ])
        
            # A = np.matrix(np.eye(9))
            A = np.diag([1,0,1,0,0,1,0,1,1])
            # A = np.array([[1,0,0,0,0,0,0,0,0],
            #               [0,0,0,0,0,0,0,0,0],
            #               [0,0,1,0,0,0,0,0,0],
            #               [0,0,0,0,0,0,0,0,0],
            #               [0,0,0,0,0,0,0,0,0],
            #               [0,0,0,0,0,1,0,0,0],
            #               [0,0,0,0,0,0,0,0,0],
            #               [0,0,0,0,0,0,0,1,0],
            #               [0,0,0,0,0,0,0,0,1]])
            
            B = np.zeros((9,1))

            # generate full observation matrix
            _, _, H = genNavMatrices(A_veh=np.eye(5), B_veh=np.zeros((5,1)), vx=float(x_[1]), yaw=float(x_[5][0]), dt=dt)

        # === Regular Time Update Setup ===
        else:
            # call vehicle state model
            _, sysd_ = tract_trail_model.latModel(steer_ang=steer_can[k+1], Vx=float(x_[1]), dt=dt)
            A = sysd_.A
            B = sysd_.B

            # generate full navigation matrices
            A, B, H = genNavMatrices(A_veh=A, B_veh=B, vx=float(x_[1]), yaw=float(x_[5][0]), dt=dt)

        # model input
        u = np.array([steer_can[k+1]])

        # if zupt:
        #     z = np.array([[0],
        #                   [0]]) # + float(x_[9])
            # R = np.diag([1e-3, 1e1])
        # else:
        # imu measurements
        z = np.array([[vx_can[k+1]],
                    [imu_gyro_z[k+1]]]) # + float(x_[9])
        
        # warm up for initial P
        if k == 0:          
            for _ in range(100):
                _, P_, K_, innov_ = kfnav.kf(T=dt,
                                    num_inputs=1,
                                    F=A,
                                    B=B,
                                    u=u,
                                    Q=Q,
                                    z=z,
                                    H=H,
                                    R=R,
                                    P=P_,
                                    x=x_)
                stop=1
        stop = 1
        # call KF
        # print(f'zupt:{zupt}, vx: {vx_can[k+1]}', flush=True)    
        x_, P_, K_, innov_ = kfnav.kf(T=dt,
                                    num_inputs=1,
                                    F=A,
                                    B=B,
                                    u=u,
                                    Q=Q,
                                    z=z,
                                    H=H,
                                    R=R,
                                    P=P_,
                                    x=x_)

        x_nav.append(x_)
        P_nav.append(P_)
        K_nav.append(K_)
        innov_nav.append(innov_)

        X_nav[k+1] = x_[0].item()
        vx_nav[k+1] = x_[1].item()
        Y_nav[k+1] = x_[2].item()
        vy_nav[k+1] = x_[3].item()
        yaw_rate_nav[k+1] = x_[4].item()
        yaw_nav[k+1] = x_[5].item()
        hitch_rate_nav[k+1] = x_[6].item()
        hitch_nav[k+1] = x_[7].item()
        bias_ar_nav[k+1] = x_[8].item()

    # nav_states = [X_nav, vx_nav, Y_nav, vy_nav, yaw_rate_nav, yaw_nav, hitch_rate_nav, hitch_nav, bias_ay_nav, bias_ar_nav]
    # filter_states = [
    #     [X_nav, 'X Position', '[m]'],
    #     [vx_nav, 'Body Frame Vx', '[m/s]'],
    #     [Y_nav, 'Y Position', '[m]'],
    #     [vy_nav, 'Body Frame Vy', '[m/s]'],
    #     [np.rad2deg(yaw_rate_nav), 'Yaw Rate', '[deg/s]'],
    #     [np.rad2deg(yaw_nav), 'Yaw Angle', '[deg]'],
    #     [np.rad2deg(hitch_rate_nav), 'Hitch Rate', '[deg/s]'],
    #     [np.rad2deg(hitch_nav), 'Hitch Angle', '[deg]'],]
    filter_states = {
        'X_nav':X_nav,
        'vx_nav':vx_nav,
        'Y_nav':Y_nav,
        'vy_nav':vy_nav,
        'yaw_rate_nav':np.rad2deg(yaw_rate_nav),
        'yaw_nav':np.rad2deg(yaw_nav),
        'hitch_rate_nav':np.rad2deg(hitch_rate_nav),
        'hitch_nav':np.rad2deg(hitch_nav),
        'bias_ar_nav': bias_ar_nav,
    }
    # filter_states = x_nav
    return filter_states



def run_simple_kf():
    ##### kalman filter #####
    # TODO: Figure out a zero velocity uppdate for the KF

    # storage lists
    sysd_cl = []
    x_cl = []
    xdot_cl = []
    P = []
    innov = []
    K = []

    # preallocate states 
    vy_cl = np.zeros(L)
    yaw_rate_cl = np.zeros(L)
    yaw_cl = np.zeros(L)
    hitch_rate_cl = np.zeros(L)
    hitch_cl = np.zeros(L)

    # initialize
    x_ = np.array([
                [vy_etal[0]],
                [yaw_rate_etal[0]],
                [yaw_etal[0]],
                [hitch_rate_etal[0]],
                [hitch_etal[0]],])

    vy_cl[0] = x_[0].item()
    yaw_rate_cl[0] = x_[1].item()
    yaw_cl[0] = x_[2].item()
    hitch_rate_cl[0] = x_[3].item()
    hitch_cl[0] = x_[4].item()

    P_ = np.diag([2.5, 1e-3, 2, 0.3, 8])
    P.append(P_)

    # process noise
    Q = np.array([[1, 0, 0, 0, 0],
                [0, 0.001, 0, 0, 0],
                [0, 0, 0.01, 0, 0],
                [0, 0, 0, 0.001, 0],
                [0, 0, 0, 0, 0.001]])

    # measurement noise
    # R = np.array([[2e4, 0],
    #             [0, 2e-3]])
    R = 2e-3

    # call kalman filter from estimators class
    kf_inst = Estimators(n=5,m=1)
    for k in range(0,L-1):

        # if abs(axle_steer[k]) <= steer_thresh:
        #     steer_ang[k] = axle_steer[k]
        # zero velocity update
        if vx_can[k+1] <= 0.44704*vx_thresh:
            vy_cl[k+1] = 0
            yaw_rate_cl[k+1] = 0
            yaw_cl[k+1] = yaw_cl[k]
            hitch_rate_cl[k+1] = 0
            hitch_cl[k+1] = hitch_cl[k]

            x_ = np.array([[vy_cl[k+1]],
                [yaw_rate_cl[k+1]],
                [yaw_cl[k+1]],
                [hitch_rate_cl[k+1]],
                [hitch_cl[k+1]]])
        
            A = np.matrix(np.eye(5))
            B = np.matrix(np.zeros((5,1)))
            
            # H = np.array([[0, 0, 0, 0, 0],
            #               [0, 1, 0, 0, 0]])
        else:
            # time update
            _, sysd_ = tract_trail_model.latModel(steer_ang=steer_can[k+1], Vx=vx_can[k+1], dt=dt)
            sysd_cl.append(sysd_)
            A = sysd_.A
            B = sysd_.B
            # # measurement map
            # H = np.array([[0, vx_input[k], 0, 0, 0],
            #               [0, 1, 0, 0, 0]])
        
        # model input
        u = np.array([steer_can[k+1]])

        # imu measurements
        # z = np.array([
        #             [imu_accel_y[k+1] ],
        #             [imu_gyro_z[k+1]]])
        z = np.array([imu_gyro_z[k+1]])
        # z = np.array([[vx_input[k]*tractor_imu['angvel'][2][k]],
        #               [tractor_imu['angvel'][2][k]]])
        # measurement map
        # H = np.array([[0, vx_can[k+1], 0, 0, 0],
        #                 [0, 1, 0, 0, 0]])
        H = np.array([[0,1,0,0,0]])
            
        # warm up for initial P
        if k == 0:          
            for _ in range(100):
                _, P_, K_, innov_ = kf_inst.kf(T=dt,
                                        num_inputs=1,
                                        F=A,
                                        B=B,
                                        u=u,
                                        Q=Q,
                                        z=z,
                                        H=H,
                                        R=R,
                                        P=P_,
                                        x=x_)
        # call KF    
        x_, P_, K_, innov_ = kf_inst.kf(T=dt,
                                        num_inputs=1,
                                        F=A,
                                        B=B,
                                        u=u,
                                        Q=Q,
                                        z=z,
                                        H=H,
                                        R=R,
                                        P=P_,
                                        x=x_)
        
        x_cl.append(x_)
        P.append(P_)
        K.append(K_)
        innov.append(innov_)

        vy_cl[k+1] = x_[0].item()
        yaw_rate_cl[k+1] = x_[1].item()
        yaw_cl[k+1] = x_[2].item()
        hitch_rate_cl[k+1] = x_[3].item()
        hitch_cl[k+1] = x_[4].item()
    
    # Manually propagate position
    X_nav = np.zeros(L)
    Y_nav = np.zeros(L)
    X_nav[0] = 0
    Y_nav[0] = 0
    for j in range(0,L-1):
        X_nav[j+1] = X_nav[j] + (vx_can[j]*np.cos(yaw_cl[j]) - vy_cl[j]*np.sin(yaw_cl[j]))*dt
        Y_nav[j+1] = Y_nav[j] + (vx_can[j]*np.sin(yaw_cl[j]) + vy_cl[j]*np.cos(yaw_cl[j]))*dt

    filter_states = {
        'X_nav':X_nav,
        'vx_nav':vx_can,
        'Y_nav':Y_nav,
        'vy_nav':vy_cl,
        'yaw_rate_nav':np.rad2deg(yaw_rate_cl),
        'yaw_nav':np.rad2deg(yaw_cl),
        'hitch_rate_nav':np.rad2deg(hitch_rate_cl),
        'hitch_nav':np.rad2deg(hitch_cl),
        }
    return filter_states


def run_model():
##### kalman filter #####
# TODO: Figure out a zero velocity uppdate for the KF

    # storage lists
    sysc_ol = []
    x_ol = []
    xdot_ol = []
    P = []
    innov = []
    K = []

    # preallocate states 
    vy_ol = np.zeros(L)
    yaw_rate_ol = np.zeros(L)
    yaw_ol = np.zeros(L)
    hitch_rate_ol = np.zeros(L)
    hitch_ol = np.zeros(L)

    # initialize
    x_ = np.array([
                [vy_etal[0]],
                [yaw_rate_etal[0]],
                [yaw_etal[0]],
                [hitch_rate_etal[0]],
                [hitch_etal[0]],])

    vy_ol[0] = x_[0].item()
    yaw_rate_ol[0] = x_[1].item()
    yaw_ol[0] = x_[2].item()
    hitch_rate_ol[0] = x_[3].item()
    hitch_ol[0] = x_[4].item()

    for k in range(0,L-1):
        # if abs(axle_steer[k]) <= steer_thresh:
        #     steer_ang[k] = axle_steer[k]
        # zero velocity update
        if vx_can[k+1] <= 0.44704*vx_thresh:
            vy_ol[k+1] = 0
            yaw_rate_ol[k+1] = 0
            yaw_ol[k+1] = yaw_ol[k]
            hitch_rate_ol[k+1] = 0
            hitch_ol[k+1] = hitch_ol[k]

            x_ = np.array([[vy_ol[k+1]],
                [yaw_rate_ol[k+1]],
                [yaw_ol[k+1]],
                [hitch_rate_ol[k+1]],
                [hitch_ol[k+1]]])
        
            # A = np.matrix(np.eye(5))
            # B = np.matrix(np.zeros((5,1)))
            
        else:
            # time update
            _, sysc_ = tract_trail_model.latModel(steer_ang=steer_can[k+1], Vx=float(x_[1]), dt=dt)
            sysc_ol.append(sysc_)
            A = sysc_.A
            B = sysc_.B

            u = np.array([steer_can[k+1]])
            # xdot_ = A*x_ + B*u
            x_ = A*x_ + B*u
            # xdot_ol.append(xdot_)

            # x_ = x_ + xdot_*dt

            x_ol.append(x_)

            vy_ol[k+1] = x_[0].item()
            yaw_rate_ol[k+1] = x_[1].item()
            yaw_ol[k+1] = x_[2].item()
            hitch_rate_ol[k+1] = x_[3].item()
            hitch_ol[k+1] = x_[4].item()

    # Manually propagate position
    X_nav = np.zeros(L)
    Y_nav = np.zeros(L)
    X_nav[0] = 0
    Y_nav[0] = 0
    for j in range(0,L-1):
        X_nav[j+1] = X_nav[j] + (vx_can[j]*np.cos(yaw_ol[j]) - vy_ol[j]*np.sin(yaw_ol[j]))*dt
        Y_nav[j+1] = Y_nav[j] + (vx_can[j]*np.sin(yaw_ol[j]) + vy_ol[j]*np.cos(yaw_ol[j]))*dt

    filter_states = {
        'X_nav':X_nav,
        'vx_nav':vx_can,
        'Y_nav':Y_nav,
        'vy_nav':vy_ol,
        'yaw_rate_nav':np.rad2deg(yaw_rate_ol),
        'yaw_nav':np.rad2deg(yaw_ol),
        'hitch_rate_nav':np.rad2deg(hitch_rate_ol),
        'hitch_nav':np.rad2deg(hitch_ol),
        }
    return filter_states

if __name__ == '__main__':
    filter_states = run_nav_kf()
    # filter_states = run_simple_kf()
    # filter_states = run_model()

    truth_states = {
        'X_truth':df["X"],
        'vx_truth':df["vx"],
        'Y_truth':df["Y"],
        'vy_truth':df["vy"],
        'yaw_rate_truth':np.rad2deg(df["yaw_rate"]),
        'yaw_truth':np.rad2deg(df["yaw"]),
        'hitch_rate_truth':np.rad2deg(df["hitch_rate"]),
        'hitch_truth':np.rad2deg(df["hitch"])}
    
    # postprocessing
    # x_plot = np.array(filter_states).squeeze().transpose().tolist()
    # x_truth_plot = [df["Y"], df["X"], vx_can, vy_etal, yaw_rate_etal, yaw_etal, hitch_rate_etal, hitch_etal]
    # standard_state_est_plotter(x_plot, x_truth_plot, t, interactive=True)
    
    plt.figure
    plt.plot(truth_states["X_truth"], truth_states["Y_truth"])
    plt.plot(filter_states["X_nav"], filter_states["Y_nav"])
    plt.title("NED Position")
    plt.legend(["Truth", "KF"])
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    plt.figure
    plt.subplot(211)
    plt.plot(t,truth_states["vx_truth"])
    plt.plot(t,filter_states["vx_nav"])
    plt.title("Vx")
    plt.legend(["Truth", "KF"])
    plt.subplot(212)
    plt.plot(t,truth_states["vy_truth"])
    plt.plot(t,filter_states["vy_nav"])
    plt.title("Vy")
    plt.tight_layout()
    plt.show()

    plt.figure
    plt.subplot(211)
    plt.plot(t,truth_states["yaw_truth"])
    plt.plot(t,filter_states["yaw_nav"])
    plt.title("Yaw")
    plt.legend(["Truth", "KF"])
    plt.subplot(212)
    plt.plot(t,truth_states["yaw_rate_truth"])
    plt.plot(t,filter_states["yaw_rate_nav"])
    plt.title("Yaw Rate")
    plt.tight_layout()
    plt.show()

    plt.figure
    plt.subplot(211)
    plt.plot(t,truth_states["hitch_truth"])
    plt.plot(t,filter_states["hitch_nav"])
    plt.title("Htich")
    plt.legend(["Truth", "KF"])
    plt.subplot(212)
    plt.plot(t,truth_states["hitch_rate_truth"])
    plt.plot(t,filter_states["hitch_rate_nav"])
    plt.title("Hitch Rate")
    plt.tight_layout()   
    plt.show()

    plt.figure
    # plt.subplot(211)
    # plt.plot(t,filter_states["bias_ay_nav"])
    # plt.ylabel('Ay Bias Estimate')
    # plt.subplot(212)
    plt.plot(t,filter_states["bias_ar_nav"])
    plt.ylabel('Gyro Bias Estimate')
    plt.tight_layout()
    plt.show()
