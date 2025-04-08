"""
#################### Tractor Trailer Lateral Navigation Using Experimental Data ####################

    Author: 
        Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn
    
    Description: Tractor Trailer Navigation using experimental data.
#####################################################################################################
"""

#%%
import numpy as np
import scipy.io
from scipy.spatial.transform import Rotation
from scipy import signal
import pandas as pd
import matplotlib
matplotlib.use('ipympl')
import matplotlib.pyplot as plt
from pyproj import Transformer
from pyproj import CRS
import folium
# %matplotlib inline
# import mpld3
# mpld3.enable_notebook()

from vehiclesim.tractor_trailer import TractorTrailer
from vehiclesim.lat_nav_plotter import *
from filter_tools.estimators import Estimators
from genNavMatrices import *

'''
TODO:
- rotate quaternions off the first quat to initialize oerientation to 0's for any given time interval
- figure out a smart way to ensure 
- figure out what etalin twist messages mean (velocity? if so, body-frame? NED frame?) -> figured out! They're body frame velocities
'''

#%%
# set globals
MAT_FILE = 'C:\\Users\\pzt0029\\Documents\\Data\\Trucks\\2025_03_07\\2025_03_07_a2_sensors\\mat\\a2_sensors_2025_03_07_02.mat'
veh_config_file = 'C:\\Users\\pzt0029\\Documents\\Vehicle_Simulations\\vehiclesim\\tests\\veh_config\\tractor_trailer\\5a_config.yaml'

time_range = [60*0,60*20] #NOTE: [start_time, end_time] in seconds
# time_range = None
vx_thresh = 2

sync_etals = False

#%%
# helper functions
def quat2eul(q, sequence='xyz'):
    '''
    Parameters:
        q : array-like, shape (4,) or (N, 4).
            Quaterion(s) in format [x, y, z, w]
        sequence :  str, optional
            The sequence of rotation. Default is 'xyz'.
    
    Returns:
        euler_angles : array, shape (3,) or (N, 3).
            Euler angles in radians in specified sequence
    '''
    # create rotation object from quaternion
    rot = Rotation.from_quat(q)
    # convert to euler angles
    euler_angles = rot.as_euler(sequence, degrees=False)

    return euler_angles


def quatdot(p,q):
    '''
    Parameters:
        p : array-like, shape (4,).
            Reference quaternion in format [w, x, y, z].
        q : array-like, shape (4,).
            target quaternion in format [w, x, y, z].
    '''
    # siphon reference quaternion
    pw = p[0]
    px = -p[1]
    py = -p[2]
    pz = -p[3]
    # create reference matrix
    p_mat = np.array([[pw, -px, -py, -pz],
                      [px, pw, -pz, py],
                      [py, pz, pw, -px],
                      [pz, -py, px, pw]]) 
    # perfrom dot product
    return p_mat @ q


def ecef2lla(x, y, z):
    """
    Convert ECEF coordinates (x, y, z) to LLA coordinates (latitude, longitude, altitude).
    
    Parameters:
        x,y,z : float or array-like.
            ECEF coordinates in meters.
        
    Returns:
        lat : float or array-like.
            Latitude in degrees.
        lon : float or array-like.
            Longitude in degrees.
        alt : float or array-like.
            Altitude in meters above WGS84 ellipsoid.
    """
    # Create a transformer object from ECEF (EPSG:4978) to WGS84 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    # Transform coordinates
    lon, lat, alt = transformer.transform(x, y, z)
    lla = np.squeeze(np.array([[lat], [lon], [alt]]))

    return lla

import numpy as np


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


def wrap_to_pi(angle):
    """
    Wrap angle to [-π, π] range using modulo.
    """
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped

#%%
# load data
mat = scipy.io.loadmat(MAT_FILE)
j1939 = mat['j1939']
tractor_etal = mat['tractor_etal']
trailer_etal = mat['trailer_etal']
# time_range = None

# parse j1939
# steer angle (deg)->(rad)
steer_angle = np.deg2rad((1/25)*(np.squeeze(j1939[0]['steerAngle'][0]['data'][0,0])))
steer_time = np.squeeze(j1939[0]['steerAngle'][0]['time'][0,0]['zeroed'][0,0])
steer_dt = round(np.mean(np.diff(steer_time)), 4)

# vehicle speed (km/h)->(m/s)
vx = np.squeeze(j1939[0]['vehicleSpeed'][0]['data'][0,0])*(1e3/3600)
vx_time = np.squeeze(j1939[0]['vehicleSpeed'][0]['time'][0,0]['zeroed'][0,0])
vx_dt = round(np.mean(np.diff(vx_time)), 4)

# parse etalin
# TODO: Find a way to guarantee unamious data points between both etalins and IMUs.

# tractor etalin imu
tractor_imu = {'linAccel': {'x': tractor_etal[0]['imu'][0]['linAccel'][0,0][0],
                            'y': tractor_etal[0]['imu'][0]['linAccel'][0,0][1],
                            'z': tractor_etal[0]['imu'][0]['linAccel'][0,0][2]},
               'angvel': {'x': tractor_etal[0]['imu'][0]['angvel'][0,0][0],
                          'y': tractor_etal[0]['imu'][0]['angvel'][0,0][1],
                          'z': tractor_etal[0]['imu'][0]['angvel'][0,0][2]}
}
# tractor etalin position
tractor_position = ecef2lla(x=tractor_etal[0]['position'][0][0],
                            y=tractor_etal[0]['position'][0][1],
                            z=tractor_etal[0]['position'][0][2])
tractor_position = {'lat': tractor_position[0,:],
                    'lon': tractor_position[1,:],
                    'alt': tractor_position[2,:]}
# tractor etalin orientation
tractor_orientation = quat2eul(tractor_etal[0]['orientation'][0].transpose()).transpose()
tractor_orientation = {'roll': wrap_to_pi(tractor_orientation[0]),
                       'pitch': wrap_to_pi(tractor_orientation[1]),
                       'yaw': wrap_to_pi(tractor_orientation[2])}
# tractor etalin velocity
tractor_velocity = tractor_etal[0]['linTwist'][0]
tractor_velocity = {'x': tractor_velocity[0],
                    'y': tractor_velocity[1],
                    'z': tractor_velocity[2]}
# tractor etalin time
tractor_etal_time = np.squeeze(tractor_etal[0]['time'][0]['zeroed'][0,0])
# tractor etalin dt
tractor_etal_dt = round(np.mean(np.diff(tractor_etal_time)), 6)


# trailer etalin imu
trailer_imu = {'linAccel': {'x': trailer_etal[0]['imu'][0]['linAccel'][0,0][0],
                            'y': trailer_etal[0]['imu'][0]['linAccel'][0,0][1],
                            'z': trailer_etal[0]['imu'][0]['linAccel'][0,0][2]},
               'angvel': {'x': trailer_etal[0]['imu'][0]['angvel'][0,0][0],
                          'y': trailer_etal[0]['imu'][0]['angvel'][0,0][1],
                          'z': trailer_etal[0]['imu'][0]['angvel'][0,0][2]}
}
# trailer etalin position
trailer_position = ecef2lla(x=trailer_etal[0]['position'][0][0],
                            y=trailer_etal[0]['position'][0][1],
                            z=trailer_etal[0]['position'][0][2])
trailer_position = {'lat': trailer_position[0,:],
                    'lon': trailer_position[1,:],
                    'alt': trailer_position[2,:]}
# trailer etalin orientation
trailer_orientation = quat2eul(trailer_etal[0]['orientation'][0].transpose()).transpose()
trailer_orientation = {'roll': wrap_to_pi(trailer_orientation[0]),
                       'pitch': wrap_to_pi(trailer_orientation[1]),
                       'yaw': wrap_to_pi(trailer_orientation[2])}
# trailer etalin velocity
trailer_velocity = trailer_etal[0]['linTwist'][0]
trailer_velocity = {'x': trailer_velocity[0],
                    'y': trailer_velocity[1],
                    'z': trailer_velocity[2]}
# trailer etalin time
trailer_etal_time = np.squeeze(trailer_etal[0]['time'][0]['zeroed'][0,0])
# trailer etalin dt
trailer_etal_dt = round(np.mean(np.diff(trailer_etal_time)), 6)

# adjust tractor imu to match time
# tractor_imu['linAccel'] = np.concatenate((tractor_imu['linAccel'],np.zeros((3,1))), axis=1)
# tractor_imu['angvel'] = np.concatenate((tractor_imu['angvel'],np.zeros((3,1))), axis=1)
# trailer_imu['linAccel'] = np.concatenate((trailer_imu['linAccel'],np.zeros((3,1))), axis=1)
# trailer_imu['angvel'] = np.concatenate((trailer_imu['angvel'],np.zeros((3,1))), axis=1)

# sync etalins
if sync_etals:
    # sync trailer time to tractor time
    trailer_etal_time = signal.resample(trailer_etal_time, len(tractor_etal_time))
    # sync trailer signals to new trailer time
    trailer_imu['linAccel']['x'] = signal.resample(trailer_imu['linAccel']['x'], len(trailer_etal_time))
    trailer_imu['linAccel']['y'] = signal.resample(trailer_imu['linAccel']['y'], len(trailer_etal_time))
    trailer_imu['linAccel']['z'] = signal.resample(trailer_imu['linAccel']['z'], len(trailer_etal_time))
    trailer_imu['angvel']['x'] = signal.resample(trailer_imu['angvel']['x'], len(trailer_etal_time))
    trailer_imu['angvel']['y'] = signal.resample(trailer_imu['angvel']['y'], len(trailer_etal_time))
    trailer_imu['angvel']['z'] = signal.resample(trailer_imu['angvel']['z'], len(trailer_etal_time))

    trailer_position['lat'] = signal.resample(trailer_position['lat'], len(trailer_etal_time))
    trailer_position['lon'] = signal.resample(trailer_position['lon'], len(trailer_etal_time))
    trailer_position['alt'] = signal.resample(trailer_position['alt'], len(trailer_etal_time))

    trailer_orientation['roll'] = signal.resample(trailer_orientation['roll'], len(trailer_etal_time))
    trailer_orientation['pitch'] = signal.resample(trailer_orientation['pitch'], len(trailer_etal_time))
    trailer_orientation['yaw'] = signal.resample(trailer_orientation['yaw'], len(trailer_etal_time))

    trailer_velocity['x'] = signal.resample(trailer_velocity['x'], len(trailer_etal_time))
    trailer_velocity['y'] = signal.resample(trailer_velocity['y'], len(trailer_etal_time))
    trailer_velocity['z'] = signal.resample(trailer_velocity['z'], len(trailer_etal_time))

    # sync tractor imus with time
    tractor_imu['linAccel']['x'] = signal.resample(tractor_imu['linAccel']['x'], len(tractor_etal_time))
    tractor_imu['linAccel']['y'] = signal.resample(tractor_imu['linAccel']['y'], len(tractor_etal_time))
    tractor_imu['linAccel']['z'] = signal.resample(tractor_imu['linAccel']['z'], len(tractor_etal_time))
    tractor_imu['angvel']['x'] = signal.resample(tractor_imu['angvel']['x'], len(tractor_etal_time))
    tractor_imu['angvel']['y'] = signal.resample(tractor_imu['angvel']['y'], len(tractor_etal_time))
    tractor_imu['angvel']['z'] = signal.resample(tractor_imu['angvel']['z'], len(tractor_etal_time))

else: 
    # sync tractor imus with time
    tractor_imu['linAccel']['x'] = signal.resample(tractor_imu['linAccel']['x'], len(tractor_etal_time))
    tractor_imu['linAccel']['y'] = signal.resample(tractor_imu['linAccel']['y'], len(tractor_etal_time))
    tractor_imu['linAccel']['z'] = signal.resample(tractor_imu['linAccel']['z'], len(tractor_etal_time))
    tractor_imu['angvel']['x'] = signal.resample(tractor_imu['angvel']['x'], len(tractor_etal_time))
    tractor_imu['angvel']['y'] = signal.resample(tractor_imu['angvel']['y'], len(tractor_etal_time))
    tractor_imu['angvel']['z'] = signal.resample(tractor_imu['angvel']['z'], len(tractor_etal_time))

    # sync trailer signals to new trailer time
    trailer_imu['linAccel']['x'] = signal.resample(trailer_imu['linAccel']['x'], len(trailer_etal_time))
    trailer_imu['linAccel']['y'] = signal.resample(trailer_imu['linAccel']['y'], len(trailer_etal_time))
    trailer_imu['linAccel']['z'] = signal.resample(trailer_imu['linAccel']['z'], len(trailer_etal_time))
    trailer_imu['angvel']['x'] = signal.resample(trailer_imu['angvel']['x'], len(trailer_etal_time))
    trailer_imu['angvel']['y'] = signal.resample(trailer_imu['angvel']['y'], len(trailer_etal_time))
    trailer_imu['angvel']['z'] = signal.resample(trailer_imu['angvel']['z'], len(trailer_etal_time))

# filter data if time range given
if time_range is not None:
    
    # steer angle
    time_mask = (steer_time >= time_range[0]) & (steer_time <= time_range[1])
    steer_angle = steer_angle[time_mask]
    steer_time = steer_time[time_mask]

    # velocity
    time_mask = (vx_time >= time_range[0]) & (vx_time <= time_range[1])
    vx = vx[time_mask]
    vx_time = vx_time[time_mask]

    # tractor etalin
    time_mask = (tractor_etal_time >= time_range[0]) & (tractor_etal_time <= time_range[1])

    tractor_imu['linAccel']['x'] = tractor_imu['linAccel']['x'][time_mask]
    tractor_imu['linAccel']['y'] = tractor_imu['linAccel']['y'][time_mask]
    tractor_imu['linAccel']['z'] = tractor_imu['linAccel']['z'][time_mask]
    tractor_imu['angvel']['x'] = tractor_imu['angvel']['x'][time_mask]
    tractor_imu['angvel']['y'] = tractor_imu['angvel']['y'][time_mask]
    tractor_imu['angvel']['z'] = tractor_imu['angvel']['z'][time_mask]

    tractor_orientation['roll'] = tractor_orientation['roll'][time_mask]
    tractor_orientation['pitch'] = tractor_orientation['pitch'][time_mask]
    tractor_orientation['yaw'] = tractor_orientation['yaw'][time_mask]

    tractor_velocity['x'] = tractor_velocity['x'][time_mask]
    tractor_velocity['y'] = tractor_velocity['y'][time_mask]
    tractor_velocity['z'] = tractor_velocity['z'][time_mask]

    tractor_position['lat'] = tractor_position['lat'][time_mask]
    tractor_position['lon'] = tractor_position['lon'][time_mask]
    tractor_position['alt'] = tractor_position['alt'][time_mask]

    tractor_etal_time = tractor_etal_time[time_mask]

    # trailer etalin
    time_mask = (trailer_etal_time >= time_range[0]) & (trailer_etal_time <= time_range[1])
    trailer_imu['linAccel']['x'] = trailer_imu['linAccel']['x'][time_mask]
    trailer_imu['linAccel']['y'] = trailer_imu['linAccel']['y'][time_mask]
    trailer_imu['linAccel']['z'] = trailer_imu['linAccel']['z'][time_mask]
    trailer_imu['angvel']['x'] = trailer_imu['angvel']['x'][time_mask]
    trailer_imu['angvel']['y'] = trailer_imu['angvel']['y'][time_mask]
    trailer_imu['angvel']['z'] = trailer_imu['angvel']['z'][time_mask]

    trailer_orientation['roll'] = trailer_orientation['roll'][time_mask]
    trailer_orientation['pitch'] = trailer_orientation['pitch'][time_mask]
    trailer_orientation['yaw'] = trailer_orientation['yaw'][time_mask]

    trailer_velocity['x'] = trailer_velocity['x'][time_mask]
    trailer_velocity['y'] = trailer_velocity['y'][time_mask]
    trailer_velocity['z'] = trailer_velocity['z'][time_mask]

    trailer_position['lat'] = trailer_position['lat'][time_mask]
    trailer_position['lon'] = trailer_position['lon'][time_mask]
    trailer_position['alt'] = trailer_position['alt'][time_mask]

    trailer_etal_time = trailer_etal_time[time_mask]

# etalin hitch angle/hitch rate
etal_hitch = wrap_to_pi(trailer_orientation['yaw'] - tractor_orientation['yaw'])
etal_hitch_rate = trailer_imu['angvel']['z'] - tractor_imu['angvel']['z']

#%%
# visualize data

# j1939 data
ax1 = plt.subplot(211)
ax1.plot(steer_time/60, np.rad2deg(steer_angle))
ax1.set_ylabel('Tire Steer Angle (deg)')
ax2 = plt.subplot(212)
ax2.plot(vx_time/60, vx)
ax2.set_ylabel('Longitudinal Velocity (m/s)')
ax2.set_xlabel('Time (min)')
plt.tight_layout()
plt.show()

# tractor imu
ax1 = plt.subplot(211)
ax1.plot(tractor_etal_time/60, tractor_imu['linAccel']['x'], linewidth=1)
ax1.plot(tractor_etal_time/60, tractor_imu['linAccel']['y'], linewidth=1)
ax1.plot(tractor_etal_time/60, tractor_imu['linAccel']['z'], linewidth=1)
ax1.set_ylabel('Linear Acceleration (m/s^2)')
ax2 = plt.subplot(212)
ax2.plot(tractor_etal_time/60, tractor_imu['angvel']['x'], linewidth=1)
ax2.plot(tractor_etal_time/60, tractor_imu['angvel']['y'], linewidth=1)
ax2.plot(tractor_etal_time/60, tractor_imu['angvel']['z'], linewidth=1)
ax2.set_ylabel('Angular Velocity (rad/s)')
plt.tight_layout()
plt.show()

# tractor orientation
ax1 = plt.subplot(311)
ax1.plot(tractor_etal_time/60, np.rad2deg(tractor_orientation['roll']))
ax1.set_ylabel('Roll')
ax2 = plt.subplot(312)
ax2.plot(tractor_etal_time/60, np.rad2deg(tractor_orientation['pitch']))
ax2.set_ylabel('Pitch')
ax3 = plt.subplot(313)
ax3.plot(tractor_etal_time/60, np.rad2deg(tractor_orientation['yaw']))
ax3.set_ylabel('Yaw')
ax3.set_xlabel('')
plt.tight_layout()
plt.show()

# global position
cords = [(lat,lon) for lat,lon in zip(tractor_position['lat'], tractor_position['lon'])]
map = folium.Map(location=[cords[0][0],cords[0][1]], zoom_start=11)
folium.PolyLine(cords).add_to(map)

folium.Marker(
    location=[cords[0][0], cords[0][1]],
    tooltip='Start',
    icon=folium.Icon(icon='play', color='green')
).add_to(map)

folium.Marker(
    location=[cords[-1][0], cords[-1][1]],
    tooltip='End',
    icon=folium.Icon(icon='stop', color='red')
).add_to(map)
map

#%%
###### Simulate bicycle model only #####

tract_trail = TractorTrailer(veh_config_file=veh_config_file, config_type='5a')
# steer_input = signal.resample(steer_angle, len(vx))
# dt = vx_dt
# L = len(vx)

# tray interpolating everything up to etaling time
x_new = tractor_etal_time
dt = tractor_etal_dt
steer_input = np.interp(x_new, steer_time, steer_angle)
vx_input = np.interp(x_new, vx_time, vx)
L = len(tractor_etal_time)

# storage lists
sysc_ol = []
x_ol = []
xdot_ol = []

# initialize
vy_ol = np.zeros(L)
yaw_rate_ol = np.zeros(L)
yaw_ol = np.zeros(L)
hitch_rate_ol = np.zeros(L)
hitch_ol = np.zeros(L)

x_ = np.array([[tractor_velocity['y'][0]],
               [tractor_imu['angvel']['z'][0]],
               [tractor_orientation['yaw'][0]],
               [etal_hitch_rate[0]],
               [etal_hitch[0]]])

vy_ol[0] = x_[0]
yaw_rate_ol[0] = x_[1]
yaw_ol[0] = x_[2]
hitch_rate_ol[0] = x_[3]
hitch_ol[0] = x_[4]

for i in range(0,L-1):

    # zero velocity update
    if vx_input[i+1] <= 0.44704*vx_thresh:
        vy_ol[i+1] = 0
        yaw_rate_ol[i+1] = 0
        yaw_ol[i+1] = yaw_ol[i]
        hitch_rate_ol[i+1] = 0
        hitch_ol[i+1] = hitch_ol[i]

        x_ = np.array([[vy_ol[i+1]],
            [yaw_rate_ol[i+1]],
            [yaw_ol[i+1]],
            [hitch_rate_ol[i+1]],
            [hitch_ol[i+1]]])

        # A = np.eye(5)
        # B = np.zeros((5,1))
    else:
        sysc_, _ = tract_trail.latModel(steer_ang=steer_input[i+1], Vx=vx_input[i+1], dt=dt)
        # sysc_ol.append(sysd_)
        A = sysc_.A
        B = sysc_.B

        u = np.array([steer_input[i+1]])
        xdot_ = A*x_ + B*u
        xdot_ol.append(xdot_)

        x_ = x_ + xdot_*dt
        x_ol.append(x_)

        vy_ol[i+1] = x_[0]
        yaw_rate_ol[i+1] = x_[1]
        yaw_ol[i+1] = wrap_to_pi(x_[2])
        hitch_rate_ol[i+1] = x_[3]
        hitch_ol[i+1] = wrap_to_pi(x_[4])

ol_states = [vy_ol, yaw_rate_ol, yaw_ol, hitch_rate_ol, hitch_ol]

#%%
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
x_ = np.array([[tractor_velocity['y'][0]],
               [tractor_imu['angvel']['z'][0]],
               [tractor_orientation['yaw'][0]],
               [etal_hitch_rate[0]],
               [etal_hitch[0]]])

vy_cl[0] = x_[0]
yaw_rate_cl[0] = x_[1]
yaw_cl[0] = x_[2]
hitch_rate_cl[0] = x_[3]
hitch_cl[0] = x_[4]

P_ = np.diag([2.5, 1e-3, 2, 0.3, 8])
P.append(P_)

# process noise
Q = np.array([[1, 0, 0, 0, 0],
              [0, 0.01, 0, 0, 0],
              [0, 0, 0.1, 0, 0],
              [0, 0, 0, 0.001, 0],
              [0, 0, 0, 0, 0.001]])

# measurement noise
R = np.array([[1e2, 0],
              [0, 1e-3]])

# call kalman filter from estimators class
kf_inst = Estimators(n=5,m=2)
for k in range(0,L-1):

    # if abs(axle_steer[k]) <= steer_thresh:
    #     steer_ang[k] = axle_steer[k]
    # zero velocity update
    if vx_input[k+1] <= 0.44704*vx_thresh:
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
        _, sysd_ = tract_trail.latModel(steer_ang=steer_input[k+1], Vx=vx_input[k+1], dt=dt)
        sysd_cl.append(sysd_)
        A = sysd_.A
        B = sysd_.B
        # # measurement map
        # H = np.array([[0, vx_input[k], 0, 0, 0],
        #               [0, 1, 0, 0, 0]])
    
    # model input
    u = np.array([steer_input[k+1]])

    # imu measurements
    z = np.array([[tractor_imu['linAccel']['y'][k+1]],
                  [tractor_imu['angvel']['z'][k+1]]])
    # z = np.array([[vx_input[k]*tractor_imu['angvel'][2][k]],
    #               [tractor_imu['angvel'][2][k]]])
    # measurement map
    H = np.array([[0, vx_input[k+1], 0, 0, 0],
                    [0, 1, 0, 0, 0]])

        
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

    vy_cl[k+1] = x_[0]
    yaw_rate_cl[k+1] = x_[1]
    yaw_cl[k+1] = wrap_to_pi(x_[2])
    hitch_rate_cl[k+1] = x_[3]
    hitch_cl[k+1] = wrap_to_pi(x_[4])

cl_states = [vy_cl, yaw_rate_cl, yaw_cl, hitch_rate_cl, hitch_cl]

#%%
##### position propagation #####

# absolute distance error helper function
def calc_abs_pos_error(truth, est):
    x_truth = truth[0]
    y_truth = truth[1]
    x_est = est[0]
    y_est = est[1]
    pos_error = np.sqrt((x_truth - x_est)**2 + (y_truth - y_est)**2)
    return pos_error

# initialize
X_mod = np.zeros(L)
Y_mod = np.zeros(L)
X_mod[0] = 0
Y_mod[0] = 0

X_kf = np.zeros(L)
Y_kf = np.zeros(L)
X_kf[0] = 0
Y_kf[0] = 0

X_sens = np.zeros(L)
Y_sens = np.zeros(L)
X_sens[0] = 0
Y_sens[0] = 0

for j in range(0,L-1):

    X_mod[j+1] = X_mod[j] + (vx_input[j]*np.cos(yaw_ol[j]) - vy_ol[j]*np.sin(yaw_ol[j]))*dt
    Y_mod[j+1] = Y_mod[j] + (vx_input[j]*np.sin(yaw_ol[j]) + vy_ol[j]*np.cos(yaw_ol[j]))*dt

    X_kf[j+1] = X_kf[j] + (vx_input[j]*np.cos(yaw_cl[j]) - vy_cl[j]*np.sin(yaw_cl[j]))*dt
    Y_kf[j+1] = Y_kf[j] + (vx_input[j]*np.sin(yaw_cl[j]) + vy_cl[j]*np.cos(yaw_cl[j]))*dt

    X_sens[j+1] = X_sens[j] + (vx_input[j]*np.cos(tractor_orientation['yaw'][j]) - tractor_velocity['y'][j]*np.sin(tractor_orientation['yaw'][j]))*dt
    Y_sens[j+1] = Y_sens[j] + (vx_input[j]*np.sin(tractor_orientation['yaw'][j]) +  tractor_velocity['y'][j]*np.cos(tractor_orientation['yaw'][j]))*dt

    # X_kf[j+1] = X_kf[j] + (vx_input[j]*np.cos(yaw_cl[j]) - tractor_velocity[1][j]*np.sin(yaw_cl[j]))*dt
    # Y_kf[j+1] = Y_kf[j] + (vx_input[j]*np.sin(yaw_cl[j]) + tractor_velocity[1][j]*np.cos(yaw_cl[j]))*dt

# convert tangent frame to LLA
ref_lla = [tractor_position['lat'][0], tractor_position['lon'][0], tractor_position['alt'][0]]

enu_mod = [Y_mod, X_mod, np.zeros(len(X_mod))]
lla_mod = enu2lla(enu=enu_mod, ref_lla=ref_lla)

enu_kf = [Y_kf, X_kf, np.zeros(len(X_kf))]
lla_kf = enu2lla(enu=enu_kf, ref_lla=ref_lla)

enu_sens = [Y_sens, X_sens, np.zeros(len(X_sens))]
lla_sens = enu2lla(enu=enu_sens, ref_lla=ref_lla)

#%%
##### Implement full navigation kalman filter #####

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
bias_ay_nav = np.zeros(L)
bias_ar_nav = np.zeros(L)

# initialize
x_ = np.array([[0],
               [vx[0]],
               [0],
                [tractor_velocity['y'][0]],
               [tractor_imu['angvel']['z'][0]],
               [tractor_orientation['yaw'][0]],
               [etal_hitch_rate[0]],
               [etal_hitch[0]],
               [0],
               [0]])

X_nav[0] = x_[0]
vx_nav[0] = x_[1]
Y_nav[0] = x_[2]
vy_nav[0] = x_[3]
yaw_rate_nav[0] = x_[4]
yaw_nav[0] = x_[5]
hitch_rate_nav[0] = x_[6]
hitch_nav[0] = x_[7]
bias_ay_nav[0] = x_[8]
bias_ar_nav[0] = x_[9]

P_ = np.diag(np.array([1.10708923e+01, 9.90195136e-03, 1.37791859e+01, 1.76029376e+01,
       1.36849423e-03, 1.01000014e+02, 6.32907841e-01, 3.12263142e-01,
       3.70107724e-01, 1.37416405e-03]))
P_nav.append(P_)

# process noise
# Q = np.diag([0.1, 1, 0.1, 1, 0.001, 0.0001, 0.001, 0.001, 0.05, 0.001])
# Q = np.diag([2,                     # X
#             0.1,                    # vx    
#             2,                      # Y
#             1,                    # vy
#             0.001,                 # yaw rate
#             0.1,                  # yaw
#             0.0001,                  # hitch_rate
#             0.0001,                  # hitch
#             1e-6,                   # bias ay
#             1e-6])                 # bias ar

Q = np.diag([0.1,                     # X
            0.5,                    # vx    
            0.1,                      # Y
            1,                    # vy
            0.01,                 # yaw rate
            0.1,                  # yaw
            0.001,                  # hitch_rate
            0.001,                  # hitch
            1e-6,                   # bias ay
            1e-6])                 # bias ar

# measurement noise
R = np.diag([1e-2, 1e2, 1e-3])

# generate a KF instance
kfnav = Estimators(n=10,m=3)

for k in range(0,L-1):

    # time uppdate
    if vx_input[k+1] <= 0.44704*vx_thresh:
        X_nav[k+1] = X_nav[k]
        vx_nav[k+1] = 0
        Y_nav[k+1] = Y_nav[k]
        vy_nav[k+1] = 0
        yaw_rate_nav[k+1] = 0
        yaw_nav[k+1] = yaw_nav[k]
        hitch_rate_nav[k+1] = 0
        hitch_nav[k+1] = hitch_nav[k]
        bias_ay_nav[k+1] = bias_ay_nav[k]
        bias_ar_nav[k+1] = bias_ar_nav[k]
        # bias_ay_nav[k] = 0
        # bias_ar_nav[k] = 0

        x_ = np.array([
            [X_nav[k+1]],
            [vx_nav[k+1]],
            [Y_nav[k+1]],
            [vy_nav[k+1]],
            [yaw_rate_nav[k+1]],
            [yaw_nav[k+1]],
            [hitch_rate_nav[k+1]],
            [hitch_nav[k+1]],
            [bias_ay_nav[k+1]],
            [bias_ar_nav[k+1]]
        ])
    
        A = np.matrix(np.eye(10))
        B = np.matrix(np.zeros((10,1)))

        # generate full observation matrix
        _, _, H = genNavMatrices(A_veh=np.eye(5), B_veh=np.zeros((5,1)), vx=float(x_[1]), yaw=float(x_[5][0]), dt=dt)

    else:
        # call vehicle state model
        _, sysd_ = tract_trail.latModel(steer_ang=steer_input[k+1], Vx=float(x_[1]), dt=dt)
        A = sysd_.A
        B = sysd_.B

        # generate full navigation matrices
        A, B, H = genNavMatrices(A_veh=A, B_veh=B, vx=float(x_[1]), yaw=float(x_[5][0]), dt=dt)

    # model input
    u = np.array([steer_input[k+1]])

    # imu measurements
    z = np.array([[vx_input[k+1]],
                  [tractor_imu['linAccel']['y'][k+1]],
                  [tractor_imu['angvel']['z'][k+1]]])
    
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
    # call KF    
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

    X_nav[k+1] = x_[0]
    vx_nav[k+1] = x_[1]
    Y_nav[k+1] = x_[2]
    vy_nav[k+1] = x_[3]
    yaw_rate_nav[k+1] = x_[4]
    yaw_nav[k+1] = x_[5]
    hitch_rate_nav[k+1] = x_[6]
    hitch_nav[k+1] = x_[7]
    bias_ay_nav[k+1] = x_[8]
    bias_ar_nav[k+1] = x_[9]

nav_states = [X_nav, vx_nav, Y_nav, vy_nav, yaw_rate_nav, yaw_nav, hitch_rate_nav, hitch_nav, bias_ay_nav, bias_ar_nav]

enu_nav = [Y_nav, X_nav, np.zeros(len(X_nav))]
lla_nav = enu2lla(enu=enu_nav, ref_lla=ref_lla)

# %%
# plots
truth_states = [
    tractor_velocity['y'],
    np.rad2deg(tractor_imu['angvel']['z']),
    np.rad2deg(tractor_orientation['yaw']),
    np.rad2deg(etal_hitch_rate),
    np.rad2deg(etal_hitch),
]

nav_states_veh = [vy_nav, yaw_rate_nav, wrap_to_pi(yaw_nav), hitch_rate_nav, hitch_nav]

# call plotter functions
plot_states(tractor_etal_time, truth_states, ol_states, cl_states, nav_states_veh, t_factor=60)
plot_estimator_specs(tractor_etal_time, P=P_nav, 
                    labels=['X','Vx','Y','Vy','Yaw Rate','Yaw','Hitch Rate','Hitch','Ay Bias','Yaw Rate Bias'],
                    t_factor=60)
plot_estimator_specs(tractor_etal_time, P=P,
                     labels=['Vy','Yaw Rate','Yaw','Hitch Rate','Hitch'],
                     t_factor=60)

# model coordinates
map = folium.Map(location=[cords[0][0],cords[0][1]], zoom_start=11)

folium.PolyLine(cords, color='cyan').add_to(map)
folium.Marker(
    location=[cords[0][0], cords[0][1]],
    tooltip='Start',
    icon=folium.Icon(icon='play', color='green')
).add_to(map)

folium.Marker(
    location=[cords[-1][0], cords[-1][1]],
    tooltip='Truth End',
    icon=folium.Icon(icon='stop', color='red')
).add_to(map)

# cords_mod = [(lat,lon) for lat,lon in zip(lla_mod[0,:], lla_mod[1,:])]
# folium.PolyLine(cords_mod, color='red').add_to(map)

# folium.Marker(
#     location=[cords_mod[0][0], cords_mod[0][1]],
#     tooltip='Start',
#     icon=folium.Icon(icon='play', color='green')
# ).add_to(map)

# folium.Marker(
#     location=[cords_mod[-1][0], cords_mod[-1][1]],
#     tooltip='Model End',
#     icon=folium.Icon(icon='stop', color='red')
# ).add_to(map)

# model coordinates
cords_kf = [(lat,lon) for lat,lon in zip(lla_kf[0,:], lla_kf[1,:])]
folium.PolyLine(cords_kf, color='black').add_to(map)

folium.Marker(
    location=[cords_kf[0][0], cords_kf[0][1]],
    tooltip='Start',
    icon=folium.Icon(icon='play', color='green')
).add_to(map)

folium.Marker(
    location=[cords_kf[-1][0], cords_kf[-1][1]],
    tooltip='KF End',
    icon=folium.Icon(icon='stop', color='red')
).add_to(map)

cords_nav = [(lat,lon) for lat,lon in zip(lla_nav[0,:], lla_nav[1,:])]
folium.PolyLine(cords_nav, color='magenta').add_to(map)

folium.Marker(
    location=[cords_nav[0][0], cords_nav[0][1]],
    tooltip='Start',
    icon=folium.Icon(icon='play', color='green')
).add_to(map)

folium.Marker(
    location=[cords_nav[-1][0], cords_nav[-1][1]],
    tooltip='Nav End',
    icon=folium.Icon(icon='stop', color='red')
).add_to(map)

# cords_sens = [(lat,lon) for lat,lon in zip(lla_sens[0,:], lla_sens[1,:])]
# folium.PolyLine(cords_sens, color='orange').add_to(map)

# folium.Marker(
#     location=[cords_sens[0][0], cords_sens[0][1]],
#     tooltip='Start',
#     icon=folium.Icon(icon='play', color='green')
# ).add_to(map)

# folium.Marker(
#     location=[cords_sens[-1][0], cords_sens[-1][1]],
#     tooltip='Sensor Mech End',
#     icon=folium.Icon(icon='stop', color='red')
# ).add_to(map)

map


# %%
