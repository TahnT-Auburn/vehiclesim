import numpy as np
from box import Box

def simulate_imu_advanced(
    accel: list,
    gyro: list,
    accel_bias_sigma: tuple[float, float, float],
    accel_bias_tau: tuple[float, float, float],
    accel_rw_sigma: tuple[float, float, float],
    gyro_bias_sigma: tuple[float, float, float],
    gyro_bias_tau: tuple[float, float, float],
    gyro_rw_sigma: tuple[float, float, float],
    dt: float,
    L: int
):
    """Simulates IMU from clean accel and gyro signals with 

    Args:
        accel : ndarray, shape (N, 3)
            Clean acceleration signals [ax, ay, az] in m/s^2
        gyro : ndarray, shape (N, 3)
            Clean gyroscope signals [wx, wy, wz] in rad/s
        accel_bias_sigma : tuple of 3 floats
            Standard deviation of FOGM bias for each accel axis (m/s^2)
        accel_bias_tau : tuple of 3 floats
            Time constant of FOGM bias for each accel axis (seconds)
        accel_rw_sigma : tuple of 3 floats
            Standard deviation of random walk for each accel axis (m/s^2)
        gyro_bias_sigma : tuple of 3 floats
            Standard deviation of FOGM bias for each gyro axis (rad/s)
        gyro_bias_tau : tuple of 3 floats
            Time constant of FOGM bias for each gyro axis (seconds)
        gyro_rw_sigma : tuple of 3 floats
            Standard deviation of random walk for each gyro axis (rad/s)
        dt : float
            Time step in seconds
    """
    
    # Convert tuples to arrays for easier manipulation
    accel = np.array(accel)
    gyro = np.array(gyro)
    accel_bias_sigma = np.array(accel_bias_sigma)
    accel_bias_tau = np.array(accel_bias_tau)
    accel_rw_sigma = np.array(accel_rw_sigma)
    gyro_bias_sigma = np.array(gyro_bias_sigma)
    gyro_bias_tau = np.array(gyro_bias_tau)
    gyro_rw_sigma = np.array(gyro_rw_sigma)
    
    # Initialize output arrays
    accel_noisy = np.zeros_like(accel)
    gyro_noisy = np.zeros_like(gyro)
    
    # Initialize FOGM bias states (starting from steady-state distribution)
    accel_bias = np.random.randn(3) * accel_bias_sigma
    gyro_bias = np.random.randn(3) * gyro_bias_sigma
    
    # FOGM parameters
    accel_phi = np.exp(-dt / accel_bias_tau)  # State transition
    gyro_phi = np.exp(-dt / gyro_bias_tau)
    
    # Process noise for FOGM (to maintain steady-state variance)
    accel_bias_noise_std = accel_bias_sigma * np.sqrt(1 - accel_phi**2)
    gyro_bias_noise_std = gyro_bias_sigma * np.sqrt(1 - gyro_phi**2)
    
    # Simulate IMU measurements
    for i in range(L):
        # Update FOGM bias (First-Order Gauss-Markov process)
        accel_bias = accel_phi * accel_bias + np.random.randn(3) * accel_bias_noise_std
        gyro_bias = gyro_phi * gyro_bias + np.random.randn(3) * gyro_bias_noise_std
        
        # Generate random walk (white noise)
        accel_noise = np.random.randn(3) * accel_rw_sigma
        gyro_noise = np.random.randn(3) * gyro_rw_sigma
        
        # Apply errors to clean signals
        accel_noisy[:,i] = accel[:,i] + accel_bias + accel_noise
        gyro_noisy[:,i] = gyro[:,i] + gyro_bias + gyro_noise

        # add gravity term to Az signal
        accel_noisy[:,2] += 9.81 # positive for NED convention 
    
    imu = Box({
        'accel': 'NaN',\
        'gyro': 'NaN',\
    })
    imu.accel = accel_noisy
    imu.gyro = gyro_noisy
    
    return imu