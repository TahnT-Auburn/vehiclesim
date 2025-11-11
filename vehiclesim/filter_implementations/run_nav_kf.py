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
    bias_ay_nav = np.zeros(L)
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
                [0],
                [0]])

    X_nav[0] = x_[0].item()
    vx_nav[0] = x_[1].item()
    Y_nav[0] = x_[2].item()
    vy_nav[0] = x_[3].item()
    yaw_rate_nav[0] = x_[4].item()
    yaw_nav[0] = x_[5].item()
    hitch_rate_nav[0] = x_[6].item()
    hitch_nav[0] = x_[7].item()
    bias_ay_nav[0] = x_[8].item()
    bias_ar_nav[0] = x_[9].item()

    P_ = np.diag(np.array([1.10708923e+01, 9.90195136e-03, 1.37791859e+01, 1.76029376e+01,
        1.36849423e-03, 1.01000014e+02, 6.32907841e-01, 3.12263142e-01,
        3.70107724e-01, 1.37416405e-03]))
    P_nav.append(P_)

    # process noise
    # Q = np.diag([0.1, 1, 0.1, 1, 0.001, 0.0001, 0.001, 0.001, 0.05, 0.001])
    # Q = np.diag([2,                     # X
    #             0.001,                    # vx    
    #             2,                      # Y
    #             0.2,                    # vy
    #             5e-4,                 # yaw rate
    #             1e2,                  # yaw
    #             0.001,                  # hitch_rate
    #             0.0001,                  # hitch
    #             1e-6,                   # bias ay
    #             1e-6])                 # bias ar

    Q = np.diag([2,                     # X
                0.1,                    # vx    
                2,                      # Y
                0.1,                    # vy
                0.001,                 # yaw rate
                0.01,                  # yaw
                0.001,                  # hitch_rate
                0.001,                  # hitch
                1e-6,                   # bias ay
                1e-6])                 # bias ar

    # measurement noise
    R = np.diag([1e-2, 3e4, 2e-3])

    # generate a KF instance
    kfnav = Estimators(n=10,m=3)

    for k in range(0,L-1):

        # time uppdate
        if vx_can[k+1] <= 0.44704*vx_thresh:
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
            _, sysd_ = tract_trail_model.latModel(steer_ang=steer_can[k+1], Vx=float(x_[1]), dt=dt)
            A = sysd_.A
            B = sysd_.B

            # generate full navigation matrices
            A, B, H = genNavMatrices(A_veh=A, B_veh=B, vx=float(x_[1]), yaw=float(x_[5][0]), dt=dt)

        # model input
        u = np.array([steer_can[k+1]])

        # imu measurements
        z = np.array([[vx_can[k+1]],
                    [imu_accel_y[k+1]], #+ float(x_[8])
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
        bias_ay_nav[k+1] = x_[8].item()
        bias_ar_nav[k+1] = x_[9].item()

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
        'bias_ay_nav': bias_ay_nav,
        'bias_ar_nav': bias_ar_nav,
        }
    return filter_states