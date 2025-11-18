import numpy as np


def vehicle_information():
    # CV id in the simulation, target vehicle is not included
    CV_list = ["flo1.2", "flo1.1"]
    # target vehicle name (speed that to be predicted)
    target_veh = "flo1.3"
    return CV_list, target_veh

class SimParameter:
    def __init__(self):
        # timestep in SUMO (sec)
        self.dt_sumo = 0.1
        # tiemstep for estimation (sec)
        self.dt_estimation = 0.1
        # length of cell (m)
        self.dx = 20
        # prediction horizon in time (sec)
        self.horizon_time = 10
        # prediction horizon
        self.horizon_step = int(self.horizon_time / self.dt_sumo)
        # communication range (m)
        self.range_V2V = 500
        self.range_V2I = 500
        # prediction horizon in distance (m)
        self.cell_range = 500
        # perception range in distance (m)
        self.perception_range = 100
        # number of cell
        self.num_cell = round((self.cell_range+self.dx)/self.dx)
        # maximum following distance (m), without considering t0*spd_following
        self.max_following_distance = 40
        # minimum following distance (m), without considering t0*spd_following
        self.min_following_distance = 5
        # desired time headway
        self.t_0 = 1.5        
        # maximum deceleration
        self.max_dec = -4.5
        # maximum acceleartion
        self.max_acc = 2.5
        # duration of lane change
        self.lc_duration = 5