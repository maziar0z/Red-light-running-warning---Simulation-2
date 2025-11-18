#!/usr/bin/env python
import os
import sys
import optparse
import pandas as pd
import matplotlib.pyplot as plt

############################################
############## Scenario 3 ##################
# no preceding vehicle, late active


# we need to import some python modules from the $SUMO_HOME/tools directory
import traci
from sumolib import checkBinary  # Checks for the binary in environ vars
import math
import numpy as np
from warning_alg import *
from UKF import *
from estimation_param import *
from est_pred_param import *
from calibrate_gradient_adaptive_step import *


# if "SUMO_HOME" in os.environ:
#     tools = os.path.join(os.environ["SUMO_HOME"], "tools")
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

sys.path.append(os.path.join("c:", os.sep, "whatever", "path", "to", "sumo", "tools"))


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option(
        "--nogui",
        action="store_true",
        default=False,
        help="run the commandline version of sumo",
    )
    options, args = opt_parser.parse_args()
    return options

def get_car_info(
    car, cumulated_d_l0, cumulated_d_l1, list_x_0, list_x_1, list_y_0, list_y_1
):
    x_car = traci.vehicle.getPosition(car)[0]
    y_car = traci.vehicle.getPosition(car)[1]
    acc_car = traci.vehicle.getAcceleration(car)
    lane_car = traci.vehicle.getLaneIndex(car)
    spd_car = traci.vehicle.getSpeed(car)
    if lane_car == 0:
        closest_index_targettoref = np.argmin(
            np.sqrt(
                (np.array(list_x_0) - x_car) ** 2 + (np.array(list_y_0) - y_car) ** 2
            )
        )
        distance_to_signal = cumulated_d_l0[closest_index_targettoref]
    elif lane_car == 1:
        closest_index_targettoref = np.argmin(
            np.sqrt(
                (np.array(list_x_1) - x_car) ** 2 + (np.array(list_y_1) - y_car) ** 2
            )
        )
        distance_to_signal = cumulated_d_l1[closest_index_targettoref]
    return spd_car, distance_to_signal, lane_car, acc_car

# main entry point
if __name__ == "__main__":
    sys.path.append(os.path.join("c:", os.sep, "whatever", "path", "to", "sumo", "tools"))

    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary("sumo")
    else:
        sumoBinary = checkBinary("sumo-gui")
    # traci starts sumo as a subprocess and then this script connects and runs
    sumoCmd = [
        sumoBinary,
        "-c",
        "config.3.sumocfg",
        "--quit-on-end",
        "true",
        "--tripinfo-output",
        "tripinfo.xml",
    ]
file_l0 = "ref_data/GFG0.csv"
file_l1 = "ref_data/GFG1.csv"

data_l0 = pd.read_csv(file_l0)
data_l1 = pd.read_csv(file_l1)

list_x_0 = data_l0["x0"].tolist()
list_y_0 = data_l0["y0"].tolist()

list_x_1 = data_l1["x1"].tolist()
list_y_1 = data_l1["x1"].tolist()

list_d_0 = data_l0["traveled0"].tolist()

list_d_1 = data_l1["traveled1"].tolist()

pos_signal_l0 = (
    1326.46,
    378.20,
)  # have this from RSU connection map message , for lane 0 center lane
pos_signal_l1 = (1324.10, 380.37)  # and for lane 1

tl_phase_duration_list = [17,8,10,5,20,5,15,5]

listcounter = 0
tot_spacing = 0


# find closest element in reference list to the true signal location
signal_id_0 = np.argmin(
    np.sqrt(
        (np.array(list_x_0) - pos_signal_l0[0]) ** 2
        + (np.array(list_y_0) - pos_signal_l0[1]) ** 2
    )
)
signal_id_1 = np.argmin(
    np.sqrt(
        (np.array(list_x_1) - pos_signal_l1[0]) ** 2
        + (np.array(list_y_1) - pos_signal_l1[1]) ** 2
    )
)

cumulated_d_l0 = [0] * (len(list_d_0)+1)

cumulated_d_l1 = [0] * (len(list_d_1)+1)

for i in range(signal_id_0, 0, -1):
    cumulated_d_l0[i - 1] = cumulated_d_l0[i] + list_d_0[i]
for i in range(signal_id_0, len(list_d_0), 1 ):
    cumulated_d_l0[i+1] =  cumulated_d_l0[i] - list_d_0[i]
for i in range(signal_id_1, 0, -1):
    cumulated_d_l1[i - 1] = cumulated_d_l1[i] + list_d_1[i]
for i in range(signal_id_1, len(list_d_1), 1 ):
    cumulated_d_l1[i+1] =  cumulated_d_l1[i] - list_d_1[i]

max_step = 1350
min_step = 400



# IDM max and min values for parameters
max_a = IDM_Param().ub_ac
min_a = IDM_Param().lb_ac
max_v = IDM_Param().ub_spd
min_s = Veh_Parameter().min_gap


a = IDM_Param().a[0]
d = IDM_Param().d[0]
c = IDM_Param().c[0]
s0 = IDM_Param().s0[0]
T = IDM_Param().T[0]
b = IDM_Param().b[0]


model_speed_list = []
model_accel_list = []

sumo_speed_list=[]
sumo_accel_list = []
location_list = []
sumo_spacing_list = []
warning_val_list = []
v_dot_list = []

step_list = []
tl_state_list = []
warning_value_list = []
# index of the desired traffic state in the traffic state list
id_tl = 0
acc_ego = 0   # used for acc found by MPC

CV_list = []
preced_veh = None
ego_veh = "car_1"
max_gap = Veh_Parameter().max_gap

RMSEE = 0
RMSE_Counter = 0

traci.start(sumoCmd)
num_horizon_pred = 100
dt = 0.1
step = 0
acc_ego_old = 0
while step < max_step:
    traci.simulationStep()
    if step >= min_step:
        vehicle_list = traci.vehicle.getIDList()
        if step >= min_step and ego_veh in vehicle_list:
            current_location = traci.vehicle.getPosition(ego_veh)
            current_loc = math.sqrt(current_location[0]**2 + current_location[1]**2)
            # pos ego: spacing between the ego vehcile and traffic light
            # pos CV and pos preced: spacing between the corresponding vehicle and traffic light

            pos_CV = []
            spd_CV = []
            lane_CV = []
            pos_ego = None
            spd_ego = None
            lane_ego = None
            pos_preced = None
            spd_preced = None
            lane_preced = None

            spd_ego, pos_ego, lane_ego, acc_ego_old = get_car_info(
                ego_veh,
                cumulated_d_l0,
                cumulated_d_l1,
                list_x_0,
                list_x_1,
                list_y_0,
                list_y_1,
            )


            

            # print( " veh dynamics are: spd ", spd_ego, " acc ", acc_ego_old)

            # " pos ", pos_ego, " lane ", lane_ego, " acc ", acc_ego_old )
            # speed_list.append(spd_ego)
            # print('pos ego', pos_ego)
            # for car in CV_list:
            #     spd_car, pos_car, lane_car, _ = get_car_info(
            #         car,
            #         cumulated_d_l0,
            #         cumulated_d_l1,
            #         list_x_0,
            #         list_x_1,
            #         list_y_0,
            #         list_y_1,
            #     )
            #     pos_CV.append(pos_car)
            #     spd_CV.append(spd_car)
            #     lane_CV.append(lane_car)

            # if preced_veh is not None:
            #     spd_preced, pos_preced, lane_preced, _ = get_car_info(
            #         preced_veh,
            #         cumulated_d_l0,
            #         cumulated_d_l1,
            #         list_x_0,
            #         list_x_1,
            #         list_y_0,
            #         list_y_1,
            #     )
            if step == min_step:
                # load simulation parameter
                sim_param = SimParameter()
                # load the paramter for UKF
                ukf_param = UKFParam(
                    sim_param.num_cell,
                    dt=sim_param.dt_sumo,
                    dx=sim_param.dx,
                    num_param=3,
                    dt_simulation=sim_param.dt_estimation,
                )
                filter_ukf = UnscentedKalmanFilter(
                    sim_param.dt_sumo, ukf_param, traffic_dynamics, measurement_fb
                )
                filter_ukf.initialize(lane_ego, spd_ego)

            tl_state = traci.trafficlight.getRedYellowGreenState("tl_1")[id_tl]
            tl_phase = traci.trafficlight.getPhase("tl_1")
            tl_next_switch = (
                traci.trafficlight.getNextSwitch("tl_1") - traci.simulation.getTime()
            )
            if tl_phase == 4:
                tl_duration = tl_next_switch
            elif tl_phase==3:
                tl_duration = tl_next_switch
            elif tl_phase<3:
                tl_duration = sum(tl_phase_duration_list[tl_phase+1:4]) + tl_next_switch
            elif tl_phase==5:
                tl_duration = tl_next_switch
            elif tl_phase==6:
                tl_duration = tl_next_switch+tl_phase_duration_list[7]+sum(tl_phase_duration_list[0:4])
            elif tl_phase==7:
                tl_duration = tl_next_switch+sum(tl_phase_duration_list[0:4])

            
            # tl_duration = (
            #     traci.trafficlight.getNextSwitch("tl_1") - traci.simulation.getTime()
            # )
            
            # print("tl state is", tl_state, "remained time is", tl_duration)
            # print("program is", traci.trafficlight.getCompleteRedYellowGreenDefinition("tl_1"))
            # current_phase = traci.trafficlight.getPhase("tl_1")


            # 0 indicates red, 1 indicates green, -1 indicates yellow

            pred_tl_state = np.zeros((num_horizon_pred + 1, 1))
            if tl_state == "r" or tl_state == "R":
                tl_state_list.append(0)
                # all red during prediction horizon
                if tl_duration >= (num_horizon_pred*dt):
                    pred_tl_state[:,0] = 0
                # change from red to green
                else:
                    pred_tl_state[0: int(tl_duration/dt),0] = 0
                    pred_tl_state[int(tl_duration/dt):,0] = 1

            if tl_state == "g" or tl_state == "G":
                # print(" spacing is " , pos_ego)
                pos_ego = max(max_gap,pos_ego)
                tl_state_list.append(1)
                # all green during prediction horizon
                if tl_duration >= (num_horizon_pred*dt):
                    pred_tl_state[:,0] = 1
                # green and yellow during prediction horizon
                elif tl_duration>=5:
                    pred_tl_state[0:int(tl_duration/dt),0] = 1
                    pred_tl_state[int(tl_duration/dt):,0] = -1
                # green, yellow and red during prediction horizon
                else:
                    pred_tl_state[0:int(tl_duration/dt),0] = 1
                    pred_tl_state[int(tl_duration/dt):int(tl_duration/dt)+50,0] = -1
                    pred_tl_state[int(tl_duration/dt)+50:,0] = 0
                            
            if tl_state == "y" or tl_state == "Y":
                # print(" spacing is " , pos_ego)
                pos_ego =  max(max_gap,pos_ego)
                tl_state_list.append(-1)
                # yellow and red during prediction horizon
                pred_tl_state[0:int(tl_duration/dt),0] = -1
                pred_tl_state[int(tl_duration/dt):,0] = 0



            ## IDM part
            if step % 2 == 0: # save every 0.2 second
                sumo_speed_list.append(spd_ego)  # current speed
                sumo_accel_list.append(acc_ego_old)  # actual acc of previous step
                warning_val_list.append(acc_ego)
                sumo_spacing_list.append(pos_ego)


                        ### CALIBRATION PART
            if step % 40 == 0 and step >= 550: # calibrate every 4 seconds
                a, d, c, s0, T, b = mainIDM(sumo_spacing_list, sumo_speed_list, sumo_speed_list, warning_val_list ,max_a, min_a, max_v, min_s)
                print("calibrated IDM parameters are: a ", a, " d ", d, " c ", c, " s0 ", s0, " T ", T, " b ", b)


        
        ## find RMSE between optimal acc and IDM prediction
            if step % 2 == 0 and step>= 550:
                sstar = s0 + max(0, spd_ego * T + (spd_ego * 1)/(2 * math.sqrt(a*abs(b) + 0.1)))

                # print("value of a is ", a, "value of v is ", v0 , " speed and accel are", acc , v )
                v_dot = a * ( 1 - (spd_ego/v0)**delta - (sstar/pos_ego)**2) - d * abs(acc_ego)**round(c , 1)
                # print( " values are spd, " , spd_ego , " a is " , a, " d is ", d, " c is ", round(c , 1), " s0 is ", s0, " T is ", T, " b is ", b, " sstar is " , sstar , " v dot is ", v_dot)

                if v_dot < IDM_Param().ub_ac*-1 * 1.1:
                    v_dot = IDM_Param().ub_ac * -1 * 1.1
                # print("v_dot is ", v_dot, " sstar is ", sstar)
                if math.isnan(v_dot):
                    v_dot = 0
                v_dot_list.append(v_dot)
                # print( " values are spd, " , spd_ego , " a is " , a, " d is ", d, " c is ", c, " s0 is ", s0, " T is ", T, " b is ", b , " sstar  is ", sstar, " pos_ego is ", pos_ego, " v_dot is ", v_dot)
                print(" IDM guess is ", v_dot, " MPC acc is ", acc_ego)

                # Compute the RMSE of the last 5 v_dot values
                RMSEE += (v_dot - acc_ego) ** 2
                RMSE_Counter += 1
                print("RMSE of the last " , RMSE_Counter , " steps is : ", math.sqrt(RMSEE) / RMSE_Counter)



                # tl_state_list.append(-1)
                # # yellow and red during prediction horizon
                # pred_tl_state[0:int(tl_duration/dt),0] = -1
                # pred_tl_state[int(tl_duration/dt):,0] = 0
            # every 10 steps (1 sec), do the MPC optimization
            if (step - min_step) % 10 == 0:
                warning = True
                acc_ego_list = warning_alg(
                    pos_CV,
                    spd_CV,
                    lane_CV,
                    pos_preced,
                    spd_preced,
                    lane_preced,
                    pos_ego,
                    spd_ego,
                    lane_ego,
                    acc_ego_old,
                    pred_tl_state,
                    filter_ukf,
                    sim_param,
                    ukf_param,
                    warning,
                )
                acc_ego = acc_ego_list[0]
                # otherwise, only do the traffic prediction
                # print("javab", acc_ego_list)
            else:
                warning = False
                _ = warning_alg(
                    pos_CV,
                    spd_CV,
                    lane_CV,
                    pos_preced,
                    spd_preced,
                    lane_preced,
                    pos_ego,
                    spd_ego,
                    lane_ego,
                    acc_ego_old,
                    pred_tl_state,
                    filter_ukf,
                    sim_param,
                    ukf_param,
                    warning,
                )
                tmp = (step - min_step) % 10
                if tmp % 2 == 0:
                    id0 = int(tmp / 2)
                    id1 = int(tmp / 2)
                else:
                    id0 = int(tmp / 2)
                    id1 = int(tmp / 2) + 1
                acc_ego = (
                    ((acc_ego_list[id0] + acc_ego_list[id1]) / 2)
                )
            warning_value_list.append(deepcopy(acc_ego))

            # speed1 = traci.vehicle.getSpeed(ego_veh)
            # speed1 = speed1 + acc_ego* dt
            # traci.vehicle.setSpeedMode(ego_veh, 32)
            # traci.vehicle.setSpeed(ego_veh, speed1)
            # if step<800:
            #     acc_ego = 0 




    step = 1 + step
    # print("step:", step)

traci.close()
sys.stdout.flush()



# use recorded data to calibrate IDM parameters
# from data_based_IDMcalibration import *
