import numpy as np
import os
import sys
import optparse
import constants as cs

import math
import matplotlib.pyplot as plt

from constants import Veh_Parameter
from constants import IDM_Param
from derivation_fast import Derivation_class
# from SUMO_section import SUMO_TRACI_CLASS
import time


# load values of acceleration, spacing, speed from SUMO section
# sumo_class = SUMO_TRACI_CLASS()
# acc_ego, speed_ego, spacing_ego, relspd_ego = sumo_class.Traci()

# IDM variables definitions
it = IDM_Param().it   # number of iterations 
iteration = IDM_Param().iteration
dxx = IDM_Param().dxx
mu = IDM_Param().mu
buffer = IDM_Param().buffer
high_buffer = IDM_Param().high_buffer
lb = IDM_Param().lb  # a lower bound for a, b, T, s0 values
lb_speed = IDM_Param().lb_speed # lower band for v0
 
num_init = IDM_Param().num_init  # number of initial points
# initial values for parametrs + lb and ub for each
s0 = np.zeros((num_init, it))
s0[: , 0] = IDM_Param().s0

a = np.zeros((num_init, it))
a[: , 0] = IDM_Param().a
# print(a[1][0])

b = np.zeros((num_init, it))
b[: , 0] =  IDM_Param().b

T = np.zeros((num_init, it))
T[: , 0] = IDM_Param().T

# parameters for the warning message (d,c)
d = np.zeros((num_init, it))
d[: , 0] = IDM_Param().d

c = np.zeros((num_init, it))
c[: , 0] = IDM_Param().c

v0 = IDM_Param().v0[0]

delta = IDM_Param().delta
epsilon = IDM_Param().epsilon
small_num = IDM_Param().small_num

## FOr now, U_display == warning_msg

# spacing function, # compute spacing using IDM   , (d,c, u_war) is for the warning algorithm
def V_dot(s0, a, b, T, d, c, delta, spd, space, u_display): 
    # print("values are ", s0, a, b, T, d, c, delta, spd, space, u_display)
    sstar = s0 + max(0, spd * T + (spd * 1)/(2 * math.sqrt(a*abs(b))))

    # print("value of a is ", a, "value of v is ", v0 , " speed and accel are", acc , v )
    v_dot = a * ( 1 - (spd/v0)**delta - (sstar/space)**2) - d * u_display**c
    # print("v_dot is ", v_dot, " sstar is ", sstar)
    if math.isnan(v_dot):
        v_dot = 0
    return v_dot




# function: find derivative for a sum (all simulation steps) in J
df_class = Derivation_class()
large_number = df_class.large_num



def dGradient_da(s00, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg):
    dJ = 0
    for i in range(len(warning_msg)):
        # print( " war message is " , warning_msg[i], " derivative is ", df_class.a_derivative(s0_value = s00, a_value = a, b_value = b, T_value = T, delta = delta, speed = speed_ego[i], spacing= spacing_ego[i]))
        # print( " v dot is", V_dot(s0, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]))
        # print("each loop value", dJ, "S is", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]), "derivation is", df_class.a_derivative(s0_value = s00, a_value = a, b_value = b, T_value = T, v0_value = v0, delta = delta, speed = speed_ego[i], rel_speed = relspd_ego[i], accel = acc_ego[i]))
        dJ = dJ + (V_dot(s00, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]) - warning_msg[i]) * df_class.a_derivative(s0_value = s00, a_value = a, b_value = b, T_value = T, delta = delta, speed = speed_ego[i], spacing = spacing_ego[i])
        # print("heyy ", dJ)
        # print("hey2 ", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]))
                # Calculate the elapsed time
    dJ = 2 * dJ/ len(warning_msg)
    return dJ



def dGradient_dd(s00, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg):
    dJ = 0
    for i in range(len(warning_msg)):
        # print("heyy ", dJ)
        # print("V_dot ", V_dot(s00, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]) , " warning msg ", warning_msg[i])
        # print( "gradient is ", df_class.d_derivative(warning__msg= warning_msg[i], c_value = c))
        # print("each loop value", dJ, "S is", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]), "derivation is", df_class.a_derivative(s0_value = s00, a_value = a, b_value = b, T_value = T, v0_value = v0, delta = delta, speed = speed_ego[i], rel_speed = relspd_ego[i], accel = acc_ego[i]))
        dJ = dJ + (V_dot(s00, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]) - warning_msg[i]) * df_class.d_derivative(warning__msg= warning_msg[i], c_value = c)

    dJ = 2 * dJ/ len(warning_msg)
    # print("derivative d is ", dJ, " len is ", len(warning_msg))
    return dJ


def dGradient_dc(s00, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg):
    dJ = 0
    for i in range(len(warning_msg)):

        # print("each loop value", dJ, "S is", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]), "derivation is", df_class.a_derivative(s0_value = s00, a_value = a, b_value = b, T_value = T, v0_value = v0, delta = delta, speed = speed_ego[i], rel_speed = relspd_ego[i], accel = acc_ego[i]))
        dJ = dJ + (V_dot(s00, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]) - warning_msg[i]) * df_class.c_derivative( d_value = d, c_value = c, warning_msg = warning_msg[i])
        # print("heyy ", dJ)
        # print("hey2 ", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]))
                # Calculate the elapsed time
    dJ = 2 * dJ/ len(warning_msg)
    return dJ


def dGradient_ds0(s00, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg):
    dJ = 0
    for i in range(len(warning_msg)):
        # print("each loop value", dJ, "S is", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]), "derivation is", df_class.a_derivative(s0_value = s00, a_value = a, b_value = b, T_value = T, v0_value = v0, delta = delta, speed = speed_ego[i], rel_speed = relspd_ego[i], accel = acc_ego[i]))
        dJ = dJ + (V_dot(s00, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]) - warning_msg[i]) * df_class.s0_derivative(  s0_value = s00, a_value = a, b_value = b, d_value = d, T_value = T, speed = speed_ego[i], spacing= spacing_ego[i])
        # print("heyy ", dJ)
        # print("hey2 ", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]))
                # Calculate the elapsed time
    dJ = 2 * dJ/ len(warning_msg)
    return dJ
    

def dGradient_dT(s00, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg):
    dJ = 0
    for i in range(len(warning_msg)):

        # print("each loop value", dJ, "S is", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]), "derivation is", df_class.a_derivative(s0_value = s00, a_value = a, b_value = b, T_value = T, v0_value = v0, delta = delta, speed = speed_ego[i], rel_speed = relspd_ego[i], accel = acc_ego[i]))
        dJ = dJ + (V_dot(s00, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]) - warning_msg[i]) * df_class.T_derivative(  s0_value = s00, a_value = a, b_value = b, d_value = d, T_value = T, speed = speed_ego[i], spacing= spacing_ego[i])
        # print("heyy ", dJ)
        # print("hey2 ", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]))
                # Calculate the elapsed time
    dJ = 2 * dJ/ len(warning_msg)
    return dJ


def dGradient_db(s00, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg):
    dJ = 0
    for i in range(len(warning_msg)):
        # print("each loop value", dJ, "S is", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]), "derivation is", df_class.a_derivative(s0_value = s00, a_value = a, b_value = b, T_value = T, v0_value = v0, delta = delta, speed = speed_ego[i], rel_speed = relspd_ego[i], accel = acc_ego[i]))
        dJ = dJ + (V_dot(s00, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]) - warning_msg[i]) * df_class.b_derivative(  s0_value = s00, a_value = a, b_value = b, d_value = d, T_value = T, speed = speed_ego[i], spacing= spacing_ego[i])
        # print("heyy ", dJ)
        # print("hey2 ", S(s00, a, b, T, v0, delta, speed_ego[i], relspd_ego[i], acc_ego[i]))
                # Calculate the elapsed time
    dJ = 2 * dJ/ len(warning_msg)
    return dJ



# find the value of J
def J(s0, a, b, T, d, c, delta, spacing_ego, speed_ego, warning_msg):
    J = 0
    a = max(a, IDM_Param().lb_ac)
    b = max(abs(b), IDM_Param().lb_ac)
    # print("man", a, b)
    for i in range(len(warning_msg)):
        # function from eq 4
        J = J + ( warning_msg[i] - V_dot(s0, a, b, T, d, c, delta, speed_ego[i], spacing_ego[i], warning_msg[i]))**2 
    
    J = J/ len(warning_msg)
    return J


# print("ideal value is", J(2.5, 3.2, 3, 1, 35, delta, spacing_ego, speed_ego, relspd_ego, acc_ego ))
# print("max values ", max_a, max_v, min_a, min_s)
last_step = [0 for _ in range(num_init)]  # store value of last iteration for each initial solution

def mainIDM(spacing_ego, speed_ego, relspd_ego, warning_msg ,max_a, min_a, max_v, min_s):
    J_actual = 10000000
    J_prev = J_actual * 0.9
    iter_counter = 0
    
    # start with inital values for parameters
    a = IDM_Param().a[0]
    d = IDM_Param().d[0]
    c = IDM_Param().c[0]
    s0 = IDM_Param().s0[0]
    T = IDM_Param().T[0]
    b = IDM_Param().b[0]
            # print(" iteration is ", n)

            # ensure a range for a

    if a > high_buffer * max_a:
        a = high_buffer * max_a
    if a < min_a:
        a = min_a


    if b > high_buffer * max_a:
        b = high_buffer * max_a
    if b < min_a:
        b = min_a


    while abs(J_prev - J_actual)/ (J_prev + epsilon) > epsilon:
        iter_counter += 1
        # J for this iteration
        J_actual = J(s0, a, b, T, d, c, delta, spacing_ego, speed_ego, warning_msg)
        


        G_s0 = dGradient_ds0(s0, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg)
        G_a = dGradient_da(s0, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg)
        G_b = dGradient_db(s0, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg)
        G_T = dGradient_dT(s0, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg)
        G_d = dGradient_dd(s0, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg)
        G_c = dGradient_dc(s0, a, b, T, delta, spacing_ego, speed_ego, d, c, warning_msg)
        
        # print("as", a[sol][n], b[sol][n])
        # print("ds0 ", G_s0)
        # print("da ", G_a)
        # print("db ", G_b)
        # print("dT ", G_T)
        # print("dv0 ", G_v0)

        # # Adaptive step size 
        mu = 1
        J_hat = J_actual + 2
        # print("J Actual is " , J_actual, " J hat will be " , J_hat)
        while J_hat > J_actual:
            
            # adjust a and v0 and b, max high buffer derivation is allowed
            while abs(mu * G_a) > max_a * high_buffer or abs(mu * G_b) > abs(min_a) * high_buffer or abs(mu * G_s0) > min_s* high_buffer or abs(mu * G_T) > 4 or abs(mu * G_d) > 2 or abs(mu * G_c) > 2:
                mu = mu / 2

            # print("parameters", s0_fixer, max(epsilon, a_fixer), max(epsilon, b_fixer), max(epsilon, T[sol][n] + mu * G_T), v0_fixer)

            # update J_hat
            J_hat = J(s0 - mu * G_s0, a - mu * G_a,  b - mu * G_b, T - mu * G_T, d - mu* G_d, c - mu* G_c, delta, spacing_ego, speed_ego, warning_msg)
            if J_hat > J_actual:
                mu = mu / 2
            
            # print(mu, "muuuu")
            
            if mu < 0.000001:
                break
            
        

        # update values for the next time step
        s0 = max(s0 - mu * G_s0, min_s)
        a = max(min_a, a - mu * G_a)
        b = max(min_a, b - mu * G_b)
        T = T - mu * G_T
        d = d - mu * G_d
        c = c - mu * G_c

        # update J_actual
        # print(" loss values is ", J_hat, " and " , J_actual, " and d is ", d, " derivative is ", G_d, " c is ", c, " derivative is ", G_c)
        J_prev = J_actual  # save previous J value
        J_actual = J_hat  # update J value for this iteration

    # exit if gettin NaN values
    # if math.isnan(G_s0):
    #     sys.exit(1)

        # return s0[:, it-1], a[:, it-1], b[:, it-1], T[:, it-1], v0[:, it-1]


    # final_J = [0] * num_init
    # for i in range(num_init):
    #     # print(" s0 list is", s0[i, last_step[i]-1])
    #     # print(" a list is", a[i, last_step[i]-1])
    #     # print(" b list is", b[i, last_step[i]-1])
    #     # print(" T list is", T[i, last_step[i]-1])
    #     # print(" v0 list is", v0[i, last_step[i]-1])
    #     # print(" value of J is", J(s0[i][last_step[i]-1], a[i][last_step[i]-1], b[i][last_step[i]-1], T[i][last_step[i]-1], v0[i][last_step[i]-1], delta, spacing_ego, speed_ego, relspd_ego, acc_ego))
    #     # print("last iteration was", last_step[i])
    #     final_J[i] = J(s0[i][last_step[i]-1], a[i][last_step[i]-1], b[i][last_step[i]-1], T[i][last_step[i]-1], delta, spacing_ego, speed_ego, acc_ego)
    # min_J_index = final_J.index(min(final_J))
    # # print(" s0 list is", s0[min_J_index, last_step[min_J_index]-1])
    # # print(" a list is", a[min_J_index, last_step[min_J_index]-1])
    # # print(" b list is", b[min_J_index, last_step[min_J_index]-1])
    # # print(" T list is", T[min_J_index, last_step[min_J_index]-1])
    # # print(" v0 list is", v0[min_J_index, last_step[min_J_index]-1

    # print("final loss is ", J_actual, " lenght of data is" , len(spacing_ego))
    return a, d, c, s0 , T, b




# plot J value vs. iterations
# fig, axs = plt.subplots(3, 2)
# # Plot data on each subplot
# axs[0, 0].plot(iteration[:], J_actual[0][:])
# axs[0, 0].set_title('1')

# axs[0, 1].plot(iteration[:], J_actual[1][:])
# axs[0, 1].set_title('2')

# axs[1, 0].plot(iteration[:], J_actual[2][:])
# axs[1, 0].set_title('3')

# axs[1, 1].plot(iteration[:], J_actual[3][:])
# axs[1, 1].set_title('4')

# axs[2, 1].plot(iteration[:], J_actual[4][:])
# axs[2, 1].set_title('5')
# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Step 5: Show or save the plot
# plt.show()


