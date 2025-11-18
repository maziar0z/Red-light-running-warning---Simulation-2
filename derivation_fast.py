import sympy
from sympy import *
import math
from constants import IDM_Param
from constants import Veh_Parameter
import time

# this file obtains partial derivation of the s function in respect to each parameter. This file is named fast because the aritmatic is done before and the
# final terms are substituded. 

# strategy is if the value of "-speed**4/v0**4 + 1 - accel/a > 0", we set low values for b, s, T derivstive until a and v values are adjusted.
class Derivation_class:
    def __init__(self):
        self.small_num = 1   # samall number
        self.large_num = 1000000
        self.counter = 0
        self.sum = 0
        self.buffer2 = 1.5



        # function gets derivative in terms of a, parametrs are s0,b T, v0  , other argumets are just number
    def a_derivative(self, s0_value, a_value, b_value, T_value, delta, speed, spacing):

        # check value of a to be in the range of the lower and upper bound
        if abs(a_value) < IDM_Param().lb_ac:
            a_value = IDM_Param().lb_ac
        if abs(a_value) > IDM_Param().ub_ac:
            a_value = IDM_Param().ub_ac

        # find the value at any point x
        s0 = s0_value
        a = a_value
        b = b_value
        T = T_value

        sstar = s0 + max(0, speed * T + (speed * 1)/(2 * math.sqrt(a*abs(b))))

 
        v0 = Veh_Parameter().max_spd
        # print(" speed is ", speed, " v0 is ", v0, " a is ", a, " accel is ", accel)
        number = 1 - (speed / v0)**delta - (sstar/spacing)**2
        return min(number, IDM_Param().ub_ac)



    def d_derivative(self, warning__msg, c_value):
        c = round(c_value, 1)
        number = - abs(warning__msg) ** c
        # print(" D DERIVE " , number , " warning msg is ", warning__msg, " c value is ", c)
        return number


    def c_derivative(self, d_value, c_value, warning_msg):
        c = round(c_value, 1)
        d = d_value 
        if warning_msg == 0:
            number = 0
        else:
            number = - d * math.log(abs(warning_msg)) * abs(warning_msg) ** round(c, 1)
        return number
    


    def s0_derivative(self, s0_value, a_value, b_value, d_value, T_value, speed, spacing ):
        a = a_value
        v0 = Veh_Parameter().max_spd
        b = b_value
        s0 = s0_value
        d = d_value
        T = T_value

        sstar = s0 + max(0, speed * T + (speed * 1)/(2 * math.sqrt(a*abs(b))))

        number = - a * 2 * sstar / spacing**2

        return number
    


    def T_derivative(self, s0_value, a_value, b_value, d_value, T_value, speed, spacing ):
        a = a_value
        v0 = Veh_Parameter().max_spd
        b = b_value
        s0 = s0_value
        d = d_value
        T = T_value

        sstar = s0 + max(0, speed * T + (speed * 1)/(2 * math.sqrt(a*abs(b))))

        number = - a * 2 * sstar * speed / spacing**2

        return number
    

    def b_derivative(self, s0_value, a_value, b_value, d_value, T_value, speed, spacing ):
        a = a_value
        v0 = Veh_Parameter().max_spd
        b = b_value
        s0 = s0_value
        d = d_value
        T = T_value

        sstar = s0 + max(0, speed * T + (speed * 1)/(2 * math.sqrt(a*abs(b))))

        number = a * 2 * sstar * speed * speed * b**(-1.5)  / (spacing**2 * 4 * a**0.5)

        return number


    










        # get the run time average
        # # Record the start time
        # start_time = time.time() 
        # self.counter = self.counter + 1

        # # Calculate the elapsed time
        # elapsed_time = time.time() - start_time
        # self.sum = self.sum + elapsed_time

        # if self.counter % 100 == 0:
        #     print("a average Time taken to run the line of code: {:.6f} seconds".format(self.sum/self.counter))