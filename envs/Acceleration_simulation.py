import numpy as np
import sys


class AccelerationSimulator:
    def __init__(self, weather_info=1.0, max_speed=2.0):
        self.old_speed = np.zeros(2,)  # speed values for last steps, in m/s
        self.computed_new_speed = np.zeros(2,)  # speed values for current steps, in m/s
        self.accelerate_value = np.zeros(2,)  # accelerate values for current step, in m/s^2
        self.goal_dis = np.zeros(2,)  # distance that agent needs to move, in meters
        self.weight_of_agent = 20.0  # the weight for agent, in kg, shouldn't smaller than or equal to 0.0
        self.gravity_coefficient = 9.80665  # gravitational constant, in m/s^2, shouldn't be easily changed
        self.coefficient_of_friction = 0.0  # coefficient of friction, changes with weather
        self.weather_info = weather_info  # weather, 0.0 present sunny, default weather is sunny
        self.gain = 1.0  # gain factor to adjust the friction, positive, maximum 1.0
        # self.none_zero = np.array([1e-32, 1e-32])
        # to prevent the speed change to be zero, which makes the incorrect force disintegration
        self.max_speed = max_speed  # maximum speed that agent is allowed to go
        '''''''''
        list of weather info
        0.0 presents sunny
        1.0 presents rain
        2.0 presents snow without freezing land
        3.0 presents snow with freezing land 
        '''''''''
    def _get_coefficient_of_friction(self):
        if self.weather_info == 0.0:  # sunny
            self.coefficient_of_friction = 0.6 * self.gain
        elif self.weather_info == 1.0:  # rain
            self.coefficient_of_friction = 0.4 * self.gain
        elif self.weather_info == 2.0:  # snow without freezing land
            self.coefficient_of_friction = 0.28 * self.gain
        elif self.weather_info == 3.0:  # snow with freezing land
            self.coefficient_of_friction = 0.18 * self.gain
        else:
            print("the weather information is not correct, please checkout your weather setting")
            sys.exit()

    def compute_actual_movement_and_speed(self, action_time, force, last_step, reset):
        if reset:
            self.old_speed = np.zeros(2,)  # while env has reset, the history speed should be back to zero
            self.accelerate_value = np.zeros(2,)  # while env has reset, the history accelerate value
            # should be back to zero
        self._get_coefficient_of_friction()  # get weather factor that effects the friction of agent
        #  force and speed direction cases, to see if the friction help or prevent the speed change.
        friction = self.weight_of_agent * self.gravity_coefficient * self.coefficient_of_friction
        if np.linalg.norm(force) < friction and np.linalg.norm(self.old_speed) == 0.0:
            friction = np.linalg.norm(force)
        force_true = np.zeros(2,)
        #  cases how friction helps to stop
        for i in range(2):
            if self.old_speed[i] >= 0.0 and force[i] >= 0.0:
                force_true[i] = force[i] - (force[i]/np.linalg.norm(force)) * friction
            elif self.old_speed[i] >= 0.0 > force[i]:
                force_true[i] = force[i] + (force[i]/np.linalg.norm(force)) * friction
            elif self.old_speed[i] < 0.0 <= force[i]:
                force_true[i] = force[i] + (force[i]/np.linalg.norm(force)) * friction
            else:
                force_true[i] = force[i] - (force[i]/np.linalg.norm(force)) * friction
        self.accelerate_value = force_true / self.weight_of_agent
        new_speed = self.old_speed + self.accelerate_value * action_time
        if new_speed[0] > self.max_speed:
            new_speed[0] = self.max_speed
        if new_speed[1] > self.max_speed:
            new_speed[1] = self.max_speed
        true_move = 0.5 * (new_speed + self.old_speed) * action_time
        if last_step:
            f_last = -1.0 * friction
            a_last = f_last / self.weight_of_agent
            t_slip = np.linalg.norm(new_speed / a_last)
            d_slip = 0.5 * new_speed * t_slip
            true_move = true_move + d_slip
        self.old_speed = new_speed
        return true_move, new_speed
