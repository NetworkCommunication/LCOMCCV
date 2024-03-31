from __future__ import absolute_import
from __future__ import print_function

import csv
import logging
import math
import time

import gym
import numpy as np
import pandas as pd
from gym import spaces
import random as rn
import os
import sys
import traci
import traci.constants as tc

from mpmath import norm
from scipy import stats
from scipy.stats import norm
import torch.nn.functional as F

# we need to import python modules from the $SUMO_HOME/tools directory
from driverStyleCluster import Point
from driverStyleCluster import driverStyleCluster

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

gui = False
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')

config_path = "data/Lane3/StraightRoad.sumocfg"

class LaneChangePredict(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):

        self.minVelocity = 0
        self.maxVelocity = 30

        self.minDistanceFrontVeh = 0
        self.maxDistanceFrontVeh = 150

        self.minDistanceRearVeh = 0
        self.maxDistanceRearVeh = 150

        self.maxRoadLength = 3000

        self.minLaneNumber = 0
        self.maxLaneNumber = 2

        self.CommRange = 150

        self.delta_t = 0.1
        self.AutoCarIDAll = ['Car24', 'Car5','Car8']  # CVs
        self.PrevSpeed = 0
        self.PrevVehDistance = 0
        self.VehicleIds = 0
        self.traFlowNumber = 0
        self.finaTCC = 0
        self.opi = [0, 0, 0]

        self.velbefore = 0

        self.punish = 0

        self.laneFlag = [0, 0, 0]
        self.lanChangeFlag = [0, 0, 0]
        self.triggerFlag = [0, 0, 0]

        self.csvfile = 'data_train5.csv'

        self.overpassFlag = [0, 0, 0]
        self.AutoCarFrontID = ["", "", ""]
        self.tempAutoCarFrontID = ''
        self.ttc_safe = 3

        self.dFront = [0, 0, 0]
        self.vFront = [0, 0, 0]

        self.action_space_vehicle =[-1, 0, 1]
        self.n_actions = len(self.action_space_vehicle)
        self.n_actions = int(self.n_actions)
        self.param_velocity = [0, 30]
        self.n_features = 16

        self.actions = np.array([[0, -1], [1, 0], [2, 1]])



    def reset(self):
        self.TotalReward = 0
        self.numberOfLaneChanges = 0
        self.numberOfOvertakes = 0
        self.currentTrackingVehId = 'None'
        self.overpassFlag = [0, 0, 0]
        self.laneFlag = [0, 0, 0]
        self.countOPI = [0, 0, 0]

        data = pd.read_csv('modified_result2.csv', header=None)
        self.myData = data.iloc[:, 4]

        z_scores = stats.zscore(self.myData)

        threshold = 3

        filtered_data = self.myData[(np.abs(z_scores) < threshold)]

        self.filteredAbsMin = abs(min(filtered_data))

        transformed_data = np.log(filtered_data - self.filteredAbsMin)

        masked_A = transformed_data[~np.isinf(transformed_data)]
        mean = np.mean(masked_A)
        std_dev = np.std(masked_A)

        standard_error = std_dev / (len(masked_A) ** 0.5)

        self.confidence_interval = stats.norm.interval(0.95, loc=mean, scale=standard_error)

        self.myData2 = data.iloc[:, 5]

        z_scores2 = stats.zscore(self.myData2)

        filtered_data2 = self.myData2[(np.abs(z_scores2) < threshold)]

        self.filteredAbsMin2 = abs(min(filtered_data2))

        transformed_data2 = np.log(filtered_data + self.filteredAbsMin2)

        masked_A2 = transformed_data2[~np.isinf(transformed_data2)]
        mean2 = np.mean(masked_A2)
        std_dev2 = np.std(masked_A2)

        standard_error2 = std_dev2 / (len(masked_A2) ** 0.5)

        self.confidence_interval2 = stats.norm.interval(0.95, loc=mean2, scale=standard_error2)

        traci.close()

        sumo_binary = "sumo"
        sumocfg_file = "data/Lane3/StraightRoad.sumocfg"

        sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--delay", "1", "--scale", "1"]
        traci.start(sumo_cmd)

        print('Resetting the layout')
        traci.simulationStep()

        if os.path.exists(self.csvfile):
            os.remove(self.csvfile)

        self.VehicleIds = traci.vehicle.getIDList()

        for veh_id in self.VehicleIds:
            traci.vehicle.subscribe(veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION])

    def find_action(self, index):
        return self.actions[index][1]

    def step(self, action, action_param, i):
        x = action
        v_n = (np.tanh(action_param) + 1) * 15
        desired_speed = float(v_n.item())
        Vehicle_Params = traci.vehicle.getAllSubscriptionResults()

        self.punish = 0

        self.PrevSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
        self.PrevVehDistance = Vehicle_Params[self.AutoCarID][tc.VAR_LANEPOSITION]

        traci.vehicle.setSpeed(self.AutoCarID, desired_speed)

        vehicles = traci.vehicle.getIDList()

        for vehicle_id in vehicles:
            time = traci.simulation.getTime()
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            vehicle_accler = traci.vehicle.getAcceleration(vehicle_id)

            with open(self.csvfile, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([time, vehicle_id, vehicle_speed, vehicle_accler])

        time = traci.simulation.getTime()
        self.lanChangeFlag[i] = 0
        if x == 1:
            if self.laneFlag[i] == 0:
                self.AutoCarFrontID[i] = self.tempAutoCarFrontID
            self.laneFlag[i] = self.laneFlag[i] + 1
            self.lanChangeFlag[i] = 1
            self.selectClusterData(time, i)
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            self.velbefore = self.state[12]
            if laneindex != 0:
                traci.vehicle.changeLane(self.AutoCarID, laneindex - 1, 100)
                self.numberOfLaneChanges += 1
                self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex - 1]
                if self.state[3] == -1:
                    self.punish = self.punish - 1
            else:
                self.punish = self.punish - 1
        elif x == -1:
            if self.laneFlag[i] == 0:
                self.AutoCarFrontID[i] = self.tempAutoCarFrontID

            self.laneFlag[i] = self.laneFlag[i] + 1

            self.lanChangeFlag[i] = 1
            self.selectClusterData(time, i)

            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            self.velbefore = self.state[8]
            if laneindex != self.maxLaneNumber:
                traci.vehicle.changeLane(self.AutoCarID, laneindex + 1, 100)
                self.numberOfLaneChanges += 1
                self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex + 1]
                if self.state[3] == -1:
                    self.punish = self.punish - 1
            else:
                self.punish = self.punish - 1
        else:
            self.selectClusterData(time, i)
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex]

        traci.simulationStep()

        self.state = self._findstate(i)

        self.end = self.is_overtake_complete(self.state, i)

        reward = self.updateReward(action, self.state, i)

        return self.state, reward, self.end

    def selectClusterData(self, time, i):
        vehicle_ids = self.id_list
        laneFlag = self.laneFlag[i]
        self.historyData = pd.DataFrame()
        csvFirst = 'data_trainFirst.csv'
        dataFirst = pd.read_csv(csvFirst, header=None)
        data = pd.read_csv(self.csvfile, header=None)
        self.selected_data = {}
        if time < 6:
            if laneFlag == 1:
                flat_car_ids = [id[0] for id in vehicle_ids if id]

                for car_id in flat_car_ids:
                    if time > 1:
                        self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - 1) & (data[0] <= time)]
                    else:
                        self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - len(data)) & (data[0] <= time)]
                    selected_data_part = dataFirst[(dataFirst[1] == car_id) & (dataFirst[0] >= 0) & (dataFirst[0] <= 6)].iloc[:,
                                             2:4]
                    self.historyData = self.historyData.append(selected_data_part, ignore_index=True)
            elif laneFlag > 1:
                flat_car_ids = [id[0] for id in vehicle_ids if id]

                for car_id in flat_car_ids:
                    if time > 1:
                        self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - 1) & (data[0] <= time)]
                    else:
                        self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - len(data)) & (data[0] <= time)]
        else:
            if laneFlag == 1:
                flat_car_ids = [id[0] for id in vehicle_ids if id]

                for car_id in flat_car_ids:
                    self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - 1) & (data[0] <= time)]
                    if self.historyData.empty:
                        self.historyData = data[(data[1] == car_id) & (data[0] >= time - 6) & (data[0] <= time - 1)].iloc[:, 2:4]
                    else:
                        selected_data_part = data[(data[1] == car_id) & (data[0] >= time - 6) & (data[0] <= time - 1)].iloc[:,
                                             2:4]
                        self.historyData = self.historyData.append(selected_data_part, ignore_index=True)
            else:
                flat_car_ids = [id[0] for id in vehicle_ids if id]

                for car_id in flat_car_ids:
                    self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - 1) & (data[0] <= time)]

    def calculateAverage(self):
        data = self.selected_data
        mean_values = {}

        for key, value in data.items():
            col_4 = value[2]
            col_5 = value[3]

            avg_col_4 = col_4.mean()
            avg_col_5 = col_5.mean()

            mean_values[key] = {'averageVel': avg_col_4, 'averageAcc': avg_col_5}

        return mean_values

    def close(self):
        traci.close()

    def _findRearVehDistance(self, vehicleparameters):
        parameters = [[0 for x in range(5)] for x in range(len(vehicleparameters))]
        i = 0
        d1 = -1
        d2 = -1
        d3 = -1
        d4 = -1
        d5 = -1
        d6 = -1
        v1 = -1
        v2 = -1
        v3 = -1
        v4 = -1
        v5 = -1
        v6 = -1

        self.id_list = [[] for _ in range(6)]

        for VehID in self.VehicleIds:
            parameters[i][0] = VehID
            parameters[i][1] = vehicleparameters[VehID][tc.VAR_LANEPOSITION]
            parameters[i][2] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]
            parameters[i][3] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]
            parameters[i][4] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]
            i = i + 1

        parameters = sorted(parameters, key=lambda x: x[1])
        index = [x for x in parameters if self.AutoCarID in x][0]
        RowIDAuto = parameters.index(index)

        if RowIDAuto == len(self.VehicleIds) - 1:
            d1 = -1
            v1 = -1
            d3 = -1
            v3 = -1
            d5 = -1
            v5 = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = 'None'
        else:
            if parameters[RowIDAuto][2] == 0:
                d5 = -1
                v5 = -1
                d6 = -1
                v6 = -1
            elif parameters[RowIDAuto][2] == (self.maxLaneNumber - 1):
                d3 = -1
                v3 = -1
                d4 = -1
                v4 = -1
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d1 = parameters[index][1] - parameters[RowIDAuto][1]
                    v1 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    veh_id_d1 = parameters[index][0]
                    self.tempAutoCarFrontID = veh_id_d1
                    self.id_list[0].append(veh_id_d1)
                    break
                index += 1
            if index == len(self.VehicleIds):
                d1 = -1
                v1 = -1
                self.CurrFrontVehID = 'None'
                self.CurrFrontVehDistance = 150
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d3 = parameters[index][1] - parameters[RowIDAuto][1]
                    v3 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    veh_id_d3 = parameters[index][0]
                    self.id_list[2].append(veh_id_d3)
                    break
                index += 1
            if index == len(self.VehicleIds):
                d3 = -1
                v3 = -1
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d5 = parameters[index][1] - parameters[RowIDAuto][1]
                    v5 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    veh_id_d5 = parameters[index][0]
                    self.id_list[4].append(veh_id_d5)
                    break
                index += 1
            if index == len(self.VehicleIds):
                d5 = -1
                v5 = -1
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d2 = parameters[RowIDAuto][1] - parameters[index][1]
                    v2 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    veh_id_d2 = parameters[index][0]
                    self.id_list[1].append(veh_id_d2)
                    break
                index -= 1
            if index < 0:
                d2 = -1
                v2 = -1
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d4 = parameters[RowIDAuto][1] - parameters[index][1]
                    v4 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    veh_id_d4 = parameters[index][0]
                    self.id_list[3].append(veh_id_d4)
                    break
                index -= 1
            if index < 0:
                d4 = -1
                v4 = -1
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d6 = parameters[RowIDAuto][1] - parameters[index][1]
                    v6 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    veh_id_d6 = parameters[index][0]
                    self.id_list[5].append(veh_id_d6)
                    break
                index -= 1
            if index < 0:
                d6 = -1
                v6 = -1
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = parameters[RowIDAuto + 1][0]
        if RowIDAuto == 0:
            RearDist = -1
        else:
            RearDist = (parameters[RowIDAuto][1] - parameters[RowIDAuto - 1][
                1])
        if RowIDAuto == len(self.VehicleIds) - 1:
            FrontDist = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
        else:
            FrontDist = (parameters[RowIDAuto + 1][1] - parameters[RowIDAuto][
                1])
            self.CurrFrontVehID = parameters[RowIDAuto + 1][0]
            self.CurrFrontVehDistance = FrontDist
        return d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6

    def _findCurrentState(self,i):
        self.AutoCarID = self.AutoCarIDAll[i]
        self.state = self._findstate(i)

    def _findCurrentOtherState(self, j, i):
        self.AutoCarID = self.AutoCarIDAll[j]
        state = self._findstate(j)
        self.AutoCarID = self.AutoCarIDAll[i]
        return state

    def _findstate(self, i):
        self.AutoCarID = self.AutoCarIDAll[i]
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6 = self._findRearVehDistance(VehicleParameters)
        if ((d1 > self.CommRange)):
            d1 = self.maxDistanceFrontVeh
            v1 = -1
        elif d1 < 0:
            d1 = self.maxDistanceFrontVeh
        if ((v1 < 0) and (d1 <= self.CommRange)):
            v1 = 0

        if ((d2 > self.CommRange)):
            d2 = self.maxDistanceRearVeh
            v2 = -1
        elif d2 < 0:
            d2 = 0
        if ((v2 < 0) and (d2 <= self.CommRange)):
            v2 = 0
        if ((d3 > self.CommRange)):
            d3 = self.maxDistanceFrontVeh
            v3 = -1
        elif d3 < 0:
            d3 = self.maxDistanceFrontVeh
        if ((v3 < 0) and (d3 <= self.CommRange)) :
            v3 = 0

        if ((d4 > self.CommRange)):
            d4 = self.maxDistanceRearVeh
            v4 = -1
        elif d4 < 0:
            d4 = self.maxDistanceRearVeh
        if ((v4 < 0) and (d4 <= self.CommRange)) :
            v4 = 0

        if ((d5 > self.CommRange)):
            d5 = self.maxDistanceFrontVeh
            v5 = -1
        elif d5 < 0:
            d5 = self.maxDistanceFrontVeh
        if ((v5 < 0) and (d5 <= self.CommRange)) :
            v5 = 0

        if ((d6 > self.CommRange)):
            d6 = self.maxDistanceRearVeh
            v6 = -1
        elif d6 < 0:
            d6 = self.maxDistanceRearVeh
        if ((v6 < 0) and (d6 <= self.CommRange)):
            v6 = 0

        va = VehicleParameters[self.AutoCarID][tc.VAR_SPEED]
        da = VehicleParameters[self.AutoCarID][tc.VAR_LANEPOSITION]
        if self.laneFlag[i] != 0:
            id = self.AutoCarFrontID[i]
            self.dFront[i] = VehicleParameters[id][tc.VAR_LANEPOSITION]
            self.vFront[i] = VehicleParameters[id][tc.VAR_SPEED]
        vacc = (va - self.PrevSpeed)/self.delta_t
        return va, da, v1, d1, v2, d2, v3, d3, v4, d4, v5, d5, v6, d6, VehicleParameters[self.AutoCarID][tc.VAR_LANE_INDEX], vacc

    def getCVInformation(self):
        distance = {}
        lane_id = {}
        vehicle_speed = {}
        for index, vehicle_id in enumerate(self.AutoCarIDAll):
            distance[index] = traci.vehicle.getLanePosition(vehicle_id)
            lane_id[index] = traci.vehicle.getLaneID(vehicle_id).split('_')[-1]
            vehicle_speed[index] = traci.vehicle.getSpeed(vehicle_id)

        return distance, lane_id, vehicle_speed


    def is_overtake_complete(self, state, i):
        if state[1] >= 400:
            self.overpassFlag[i] = 1
        return self.overpassFlag[i]

    def is_laneOvertakeComp(self, state, i):
        delta_v = abs(state[0] - self.vFront)
        overtake_distance = self.ttc_safe * delta_v
        if (state[1] - self.dFront[i] - 5) >= overtake_distance:
            self.laneFlag[i] = 0
            self.triggerFlag[i] = 0
            self.punish = self.punish + 1
            self.countOPI[i] = 0

        if self.dFront[i] - state[1] > 100:
            self.laneFlag[i] = 0
            self.punish = self.punish - 1

    def trafficFlowCal(self, state):
        front_distance_min = -500
        front_distance_max = 500
        front_position_y_min = state + front_distance_min
        front_position_y_max = state + front_distance_max
        if front_position_y_max > self.maxRoadLength:
            front_position_y_max = self.maxRoadLength
        if front_position_y_min < 0:
            front_position_y_min = 0
        L = front_position_y_max - front_position_y_min
        target_lane0 = 'Lane_0'
        target_lane1 = 'Lane_1'
        target_lane2 = 'Lane_2'
        target_lane0_vehicles = traci.lane.getLastStepVehicleIDs(target_lane0)
        target_lane1_vehicles = traci.lane.getLastStepVehicleIDs(target_lane1)
        target_lane2_vehicles = traci.lane.getLastStepVehicleIDs(target_lane2)
        lane_traffic = {0: 0, 1: 0, 2: 0}
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        for veh_id in target_lane0_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[0] += 1
        for veh_id in target_lane1_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[1] += 1
        for veh_id in target_lane2_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[2] += 1

        return lane_traffic

    def calTTCDri(self, action, state, i):
        x = action

        w_front = 0.5

        d_a = -0.1
        d_n = 0
        d_d = 0.2
        dsID = self.id_list
        u = self.driverStyleReward(i)

        leftFrontDsId = 0
        leftBehDsId = 0
        rightFrontDsId = 0
        rightBehDsId = 0
        middleFrontDsId = 0
        u_lf = []
        u_lb = []
        u_rf = []
        u_rb = []
        u_mb = []
        if x == -1:
            if state[6] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[2]) > 0:
                        leftFrontDsId = dsID[2][0]
                    for car, values in u.items():
                        if car == leftFrontDsId:
                            u_lf = values
                    if leftFrontDsId != 0 & len(u_lf) != 0:
                        dr = u_lf[0] * d_a + u_lf[1] * d_n + u_lf[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V1 = state[0] - state[6]
                delta_D1 = state[7]
                if delta_V1 <= 0:
                    TCC_front = 10
                else:
                    TCC_front = (delta_D1 / delta_V1) * dr
            else:
                TCC_front = 10
            if state[8] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[3]) > 0:
                        leftBehDsId = dsID[3][0]
                    for car, values in u.items():
                        if car == leftBehDsId:
                            u_lb = values
                    if leftBehDsId != 0 & len(u_lb) != 0:
                        dr = u_lb[0] * d_a + u_lb[1] * d_n + u_lb[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V2 = state[0] - state[8]
                delta_D2 = state[9]
                if delta_V2 >= 0:
                    TCC_back = 10
                else:
                    TCC_back = (delta_D2 / delta_V2) * dr
            else:
                TCC_back = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back

        elif x == 1:
            if state[10] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[4]) > 0:
                        rightFrontDsId = dsID[4][0]
                    for car, values in u.items():
                        if car == rightFrontDsId:
                            u_rf = values
                    if rightFrontDsId != 0 & len(u_rf) != 0:
                        dr = u_rf[0] * d_a + u_rf[1] * d_n + u_rf[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V1 = state[0] - state[10]
                delta_D1 = state[11]
                if delta_V1 <= 0:
                    TCC_front = 10
                else:
                    TCC_front = (delta_D1 / delta_V1) * dr
            else:
                TCC_front = 10
            if state[12] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[5]) > 0:
                        rightBehDsId = dsID[5][0]
                    for car, values in u.items():
                        if car == rightBehDsId:
                            u_rb = values
                    if rightFrontDsId != 0 & len(u_rb) != 0:
                        dr = u_rb[0] * d_a + u_rb[1] * d_n + u_rb[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V2 = state[0] - state[12]
                delta_D2 = state[13]
                if delta_V2 >= 0:
                    TCC_back = 10
                else:
                    TCC_back = (delta_D2 / delta_V2) * dr
            else:
                TCC_back = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back

        else:
            if state[2] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[0]) > 0:
                        middleFrontDsId = dsID[0][0]
                    for car, values in u.items():
                        if car == middleFrontDsId:
                            u_mb = values
                    if middleFrontDsId != 0 & len(u_mb) != 0:
                        dr = u_mb[0] * d_a + u_mb[1] * d_n + u_mb[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V = state[0] - state[2]
                delta_D = state[3]
                if delta_V <= 0:
                    TCC_front = 10
                else:
                    TCC_front = (delta_D / delta_V) * dr
            else:
                TCC_front = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            TCC_surround = TCC_front

        finaTCC = TCC_surround

        return finaTCC

    def priority_trigger_mechanism(self, state, i):
        if state[3] != -1 and (state[3] > self.filteredAbsMin) and (
                (state[0] - state[2] + self.filteredAbsMin2) > 0):
            transformed_new_x = np.log(state[3] - self.filteredAbsMin)

            transformed_new_x2 = np.log((state[0] - state[2]) + self.filteredAbsMin2)

            if (self.confidence_interval[0] <= transformed_new_x <= self.confidence_interval[1]) and (
                    self.confidence_interval2[0] <= transformed_new_x2 <= self.confidence_interval2[1]):
                self.triggerFlag[i] = 1

    def priority_settings(self, state, i):
        th = 1.2
        w1 = 1
        w2 = 1
        w3 = 1
        w4 = 1

        if state[0] > state[2]:
            fv = self.min_max_normalize(state[2] - state[0], 0, 30)
        else:
            fv = 0

        if state[0] > state[2]:
            fdTemp = -math.log((state[3] - 5) / (th * (state[0] - state[2])))
            fd = np.clip(fdTemp, -1, 1)
        else:
            fd = 0

        laneTraffic = self.trafficFlowCal(self.state[1])
        laneindex = state[14]
        if laneindex < self.maxLaneNumber and laneindex > self.minLaneNumber:
            laneTraLeft = laneTraffic[laneindex + 1]
            laneTraRight = laneTraffic[laneindex - 1]
        elif laneindex == self.maxLaneNumber:
            laneTraLeft = -1
            laneTraRight = laneTraffic[laneindex - 1]
        elif laneindex == self.minLaneNumber:
            laneTraLeft = -1
            laneTraRight = laneTraffic[laneindex + 1]

        if laneTraLeft != 0 and laneTraRight != 0:
            ft = max(self.min_max_normalize(laneTraffic[laneindex] / laneTraLeft, 0, 30), self.min_max_normalize(laneTraffic[laneindex] / laneTraRight, 0, 30))
        else:
            ft = 1

        ttc = max(self.calTTCDri(-1, state, i), self.calTTCDri(1, state, i))
        if ttc <= self.ttc_safe-1:
            fs = self.min_max_normalize(ttc, 0, self.ttc_safe-1)
        else:
            fs = 1

        sigma_t = np.random.normal(0, 0.001)

        opi = w1 * fv + w2 * fd + w3 * ft + w4 * fs + sigma_t

        return opi / 3

    def check_lane_change_conflict(self, lineEgo, lineOther, disEgo, disOther, laneChangeFlag, i):
        if laneChangeFlag[i] == 1:
            distance_threshold = 20
            for lc in range(len(laneChangeFlag)):
                if lc != i and laneChangeFlag[lc] == 1 and lineEgo == lineOther:
                    position_diff = abs(disEgo - disOther)
                    if position_diff < distance_threshold:
                        return True
        return False

    def check_velocity_conflict(self, preVel, currVel, i, disEgo, disOther, laneChangeFlag):
        distance_threshold = 20
        position_diff = abs(disEgo - disOther)

        if position_diff < distance_threshold:
            for lc in range(len(laneChangeFlag)):
                if lc != i and laneChangeFlag[lc] == 1 and currVel < preVel:
                    return True
        return False

    def min_max_normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def driverStyleReward(self, i):
        clusterCenterNumber = 3
        weight = 2
        ds = driverStyleCluster()
        new_memberships = {}
        mean_value = self.calculateAverage()
        if self.laneFlag[i] == 1:
            data = self.historyData.values.tolist()

            points = [Point(clusterCenterNumber) for _ in range(len(data))]

            for i, point in enumerate(points):
                x, y = float(data[i][0]), float(data[i][1])
                point.x = x
                point.y = y

            filtered_data = ds.remove_outliers(points)
            self.clusterCenterGroup, clusterCenterTrace = ds.fuzzyCMeansClustering(filtered_data, clusterCenterNumber, weight)
            categories = ds.categorize_clusters(filtered_data, self.clusterCenterGroup)

            for car, values in mean_value.items():
                new_data_point = Point(clusterCenterNumber, x=values['averageVel'], y=values['averageAcc'])
                new_memberships[car] = ds.calculateMembership(new_data_point, self.clusterCenterGroup, weight)
        else:
            for car, values in mean_value.items():
                new_data_point = Point(clusterCenterNumber, x=values['averageVel'], y=values['averageAcc'])
                ds.updateClusterCenter(new_data_point, self.clusterCenterGroup, weight)
                new_memberships[car] = ds.calculateMembership(new_data_point, self.clusterCenterGroup, weight)

        return new_memberships

    def updateReward(self, action, state, i):
        a_max = 5

        w_sd = 1
        w_comfVel = 1
        w_ef = 1
        w_pr = 1
        w_no = 1

        TCC_surround = self.calTTCDri(action, state, i)
        self.finaTCC = TCC_surround

        if (TCC_surround <= self.ttc_safe - 1) and TCC_surround > 0:
            r_dis = -1 * self.min_max_normalize(TCC_surround, 0, self.ttc_safe - 1)
        elif TCC_surround < 0:
            r_dis = -1
        else:
            r_dis = 1
        r_sd = r_dis

        if action == -1:
            velafter = state[8]
        elif action == 1:
            velafter = state[12]
        else:
            velafter = -1
        if velafter != -1:
            r_ef = (velafter - self.velbefore) / self.maxVelocity
        else:
            r_ef = 1

        r_comf = - self.min_max_normalize(abs(state[15]), 0, a_max)
        if state[0] > self.vFront[i]:
            r_v = 2 * self.min_max_normalize(state[0] - self.vFront[i], 0, 15)
        else:
            r_v = -2 * self.min_max_normalize(self.vFront[i] - state[0], 0, 15)

        r_comfVel = r_comf + r_v

        self.priority_trigger_mechanism(state, i)
        r_pr = 1

        if self.triggerFlag.count(1) > 0:
            if self.countOPI[i] == 0:
                self.countOPI[i] += 1
                self.opi[i] = self.priority_settings(state, i)
            distance, lane_id, vehicle_speed = self.getCVInformation()
            egoVehicle = self.AutoCarIDAll[i]
            for veh, _ in enumerate(self.AutoCarIDAll):
                r_pp = np.array([])
                if egoVehicle != self.AutoCarIDAll[veh]:
                    r_pr1 = 0
                    r_pr2 = 0
                    flagGiveLine = self.check_lane_change_conflict(lane_id[i], lane_id[veh], distance[i],
                                                              distance[veh], self.lanChangeFlag, i)

                    currentOtherState = self._findCurrentOtherState(veh, i)
                    currentOtherPri = self.priority_settings(currentOtherState, i)
                    priority_diff = self.opi[i] - currentOtherPri
                    if flagGiveLine:
                        if priority_diff < 0:
                            r_pr1 = priority_diff * self.opi[i]
                        else:
                            r_pr1 = (1 + priority_diff) * self.opi[i]

                    flagGiveVel = self.check_velocity_conflict(self.PrevSpeed, vehicle_speed[i], i, distance[i], distance[veh], self.lanChangeFlag)
                    if flagGiveVel:
                        if priority_diff > 0:
                            r_pr2 = priority_diff * self.opi[i]
                        else:
                            r_pr2 = (1 + priority_diff) * self.opi[i]
                    r_p = 0.5 * r_pr1 + 0.5 * r_pr2
                    r_pp = np.concatenate((r_pp, [r_p]))
                r_pr = np.mean(r_pp)
        else:
            r_pr = 0
        r_no = self.punish

        r_total = w_sd * r_sd + w_comfVel * r_comfVel + w_ef * r_ef + w_pr * r_pr + w_no * r_no

        return r_total

    def getFinaTCC(self):

        return self.finaTCC


