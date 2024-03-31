import copy
import csv
import math
import random

import numpy as np
import pandas as pd
import pylab

try:
    import psyco

    psyco.full()
except ImportError:
    pass

FLOAT_MAX = 1e100

random.seed(0)

class Point:
    __slots__ = ["x", "y", "group", "membership"]
    def __init__(self, clusterCenterNumber, x=0, y=0, group=0):
        self.x, self.y, self.group = x, y, group
        self.membership = [0.0 for _ in range(clusterCenterNumber)]

class driverStyleCluster:
    def __init__(self):
        pass

    def dealCSVPoints(self, csv_filename, clusterCenterNumber):
        with open(csv_filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            data = list(csv_reader)

        points = [Point(clusterCenterNumber) for _ in range(len(data))]

        for i, point in enumerate(points):
            x, y = float(data[i][0]), float(data[i][1])
            point.x = x
            point.y = y

        return points

    def solveDistanceBetweenPoints(self, pointA, pointB):
        return (pointA.x - pointB.x) * (pointA.x - pointB.x) + (pointA.y - pointB.y) * (pointA.y - pointB.y)

    def getNearestCenter(self, point, clusterCenterGroup):
        minIndex = point.group
        minDistance = FLOAT_MAX
        for index, center in enumerate(clusterCenterGroup):
            distance = self.solveDistanceBetweenPoints(point, center)
            if (distance < minDistance):
                minDistance = distance
                minIndex = index
        return (minIndex, minDistance)

    def kMeansPlusPlus(self, points, clusterCenterGroup):
        clusterCenterGroup[0] = copy.copy(random.choice(points))
        distanceGroup = [0.0 for _ in range(len(points))]
        sum = 0.0
        for index in range(1, len(clusterCenterGroup)):
            for i, point in enumerate(points):
                distanceGroup[i] = self.getNearestCenter(point, clusterCenterGroup[:index])[1]
                sum += distanceGroup[i]
            sum *= random.random()
            for i, distance in enumerate(distanceGroup):
                sum -= distance
                if sum < 0:
                    clusterCenterGroup[index] = copy.copy(points[i])
                    break
        return

    def fuzzyCMeansClustering(self, points, clusterCenterNumber, weight):
        clusterCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
        self.kMeansPlusPlus(points, clusterCenterGroup)
        clusterCenterTrace = [[clusterCenter] for clusterCenter in clusterCenterGroup]
        tolerableError, currentError = 1.0, FLOAT_MAX
        while currentError >= tolerableError:
            for point in points:
                self.getSingleMembership(point, clusterCenterGroup, weight)
            currentCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
            for centerIndex, center in enumerate(currentCenterGroup):
                upperSumX, upperSumY, lowerSum = 0.0, 0.0, 0.0
                for point in points:
                    membershipWeight = pow(point.membership[centerIndex], weight)
                    upperSumX += point.x * membershipWeight
                    upperSumY += point.y * membershipWeight
                    lowerSum += membershipWeight
                center.x = upperSumX / lowerSum
                center.y = upperSumY / lowerSum
            currentError = 0.0
            for index, singleTrace in enumerate(clusterCenterTrace):
                singleTrace.append(currentCenterGroup[index])
                currentError += self.solveDistanceBetweenPoints(singleTrace[-1], singleTrace[-2])
                clusterCenterGroup[index] = copy.copy(currentCenterGroup[index])

        for point in points:
            maxIndex, maxMembership = 0, 0.0
            for index, singleMembership in enumerate(point.membership):
                if singleMembership > maxMembership:
                    maxMembership = singleMembership
                    maxIndex = index
            point.group = maxIndex
        return clusterCenterGroup, clusterCenterTrace

    def getSingleMembership(self, point, clusterCenterGroup, weight):
        distanceFromPoint2ClusterCenterGroup = [self.solveDistanceBetweenPoints(point, clusterCenterGroup[index]) for index
                                                in
                                                range(len(clusterCenterGroup))]
        for centerIndex, singleMembership in enumerate(point.membership):
            sum = 0.0
            isCoincide = [False, 0]
            for index, distance in enumerate(distanceFromPoint2ClusterCenterGroup):
                if distance == 0:
                    isCoincide[0] = True
                    isCoincide[1] = index
                    break
                sum += pow(float(distanceFromPoint2ClusterCenterGroup[centerIndex] / distance), 1.0 / (weight - 1.0))
            if isCoincide[0]:
                if isCoincide[1] == centerIndex:
                    point.membership[centerIndex] = 1.0
                else:
                    point.membership[centerIndex] = 0.0
            else:
                point.membership[centerIndex] = 1.0 / sum

    def showClusterAnalysisResults(self, points, clusterCenterTrace, categories):
        colorMap = {'aggressive': 'r', 'normal': 'b', 'defensive': 'g'}
        pylab.figure(figsize=(9, 9), dpi=80)

        legend_dict = {}

        for point in points:
            color = colorMap[categories[point.group]]
            pylab.plot(point.x, point.y, color + 'o')
            legend_dict[categories[point.group]] = color + 'o'

        for singleTrace in clusterCenterTrace:
            category = categories[clusterCenterTrace.index(singleTrace)]
            color = colorMap[category]
            pylab.plot([center.x for center in singleTrace], [center.y for center in singleTrace], color, linewidth=2)
            legend_dict[category] = color

        legend_handles = [pylab.Line2D([0], [0], color=color[0], marker='o', linestyle='', label=label)
                          for label, color in legend_dict.items()]
        pylab.legend(handles=legend_handles, loc='best')

        pylab.show()

    def calculateMembership(self, new_point, clusterCenterGroup, weight):
        memberships = []
        for center in clusterCenterGroup:
            distance = self.solveDistanceBetweenPoints(new_point, center)
            membership = 1.0 / sum(
                [((distance / self.solveDistanceBetweenPoints(new_point, other_center)) ** (2 / (weight - 1))) for
                 other_center
                 in clusterCenterGroup]
            )
            memberships.append(membership)

        total_membership = sum(memberships)
        normalized_memberships = [membership / total_membership for membership in memberships]

        return normalized_memberships

    def updateClusterCenter(self, new_point, clusterCenterGroup, weight):
        memberships = self.calculateMembership(new_point, clusterCenterGroup, weight)
        for index, center in enumerate(clusterCenterGroup):
            center.x = (1 - memberships[index]) * center.x + memberships[index] * new_point.x
            center.y = (1 - memberships[index]) * center.y + memberships[index] * new_point.y

    def categorize_clusters(self, points, clusterCenterGroup):
        cluster_speeds = {i: [] for i in range(len(clusterCenterGroup))}
        cluster_accelerations = {i: [] for i in range(len(clusterCenterGroup))}

        for point in points:
            cluster_idx = point.group
            cluster_speeds[cluster_idx].append(point.x)
            cluster_accelerations[cluster_idx].append(point.y)

        cluster_scores = []
        for cluster_idx in cluster_speeds:
            speeds = cluster_speeds[cluster_idx]
            accelerations = cluster_accelerations[cluster_idx]
            avg_speed = np.mean(speeds)
            avg_acceleration = np.mean(accelerations)
            speed_std = np.std(speeds)
            acceleration_std = np.std(accelerations)

            score = avg_speed + avg_acceleration + speed_std + acceleration_std
            cluster_scores.append((cluster_idx, score))

        cluster_scores.sort(key=lambda x: x[1], reverse=True)

        categories = ['normal' for _ in clusterCenterGroup]
        categories[cluster_scores[0][0]] = 'aggressive'
        categories[cluster_scores[-1][0]] = 'defensive'

        return categories

    def remove_outliers(self, points):
        data = np.array([[point.x, point.y] for point in points])

        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        non_outliers_mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        filtered_points = [points[i] for i in range(len(points)) if non_outliers_mask[i]]

        return filtered_points

