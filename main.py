# Authors:
# Karol Kraus s20687
# Piotr Mastalerz s21911

# This project accepts student's grades for different activities during classes, compute the values and returns
# the number of final points
# Enter values are provided in csv file - data.csv

# environmental instructions
# create venv
#   python3 -m venv venv
# activate venv
#   source venv/bin/activate
# install packages
#   pip3 install -r requirements.txt
# run app
#   python3 main.py


import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import csv
import math


def calculate(a, b, c):
    # variables representing inputs of problem - hold universe variables and membership functions
    # home_work - points of homework
    # home_work - points of colloquia
    # activity - points of activity during classes
    # rating - OUTPUT
    home_work = ctrl.Antecedent(np.arange(0, 11, 1), 'home_work')
    colloquia = ctrl.Antecedent(np.arange(0, 11, 1), 'colloquia')
    activity = ctrl.Antecedent(np.arange(0, 11, 1), 'activity')
    rating = ctrl.Consequent(np.arange(0, 50, 1), 'rating')

    # Automatically populate the universe with membership functions.
    home_work.automf(7)
    colloquia.automf(7)
    activity.automf(7)

    # Define the rating result based on gained points
    rating['1'] = fuzz.trimf(rating.universe, [0, 0, 13])
    rating['2'] = fuzz.trimf(rating.universe, [0, 13, 25])
    rating['3'] = fuzz.trimf(rating.universe, [13, 25, 25])
    rating['4'] = fuzz.trimf(rating.universe, [25, 37, 37])
    rating['5'] = fuzz.trimf(rating.universe, [37, 45, 45])
    rating['6'] = fuzz.trimf(rating.universe, [45, 49, 49])

    #  define the relationship between input and output variables. We defined rule for each rating
    rule1 = ctrl.Rule(home_work['poor'] | activity['poor'] | colloquia['poor'], rating['1'])
    rule2 = ctrl.Rule(home_work['average'] | activity['average'] | colloquia['average'], rating['2'])
    rule3 = ctrl.Rule(home_work['poor'] | activity['average'] | colloquia['average'], rating['3'])
    rule4 = ctrl.Rule(home_work['good'] | activity['good'] | colloquia['good'], rating['4'])
    rule5 = ctrl.Rule(home_work['average'] | activity['good'] | colloquia['good'], rating['5'])
    rule6 = ctrl.Rule(home_work['good'] | activity['good'] | colloquia['excellent'], rating['6'])

    # create a control system
    assessment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    assessment = ctrl.ControlSystemSimulation(assessment_ctrl)

    # Pass inputs to the ControlSystem
    assessment.input['colloquia'] = a
    assessment.input['activity'] = b
    assessment.input['home_work'] = c

    # compute values
    assessment.compute()

    # return computed value
    return math.floor(assessment.output['rating'])


if __name__ == '__main__':

    # read the file and print result for each person:
    with open('data.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            print(row['name'], 'get score:',
                  calculate(int(row['colloguia']), int(row['Activity']), int(row['home_work'])))
