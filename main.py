
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl





home_work = ctrl.Antecedent(np.arange(0, 11, 1), 'home_work')
colloquia = ctrl.Antecedent(np.arange(0, 11, 1), 'colloquia')
activity = ctrl.Antecedent(np.arange(0, 11, 1), 'activity')
rating = ctrl.Consequent(np.arange(0, 50, 1), 'rating')


home_work.automf(7)
colloquia.automf(7)
activity.automf(7)


rating['1'] = fuzz.trimf(rating.universe, [0, 0, 13])
rating['2'] = fuzz.trimf(rating.universe, [0, 13, 25])
rating['3'] = fuzz.trimf(rating.universe, [13, 25, 25])
rating['4'] = fuzz.trimf(rating.universe, [25, 37, 37])
rating['5'] = fuzz.trimf(rating.universe, [37, 45, 45])
rating['6'] = fuzz.trimf(rating.universe, [45, 49, 49])



home_work['average'].view()

colloquia.view()

activity.view()

rating.view()


rule1 = ctrl.Rule(home_work['poor'] | activity['poor'] | colloquia['poor'], rating['1'])
rule2 = ctrl.Rule(home_work['average'] | activity['average'] | colloquia['average'], rating['2'])
rule3 = ctrl.Rule(home_work['poor'] | activity['average'] | colloquia['average'], rating['3'])
rule4 = ctrl.Rule(home_work['good'] | activity['good'] | colloquia['good'], rating['4'])
rule5 = ctrl.Rule(home_work['average'] | activity['good'] | colloquia['good'], rating['5'])
rule6 = ctrl.Rule(home_work['good'] | activity['good'] | colloquia['excellent'], rating['6'])







assessment_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])



assessment = ctrl.ControlSystemSimulation(assessment_ctrl)



assessment.input['colloquia'] = 9
assessment.input['activity'] = 9
assessment.input['home_work'] = 9


assessment.compute()


print (assessment.output['rating'])
rating.view(sim=assessment)


plt.show() 