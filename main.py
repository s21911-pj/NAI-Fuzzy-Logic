
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl





praceDomowe = ctrl.Antecedent(np.arange(0, 11, 1), 'praceDomowe')
kolokwia = ctrl.Antecedent(np.arange(0, 11, 1), 'kolokwia')
aktywnosc = ctrl.Antecedent(np.arange(0, 11, 1), 'aktywnosc')
ocena = ctrl.Consequent(np.arange(0, 50, 1), 'ocena')


praceDomowe.automf(5)
kolokwia.automf(5)
aktywnosc.automf(5)


ocena['1'] = fuzz.trimf(ocena.universe, [0, 0, 13])
ocena['2'] = fuzz.trimf(ocena.universe, [0, 13, 25])
ocena['3'] = fuzz.trimf(ocena.universe, [13, 25, 25])
ocena['4'] = fuzz.trimf(ocena.universe, [25, 37, 37])
ocena['5'] = fuzz.trimf(ocena.universe, [37, 45, 45])
ocena['6'] = fuzz.trimf(ocena.universe, [45, 49, 49])



praceDomowe['average'].view()

kolokwia.view()

aktywnosc.view()

ocena.view()


rule1 = ctrl.Rule(praceDomowe['poor'] | aktywnosc['poor'] | kolokwia['poor'], ocena['1'])
rule2 = ctrl.Rule(praceDomowe['average'] | aktywnosc['average'] | kolokwia['average'], ocena['2'])
rule3 = ctrl.Rule(praceDomowe['poor'] | aktywnosc['average'] | kolokwia['average'], ocena['3'])
rule4 = ctrl.Rule(praceDomowe['good'] | aktywnosc['good'] | kolokwia['good'], ocena['4'])
rule5 = ctrl.Rule(praceDomowe['average'] | aktywnosc['good'] | kolokwia['good'], ocena['5'])
rule6 = ctrl.Rule(praceDomowe['good'] | aktywnosc['good'] | kolokwia['good'], ocena['6'])







ocenianie_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])



ocenianie = ctrl.ControlSystemSimulation(ocenianie_ctrl)



ocenianie.input['kolokwia'] = 1
ocenianie.input['aktywnosc'] = 1
ocenianie.input['praceDomowe'] = 1


ocenianie.compute()


print (ocenianie.output['ocena'])
ocena.view(sim=ocenianie)


plt.show() 