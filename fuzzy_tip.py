from simpful import *

# A simple fuzzy inference system for the tipping problem
# Create a fuzzy system object
FS = FuzzySystem()

# Define fuzzy sets and linguistic variables
S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=5), term='poor')
S_2 = FuzzySet(function=Triangular_MF(a=0, b=5, c=10), term='good')
S_3 = FuzzySet(function=Triangular_MF(a=5, b=10, c=10), term='excellent')
FS.add_linguistic_variable('Service', LinguisticVariable([S_1, S_2, S_3], concept='Service quality', universe_of_discourse=[0, 10]))

F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=10), term='rancid')
F_2 = FuzzySet(function=Triangular_MF(a=0, b=10, c=10), term='delicious')
FS.add_linguistic_variable('Food', LinguisticVariable([F_1, F_2], concept='Food quality', universe_of_discourse=[0, 10]))

# Define output fuzzy sets and linguistic variable
T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=15), term='small')
T_2 = FuzzySet(function=Triangular_MF(a=0, b=15, c=30), term='average')
T_3 = FuzzySet(function=Trapezoidal_MF(a=15, b=30, c=30), term='generous')
FS.add_linguistic_variable('Tip', LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 35]))

FS.plot_variable('Service')
FS.plot_variable('Food')
FS.plot_variable('Tip')

# Define fuzzy rules
R1 = 'IF (Service IS poor) OR (Food IS rancid) THEN (Tip IS small)'
R2 = 'IF (Service IS good) THEN (Tip IS average)'
R3 = 'IF (Service IS excellent) OR (Food IS delicious) THEN (Tip IS generous)'
FS.add_rules([R1, R2, R3])

# Set antecedents values
FS.set_variable('Service', 2)
FS.set_variable('Food', 8)

# Perform Mamdani inference and print output
print(FS.Mamdani_inference(['Tip']))

