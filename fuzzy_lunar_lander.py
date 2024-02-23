import gym
from simpful import *
import numpy as np

env = gym.make("LunarLander-v2", render_mode='human')
env.reset()

FS = FuzzySystem()

Px_1 = FuzzySet(function=Triangular_MF(a=-90, b=-90, c=-0.1), term='Negative')
Px_2 = FuzzySet(function=Triangular_MF(a=-0.1, b=0, c=0.1), term='Zero')
Px_3 = FuzzySet(function=Triangular_MF(a=0.1, b=90, c=90), term='Positive')
FS.add_linguistic_variable('PositionX', LinguisticVariable([Px_1, Px_2, Px_3], concept='PositionX', universe_of_discourse=[-90, 90]))

# Py_1 = FuzzySet(function=Triangular_MF(a=-90, b=-90, c=0), term='Negative')
# Py_2 = FuzzySet(function=Triangular_MF(a=-45, b=0, c=45), term='Zero')
# Py_3 = FuzzySet(function=Triangular_MF(a=0, b=90, c=90), term='Positive')
# FS.add_linguistic_variable('PositionY', LinguisticVariable([Py_1, Py_2, Py_3], concept='PositionY', universe_of_discourse=[-90, 90]))

Vx_1 = FuzzySet(function=Triangular_MF(a=-5, b=-5, c=-0.2), term='Negative')
Vx_2 = FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term='Zero')
Vx_3 = FuzzySet(function=Triangular_MF(a=0.2, b=5, c=5), term='Positive')
FS.add_linguistic_variable('VelocityX', LinguisticVariable([Vx_1, Vx_2, Vx_3], concept='VelocityX', universe_of_discourse=[-5, 5]))

Vy_1 = FuzzySet(function=Triangular_MF(a=-5, b=-5, c=-0.2), term='Negative')
Vy_2 = FuzzySet(function=Triangular_MF(a=-0.2, b=0, c=0.2), term='Zero')
Vy_3 = FuzzySet(function=Triangular_MF(a=0.2, b=5, c=5), term='Positive')
FS.add_linguistic_variable('VelocityY', LinguisticVariable([Vy_1, Vy_2, Vy_3], concept='VelocityY', universe_of_discourse=[-5, 5]))

A_1 = FuzzySet(function=Triangular_MF(a=-3, b=-3, c=-0.3), term='Negative')
A_2 = FuzzySet(function=Triangular_MF(a=-0.3, b=0, c=0.3), term='Zero')
A_3 = FuzzySet(function=Triangular_MF(a=0.3, b=3, c=3), term='Positive')
FS.add_linguistic_variable('Angle', LinguisticVariable([A_1, A_2, A_3], concept='Angle', universe_of_discourse=[-3, 3]))

Ac_1 = FuzzySet(function=Triangular_MF(a=-0.5, b=0, c=0.5), term='Nothing')
Ac_2 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1.5), term='Left')
Ac_3 = FuzzySet(function=Triangular_MF(a=1.5, b=2, c=2.5), term='Up')
Ac_4 = FuzzySet(function=Triangular_MF(a=2.5, b=3, c=3.5), term='Right')
FS.add_linguistic_variable('Action', LinguisticVariable([Ac_1, Ac_2, Ac_3, Ac_4], concept='Action', universe_of_discourse=[-1, 4]))

FS.plot_variable('PositionX')
FS.plot_variable('VelocityX')
FS.plot_variable('VelocityY')
FS.plot_variable('Angle')
FS.plot_variable('Action')

R1 = 'IF (PositionX IS Negative) OR (VelocityX IS Negative) THEN (Action IS Right)'
R2 = 'IF (PositionX IS Positive) OR (VelocityX IS Positive) THEN (Action IS Left)'
R3 = 'IF (Angle IS Negative) THEN (Action IS Left)'
R4 = 'IF (Angle IS Positive) THEN (Action IS Right)'
R5 = 'IF (VelocityY IS Negative) THEN (Action IS Up)'
R6 = 'IF (PositionX IS Zero) AND (VelocityX IS Zero) AND (VelocityY IS Zero) AND (Angle IS Zero) THEN (Action IS Nothing)'
FS.add_rules([R1, R2, R3, R4, R5, R6])

for i in range(10):
    env.reset()
    done = False
    action = 0
    score = 0
    while not done:

        observation, reward, done, _, _ = env.step(action)
        score += reward

        positionX = observation[0]
        velocityX = observation[2]
        velocityY = observation[3]
        angle = observation[4]

        FS.set_variable('PositionX', positionX)
        FS.set_variable('VelocityX', velocityX)
        FS.set_variable('VelocityY', velocityY)
        FS.set_variable('Angle', angle)

        action = FS.Mamdani_inference(['Action'])

        action = round(action['Action'])

    print('game ', i + 1, ': ', score, sep='')
