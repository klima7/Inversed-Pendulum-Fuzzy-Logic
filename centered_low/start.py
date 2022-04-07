#
# Template Author: Tomasz Jaworski
#

import gym # Instalacja: https://github.com/openai/gym
import time
import math
import numpy as np
from helper import HumanControl, Keys, CartForce
import matplotlib.pyplot as plt


import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

#
# przygotowanie środowiska
#
control = HumanControl()
env = gym.make('gym_PSI:CartPole-v2')
env.reset()
env.render()


def on_key_press(key: int, mod: int):
    global control
    force = 10
    if key == Keys.LEFT:
        control.UserForce = force * CartForce.UNIT_LEFT # krok w lewo
    if key == Keys.RIGHT:
        control.UserForce = force * CartForce.UNIT_RIGHT # krok w prawo
    if key == Keys.P: # pauza
        control.WantPause = True
    if key == Keys.R: # restart
        control.WantReset = True
    if key == Keys.ESCAPE or key == Keys.Q: # wyjście
        control.WantExit = True

env.unwrapped.viewer.window.on_key_press = on_key_press

#########################################################
# KOD INICJUJĄCY - do wypełnienia
#########################################################

"""

1. Określ dziedzinę dla każdej zmiennej lingwistycznej. Każda zmienna ma własną dziedzinę.
2. Zdefiniuj funkcje przynależności dla wybranych przez siebie zmiennych lingwistycznych.
3. Wyświetl je, w celach diagnostycznych.

Przykład wyświetlania:

fig, (ax0) = plt.subplots(nrows=1, figsize=(8, 9))

ax0.plot(x_variable, variable_left, 'b', linewidth=1.5, label='Left')
ax0.plot(x_variable, variable_zero, 'g', linewidth=1.5, label='Zero')
ax0.plot(x_variable, variable_right, 'r', linewidth=1.5, label='Right')
ax0.set_title('Angle')
ax0.legend()


plt.tight_layout()
plt.show()
"""

#########################################################
# KONIEC KODU INICJUJĄCEGO
#########################################################


#
# Główna pętla symulacji
#
while not control.WantExit:

    #
    # Wstrzymywanie symulacji:
    # Pierwsze wciśnięcie klawisza 'p' wstrzymuje; drugie wciśnięcie 'p' wznawia symulację.
    #
    if control.WantPause:
        control.WantPause = False
        while not control.WantPause:
            time.sleep(0.1)
            env.render()
        control.WantPause = False

    #
    # Czy użytkownik chce zresetować symulację?
    if control.WantReset:
        control.WantReset = False
        env.reset()


    ###################################################
    # ALGORYTM REGULACJI - do wypełnienia
    ##################################################

    """
    Opis wektora stanu (env.state)
        cart_position   -   Położenie wózka w osi X. Zakres: -2.5 do 2.5. Ppowyżej tych granic wózka znika z pola widzenia.
        cart_velocity   -   Prędkość wózka. Zakres +- Inf, jednak wartości powyżej +-2.0 generują zbyt szybki ruch.
        pole_angle      -   Pozycja kątowa patyka, a<0 to odchylenie w lewo, a>0 odchylenie w prawo. Pozycja kątowa ma
                            charakter bezwzględny - do pozycji wliczane są obroty patyka.
                            Ze względów intuicyjnych zaleca się konwersję na stopnie (+-180).
        tip_velocity    -   Prędkość wierzchołka patyka. Zakres +- Inf. a<0 to ruch przeciwny do wskazówek zegara,
                            podczas gdy a>0 to ruch zgodny z ruchem wskazówek zegara.
                            
    Opis zadajnika akcji (fuzzy_response):
        Jest to wartość siły przykładana w każdej chwili czasowej symulacji, wyrażona w Newtonach.
        Zakładany krok czasowy symulacji to env.tau (20 ms).
        Przyłożenie i utrzymanie stałej siły do wózka spowoduje, że ten będzie przyspieszał do nieskończoności,
        ruchem jednostajnym.
    """

    cart_position, cart_velocity, pole_angle, tip_velocity = env.state # Wartości zmierzone

    points = 101

    # Membership functions
    x_pos = np.linspace(-2.5, 2.5, points)
    x_cvel = np.linspace(-2, 2, points)
    x_angle = np.linspace(-2, 2, points)
    x_tvel = np.linspace(-5, 5, points)
    x_force = np.linspace(-30, 30, points)

    pos_left = fuzz.trapmf(x_pos, [-2.5, -2.5, 0, 0])
    pos_center = fuzz.trapmf(x_pos, [-0.3, -0.3, 0.3, 0.3])
    pos_right = fuzz.trapmf(x_pos, [0, 0, 2.5, 2.5])

    cvel_zero = fuzz.trimf(x_cvel, [-1, 0, 1])

    angle_center = fuzz.trimf(x_angle, [-1, 0, 1])
    angle_slight_left = fuzz.trimf(x_angle, [-2, -1, 0])
    angle_slight_right = fuzz.trimf(x_angle, [0, 1, 2])
    angle_left = fuzz.trimf(x_angle, [-2, -2, -1])
    angle_right = fuzz.trimf(x_angle, [1, 2, 2])

    tvel_zero = fuzz.trimf(x_tvel, [-2.5, 0, 2.5])
    tvel_slight_left = fuzz.trimf(x_tvel, [-5, -2.5, 0])
    tvel_slight_right = fuzz.trimf(x_tvel, [0, 2.5, 5])

    force_zero = fuzz.trimf(x_force, [-10, 0, 10])
    force_slight_left = fuzz.trimf(x_force, [-6, -3, 0])
    force_slight_right = fuzz.trimf(x_force, [0, 3, 6])
    force_left = fuzz.trimf(x_force, [-30, -20, -10])
    force_right = fuzz.trimf(x_force, [10, 20, 30])
    force_strong_left = fuzz.trimf(x_force, [-30, -30, -20])
    force_strong_right = fuzz.trimf(x_force, [20, 30, 30])

    # Membership level
    pos_left_level = fuzz.interp_membership(x_pos, pos_left, cart_position)
    pos_center_level = fuzz.interp_membership(x_pos, pos_center, cart_position)
    pos_right_level = fuzz.interp_membership(x_pos, pos_right, cart_position)

    cvel_zero_level = fuzz.interp_membership(x_cvel, cvel_zero, cart_velocity)

    angle_center_level = fuzz.interp_membership(x_angle, angle_center, pole_angle)
    angle_slight_left_level = fuzz.interp_membership(x_angle, angle_slight_left, pole_angle)
    angle_slight_right_level = fuzz.interp_membership(x_angle, angle_slight_right, pole_angle)
    angle_left_level = fuzz.interp_membership(x_angle, angle_left, pole_angle)
    angle_right_level = fuzz.interp_membership(x_angle, angle_right, pole_angle)

    tvel_zero_level = fuzz.interp_membership(x_tvel, tvel_zero, tip_velocity)
    tvel_slight_left_level = fuzz.interp_membership(x_tvel, tvel_slight_left, tip_velocity)
    tvel_slight_right_level = fuzz.interp_membership(x_tvel, tvel_slight_right, tip_velocity)

    # Rules
    zero_rule = min(angle_center_level, angle_center_level, cvel_zero_level)

    slight_left_rule = min(angle_center_level, tvel_zero_level, pos_left_level, cvel_zero_level)
    left_rule = min(angle_slight_left_level, tvel_zero_level)
    strong_left_rule_1 = min(angle_slight_left_level, tvel_slight_left_level)
    strong_left_rule_2 = min(angle_left_level, tvel_zero_level)
    strong_left_rule = max(strong_left_rule_1, strong_left_rule_2)

    slight_right_rule = min(angle_center_level, tvel_zero_level, pos_right_level, cvel_zero_level)
    right_rule = min(angle_slight_right_level, tvel_zero_level)
    strong_right_rule_1 = min(angle_slight_right_level, tvel_slight_right_level)
    strong_right_rule_2 = min(angle_right_level, tvel_zero_level)
    strong_right_rule = max(strong_right_rule_1, strong_right_rule_2)

    # Activation functions
    zero_activation = np.fmin(zero_rule, force_zero)
    
    slight_left_activation = np.fmin(slight_left_rule, force_slight_left)
    left_activation = np.fmin(left_rule, force_left)
    strong_left_activation = np.fmin(strong_left_rule, force_strong_left)
    
    slight_right_activation = np.fmin(slight_right_rule, force_slight_right)
    right_activation = np.fmin(right_rule, force_right)
    strong_right_activation = np.fmin(strong_right_rule, force_strong_right)

    # Aggregation
    force_aggregated = np.fmax.reduce([zero_activation, slight_left_activation, left_activation, strong_left_activation,
                            slight_right_activation, right_activation, strong_right_activation])

    # Defuzzification
    force_x = fuzz.defuzz(x_force, force_aggregated, 'centroid')
    force = fuzz.interp_membership(x_force, force_aggregated, force_x)

    fuzzy_response = force_x

    """
    
    1. Przeprowadź etap rozmywania, w którym dla wartości zmierzonych wyznaczone zostaną ich przynależności do poszczególnych
       zmiennych lingwistycznych. Jedno fizyczne wejście (źródło wartości zmierzonych, np. położenie wózka) posiada własną
       zmienną lingwistyczną.
       
       Sprawdź funkcję interp_membership
       
    2. Wyznacza wartości aktywacji reguł rozmytych, wyznaczając stopień ich prawdziwości.
       Przykład reguły:
       JEŻELI kąt patyka jest zerowy ORAZ prędkość wózka jest zerowa TO moc chwilowa jest zerowa
       JEŻELI kąt patyka jest lekko ujemny ORAZ prędkość wózka jest zerowa TO moc chwilowa jest lekko ujemna
       JEŻELI kąt patyka jest średnio ujemny ORAZ prędkość wózka jest lekko ujemna TO moc chwilowa jest średnio ujemna
       JEŻELI kąt patyka jest szybko rosnący w kierunku ujemnym TO moc chwilowa jest mocno ujemna
       .....
       
       Przyjmując, że spójnik LUB (suma rozmyta) to max() a ORAZ/I (iloczyn rozmyty) to min() sprawdź funkcje fmax i fmin.
    
    
    3. Przeprowadź agregację reguł o tej samej konkluzji.
       Jeżeli masz kilka reguł, posiadających tę samą konkluzję (ale różne przesłanki) to poziom aktywacji tych reguł
       należy agregować tak, aby jedna konkluzja miała jeden poziom aktywacji. Skorzystaj z sumy rozmytej.
    
    4. Dla każdej reguły przeprowadź operację wnioskowania Mamdaniego.
       Operatorem wnioskowania jest min().
       Przykład: Jeżeli lingwistyczna zmienna wyjściowa ForceToApply ma 5 wartości (strong left, light left, idle, light right, strong right)
       to liczba wyrażeń wnioskujących wyniesie 5 - po jednym wywołaniu operatora Mamdaniego dla konkluzji.
       
       W ten sposób wyznaczasz aktywacje poszczególnych wartości lingwistycznej zmiennej wyjściowej.
       Uważaj - aktywacja wartości zmiennej lingwistycznej w konkluzji to nie liczba a zbiór rozmyty.
       Ponieważ stosujesz operator min(), to wynikiem będzie "przycięty od góry" zbiór rozmyty. 
       
    5. Agreguj wszystkie aktywacje dla danej zmiennej wyjściowej.
    
    6. Dokonaj defuzyfikacji (np. całkowanie ważone - centroid).
    
    7. Czym będzie wyjściowa wartość skalarna?
    
    """

    #
    # KONIEC algorytmu regulacji
    #########################

    # Jeżeli użytkownik chce przesunąć wózek, to jego polecenie ma wyższy priorytet
    if control.UserForce is not None:
        applied_force = control.UserForce
        control.UserForce = None
    else:
        applied_force = fuzzy_response

    #
    # Wyświetl stan środowiska oraz wartość odpowiedzi regulatora na ten stan.
    print(
        f"cpos={cart_position:8.4f}, cvel={cart_velocity:8.4f}, pang={pole_angle:8.4f}, tvel={tip_velocity:8.4f}, force={applied_force:8.4f}")

    #
    # Wykonaj krok symulacji
    env.step(applied_force)

    #
    # Pokaż kotku co masz w środku
    env.render()

#
# Zostaw ten patyk!
env.close()

