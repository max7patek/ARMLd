import time
# import sys
import tellopy
import pygame
import pygame.display
import pygame.key
import pygame.locals
import pygame.font
import os
import datetime
from subprocess import Popen, PIPE

prev_flight_data = None
date_fmt = '%Y-%m-%d_%H%M%S'

controls = {
    'w': 'forward',
    's': 'backward',
    'a': 'left',
    'd': 'right',
    'space': 'up',
    'left shift': 'down',
    'right shift': 'down',
    'q': 'counter_clockwise',
    'e': 'clockwise',
    # arrow keys for fast turns and altitude adjustments
    'right': lambda drone, speed: drone.counter_clockwise(speed*2),
    'left': lambda drone, speed: drone.clockwise(speed*2),
    'down': lambda drone, speed: drone.up(speed*2),
    'up': lambda drone, speed: drone.down(speed*2),
    'o': lambda drone, speed: drone.takeoff(),
    'l': lambda drone, speed: drone.land(),

    '1': lambda drone: drone.set_throttle(1)
}

class FlightDataDisplay(object):
    # previous flight data value and surface to overlay
    _value = None
    _surface = None
    # function (drone, data) => new value
    # default is lambda drone,data: getattr(data, self._key)
    _update = None
    def __init__(self, key, format, color=(255,255,255), update=None):
        self._key = key
        self._format = format
        self._color = color

        if update:
            self._update = update
        else:
            self._update = lambda drone,data: getattr(data, self._key)

    def update(self, drone, data):
        new_value = self._update(drone, data)
        if self._value != new_value:
            self._value = new_value
            self._surface = font.render(self._format % (new_value,), True, self._color)
        return self._surface


def flight_data_mode(drone, *args):
    return (drone.zoom and "VID" or "PIC")


def update_hud(hud, drone, flight_data):
    (w, h) = (158, 0)  # width available on side of screen in 4:3 mode
    blits = []
    for element in hud:
        surface = element.update(drone, flight_data)
        if surface is None:
            continue
        blits += [(surface, (0, h))]
        # w = max(w, surface.get_width())
        h += surface.get_height()
    h += 64  # add some padding
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0))  # remove for mplayer overlay mode
    for blit in blits:
        overlay.blit(*blit)
    pygame.display.get_surface().blit(overlay, (0, 0))
    pygame.display.update(overlay.get_rect())


def status_print(text):
    pygame.display.set_caption(text)


hud = [
    FlightDataDisplay('height', 'ALT %3d'),
    FlightDataDisplay('ground_speed', 'SPD %3d'),
    FlightDataDisplay('battery_percentage', 'BAT %3d%%'),
    FlightDataDisplay('wifi_strength', 'NET %3d%%'),
    FlightDataDisplay(None, 'CAM %s', update=flight_data_mode),
]

def flight_data_handler(event, sender, data):
    global prev_flight_data
    text = str(data)
    if prev_flight_data != text:
        update_hud(hud, sender, data)
        prev_flight_data = text


def main():
    pygame.init()
    pygame.display.init()
    pygame.display.set_mode((1280//2, 720))
    pygame.font.init()

    global font
    font = pygame.font.SysFont("dejavusansmono", 32)

    global wid
    wid = None
    if 'window' in pygame.display.get_wm_info():
        wid = pygame.display.get_wm_info()['window']
    print("Tello video WID:", wid)

    drone = tellopy.Tello()
    drone.connect()

    drone.subscribe(drone.EVENT_FLIGHT_DATA, flight_data_handler)
    # drone.set_throttle(1)
    # can be recalibratedß
    speed = 100

    try:
        while True:
            # Haven't extensively tested what happens when you reduce this too much.
            time.sleep(0.05)

            for e in pygame.event.get():
                if e.type == pygame.locals.KEYDOWN:
                    print("key:", e.key)
                    keyname = pygame.key.name(e.key)
                    print('+' + keyname)
                    if keyname == "escape":
                        exit(0)
                    if keyname in controls:
                        key_handler = controls[keyname]
                        if type(key_handler) == str:
                            getattr(drone, key_handler)(speed)
                            pass
                        else:
                            key_handler(drone, speed)
                            pass
                elif e.type == pygame.locals.KEYUP:
                    keyname = pygame.key.name(e.key)
                    print('-' + keyname)
                    if keyname in controls:
                        key_handler = controls[keyname]
                        if type(key_handler) == str:
                            getattr(drone, key_handler)(0)
                            pass
                        else:
                            key_handler(drone, 0)
                            pass
            # print("hello 2")
                    
    except Exception as e:
        print("------EXCEPTION RAISED-------")
        print("Exception: ", str(e))
        # raise e
    finally:
        print('Shutting down connection to drone...')
        drone.quit()
    exit(1)


if __name__ == '__main__':
    main()
