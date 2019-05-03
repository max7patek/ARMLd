from flask import Flask, request
import tellopy
import numpy as np
import math
from unittest.mock import MagicMock

import server_calibration as CALIBRATION

assert CALIBRATION.PRIMARY_AXIS != CALIBRATION.VERTICAL_AXIS


app = Flask(__name__)

speed = 100
landed = True

drone = None

DIM_TO_DIR = {
    CALIBRATION.PRIMARY_AXIS : ('forward', 'backward'),
    CALIBRATION.VERTICAL_AXIS : ('up', 'down'),
    ({0, 1, 2} - {CALIBRATION.PRIMARY_AXIS, CALIBRATION.VERTICAL_AXIS}).pop() : ('left', 'right')
}

def magnitude(array):
    sum = 0
    for i in array:
        sum += i*i
    return math.sqrt(sum)

def expert_action(p2_pos, ball_pos, ball_vel):
    if magnitude(p2_pos - ball_pos) > 30:
        rollout = ball_pos + 12*ball_vel
        #rollout[0] += 6 * self.ball_x_accel * (-1 if self.ball.pos[0] < self.width/2 else 1)
        vector = rollout - p2_pos
        if CALIBRATION.AGENT_HIGH:
            vector[CALIBRATION.PRIMARY_AXIS] = CALIBRATION.WIDTH*.95 - p2_pos[CALIBRATION.PRIMARY_AXIS]
        else:
            vector[CALIBRATION.PRIMARY_AXIS] = CALIBRATION.WIDTH*.05 - p2_pos[CALIBRATION.PRIMARY_AXIS]
    else:
        rollout = ball_pos + 2*ball_vel
        #rollout[0] += self.ball_x_accel * (-1 if self.ball.pos[0] < self.width/2 else 1)
        vector = rollout - p2_pos
    return vector

def correct_orientation(p2_rad):
    rads = p2_rad - CALIBRATION.RADIAN_OFFSET
    rads %= 2*math.pi
    if rads < math.pi:
        drone.counter_clockwise(rads*CALIBRATION.ORIENTATION_CORRECTION)
        return 'counter_clockwise:' + str(rads*CALIBRATION.ORIENTATION_CORRECTION)
    else:
        drone.clockwise((2*math.pi - rads)*CALIBRATION.ORIENTATION_CORRECTION)
        return 'clockwise:' + str((2*math.pi - rads)*CALIBRATION.ORIENTATION_CORRECTION)

def approximate_velocity(target_vel):
    dim = max((0, 1, 2), key=lambda i: abs(target_vel[i]))
    direction = DIM_TO_DIR[dim][int(target_vel[dim] < 0)]
    speed = abs(target_vel[dim] * CALIBRATION.DRONE_SPEED_PER_MEAS_SPEED)
    getattr(drone, direction)(speed)
    return direction + ':' + str(speed)

@app.route("/", methods=['POST'])
def hello():
    """
        POST request sending game state and triggering command to the drone.
        Should include 1 arg, 'data', that is comma-seperated:
            1. ball_pos
            2. ball_vel
            3. p1_pos  
            4. p1_vel  
            5. p2_pos  
            6. p2_vel  
            7. p2_rad  
        each being a 3-tuple (except for p2_rad), flattenned into one large 19-tuple.
        p2_rad should be a single float representing the orientation of the drone in radians.
    """
    global drone
    if drone is None:
        print("connecting to drone")
        drone = MagicMock(tellopy.Tello())
        drone.connect()
    data = tuple(map(float, request.form['data'].split(',')))
    if len(data) != 19:
        return "Must pass 19 comma-seperated values as 'data'"
    vecs = []
    for i in range(0, 6*3, 3):
        vecs.append(data[i:i+3])
    ball_pos = np.array(vecs[0])
    ball_vel = np.array(vecs[1])
    p1_pos   = np.array(vecs[2])
    p1_vel   = np.array(vecs[3])
    p2_pos   = np.array(vecs[4])
    p2_vel   = np.array(vecs[5])
    p2_rad   = data[-1]

    orientation_correction = correct_orientation(p2_rad)
    dir_and_speed = approximate_velocity(expert_action(p2_pos, ball_pos, ball_vel))

    return dir_and_speed + " and " + orientation_correction  # for debugging



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

