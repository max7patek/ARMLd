from flask import Flask
import tellopy

app = Flask(__name__)

speed = 100
landed = True
drone = None

@app.route("/")
def hello():
    global landed
    global drone
    if drone is None:
        print("connecting to drone")
        drone = tellopy.Tello()
        drone.connect()
    if landed:
        drone.takeoff()
        ret = "Taking Off"
    else:
        drone.land()
        ret = "Landing"
    landed = not landed
    return ret
    #return "9000?"



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

