#
# Tello Python3 Control Demo
#
# http://www.ryzerobotics.com/
#
# 1/1/2018

import threading
import socket
import sys
import time
import platform

host = ''
port = 9000
locaddr = (host,port)


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tello_address = ('192.168.10.1', 8889)

sock.bind(locaddr)

def recv():
    count = 0
    while True:
        try:
            data, server = sock.recvfrom(1518)
            print(data.decode(encoding="utf-8"))
        except Exception:
            print ('\nExit . . .\n')
            break


print ('\r\n\r\nTello Python3 Demo.\r\n')

print ('Tello: command takeoff land flip forward back left right \r\n       up down cw ccw speed speed?\r\n')

print ('end -- quit demo.\r\n')


#recvThread create
recvThread = threading.Thread(target=recv)
recvThread.start()
def takeoff():
    msgs = []
    msgs.append("command")
    msgs.append("takeoff")
    for msg in msgs:
        msg = msg.encode(encoding="utf-8")
        sent = sock.sendto(msg, tello_address)
def forward():
    msgs = []
    msgs.append("forward 20")
    msgs.append("left 10")
    #msgs.append("flip l")
    #msgs.append("left r")
    msgs.append("back 20")
    msgs.append("right 10")
    for msg in msgs:
        msg = msg.encode(encoding="utf-8")
        while True:
            sock.settimeout(10)
            try:
                sock.sendto(msg, tello_address)
                break
            except sock.timeout:
                continue



def get_input():
    try:
        msg = input("");
        if not msg:
            return
        if 'end' in msg:
            print ('...')
            sock.close()
            return
        return msg
    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close()
        return
takeoff()
#time.sleep(5)
#forward()
while True:
    msg = get_input()
    msg = msg.encode(encoding="utf-8")
    sent = sock.sendto(msg, tello_address)
