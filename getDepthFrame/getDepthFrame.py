
from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import cv2
import rpyc

import time

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

DEPTH_PORT = 20100
depth_stream = None

class depthCommands(rpyc.Service):

    def exposed_echo(self, text):
        return text + " from getDepthFrame"

    def exposed_activateKinect(self):
        return startKinect()

    def exposed_deactivateKinect(self):
        return stopKinect()

    def exposed_getDepth(self, orientation):
        return obstacleMap(orientation)



def obstacleMap(orientation):

    screen = np.zeros((480, 640,1),np.uint8)

    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    arr1d = np.frombuffer(frame_data, dtype=np.uint16)

    depth = arr1d.astype(float)
    depth.shape = (480,640)
    depth = cv2.flip(depth, 1)
    
    np.save(f"depth_{orientation}.npy", depth)

    #cv2.imwrite("raw.png", depth)

    #cv2.imshow('raw', depth)
    #cv2.waitKey(0)
    np.warnings.filterwarnings('ignore')
    depth[depth < 800] = np.NaN
    depth[depth > 2800] = np.NaN

    # for each column the closest point
    obstacles  = np.nanmin(depth, axis=0)

    for index, i in enumerate(obstacles):
        if not np.isnan(i):
            cv2.circle(screen, (index, int(i/10)), 50, 100, -1)

    for index, i in enumerate(obstacles):
        if not np.isnan(i):
            screen[int(i/10), index] = 255

    #cv2.imshow('top obstacle view', screen)
    #cv2.waitKey(0)
    return screen



def startKinect():

    global depth_stream

    try:
        openni2.initialize("C:/Program Files (x86)/OpenNI2/Redist/")

        dev = openni2.Device.open_any()
        depth_stream = dev.create_depth_stream()
        depth_stream.start()
        depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))
        print("Kinect connected")
        return True
    except:
        print("startKinect failed")
        try:
            openni2.unload()
        except:
            pass
        return False


def stopKinect():

    openni2.unload()
    return True


if __name__ == '__main__':

    import threading

    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(depthCommands, port = DEPTH_PORT, protocol_config = rpyc.core.protocol.DEFAULT_CONFIG)
    t.start()


