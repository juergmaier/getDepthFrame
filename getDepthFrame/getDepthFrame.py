
from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import cv2
import rpyc

rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

DEPTH_PORT = 20100

KINECT_MIN_DISTANCE = 800
KINECT_MAX_DISTANCE = 4000
KINECT_VERTICAL_RANGE_DEG = 47  # kinect vertical view angle
KINECT_MOUNT_HEIGHT = 1100      # distance to floor
CEILING = 1750                  # obstacles above this height can be ignored
KINECT_VERTICAL_MOUNT_ANGLE_DEG = -3
DEPTH_ROWS = 480
DEPTH_COLS = 640

OFFSET_KINECT_FROM_CART_CENTER = 200    # Kinect from is not at cart rotation center, offset in mm

# Each row in the depth data represents a vertical angle range
topAngle = KINECT_VERTICAL_RANGE_DEG/2 + KINECT_VERTICAL_MOUNT_ANGLE_DEG
bottomAngle = KINECT_VERTICAL_RANGE_DEG/2 - KINECT_VERTICAL_MOUNT_ANGLE_DEG
RowDeg = KINECT_VERTICAL_RANGE_DEG / DEPTH_ROWS
RowRad = np.radians(RowDeg)

depth_stream = None

class depthCommands(rpyc.Service):
    '''
    list of all remote calls to this thread
    '''

    def exposed_echo(self, text):
        return text + " from getDepthFrame"

    def exposed_activateKinect(self):
        return startKinect()

    def exposed_deactivateKinect(self):
        return stopKinect()

    def exposed_getDepth(self, orientation):
        return obstacleMap(orientation)


def removeUpperRange(depth):
    '''
    based on mount height and mount angle of kinect and max head room needed for the robot
    calculate for each row of the depth array the "ceiling distance"
    remove points exceeding this distance (irrelevant for movements)
    '''
    # finally for each row (angle) the distance to 1800 mm
    distanceToCeiling = np.zeros(DEPTH_ROWS)

    # Angles of ceiling limits for distance calculation
    topRoom = CEILING - KINECT_MOUNT_HEIGHT
    ceilingAngleClosest = KINECT_VERTICAL_RANGE_DEG / 2 + KINECT_VERTICAL_MOUNT_ANGLE_DEG
    ceilingAngleFarthest = np.degrees(np.arctan(topRoom / KINECT_MAX_DISTANCE))

    # a row is a quantized vertical angle, each row has RowRad height
    ceilingStartRow = 0
    ceilingEndRow = int((ceilingAngleClosest - ceilingAngleFarthest) / RowDeg)
    #print(f"ceilingAngleClosest: {round(ceilingAngleClosest)}, ceilingStartRow: {ceilingStartRow}, ceilingAngleFarthest: {round(ceilingAngleFarthest)}, ceilingEndRow: {ceilingEndRow}")

    # for each row the distance to the ceiling, distance values greater than this can be removed
    rad = np.radians(ceilingAngleClosest)
    for row in range(ceilingStartRow, ceilingEndRow):
        distanceToCeiling[row] = (topRoom / np.tan(rad))
        #print(f"row: {row} deg: {round(np.degrees(rad))}, distanceToCeiling: {distanceToCeiling[row]}")

        rad -= RowRad

    # remove points above 1800 mm (ceiling)
    for row in range(ceilingStartRow, ceilingEndRow):
        if distanceToCeiling[row] > 0:
            for col in range(0, DEPTH_COLS):
                if not np.isnan(depth[row, col]):
                    #print(f"row: {row}, col: {col}, depth[row,col]: {depth[row,col]}, distanceToFloor[row]: {distanceToFloor[row]}")
                    if depth[row, col] >= distanceToCeiling[row]:
                        depth[row, col] = np.NaN
                    #print(f"depth[row,col]: {depth[row,col]}")


def removeFloor(depth):
    '''
    based on mount height and mount angle of kinect provide the floor distance for each row of the depth map
    filter out points in this distance range (abyss will become an obstacle)
    '''
    # for each row of the depth map the distance to the floor
    distanceToFloor = np.zeros(DEPTH_ROWS)

    # angles of floor limits for distance calculation
    floorAngleFarthest = np.degrees(np.arctan(KINECT_MOUNT_HEIGHT / KINECT_MAX_DISTANCE)) + KINECT_VERTICAL_MOUNT_ANGLE_DEG

    # set start and end row where the floor will show up
    floorStartRow = int(floorAngleFarthest / RowDeg + DEPTH_ROWS/2)
    floorEndRow = DEPTH_ROWS

    #print(f"floorAngleClosest: {round(floorAngleClosest)}, floorStartRow: {floorStartRow}, floorAngleFarthest: {round(floorAngleFarthest)}, floorEndRow: {floorEndRow}")

    rad = np.radians(floorAngleFarthest - KINECT_VERTICAL_MOUNT_ANGLE_DEG)
    # for each of these rows calc the distance to the floor
    for row in range(floorStartRow, floorEndRow):
        distanceToFloor[row] = KINECT_MOUNT_HEIGHT / np.tan(rad)
        #print(f"row: {row} deg: {round(np.degrees(rad))}, distanceToFloor: {distanceToFloor[row]}")

        rad += RowRad

    # remove floor
    for row in range(floorStartRow, floorEndRow):
        if distanceToFloor[row] > 0:
            for col in range(0, DEPTH_COLS):
                if not np.isnan(depth[row, col]):
                    # for the floor do not remove measurements far below the floor, it could be downward stairs
                    #print(f"row: {row}, col: {col}, depth[row,col]: {depth[row,col]}, distanceToFloor[row]: {distanceToFloor[row]}")
                    if np.abs(depth[row, col] - distanceToFloor[row]) < depth[row, col] * 0.3:
                        depth[row, col] = np.NaN
                    #print(f"depth[row,col]: {depth[row,col]}")


def obstacleMap(orientation):
    '''
    take a depth picture from the kinect
    floor and ceiling are not considered obstacles
    for each column take the closest point to create a top view of obstacles
    enlarge obstacle points with circles representing the cart size
    return the top viwe as a bw image
    '''
    screen = np.zeros((480, 640, 1), np.uint8)

    frame = depth_stream.read_frame()
    frame_data = frame.get_buffer_as_uint16()
    arr1d = np.frombuffer(frame_data, dtype=np.int16)

    depth = arr1d.astype(float)
    depth.shape = (480, 640)
    #depth = cv2.flip(depth, 1)
    
    np.save(f"depth_{orientation}.npy", depth)

    #cv2.imwrite("raw.png", depth)

    #cv2.imshow('raw', depth)
    #cv2.waitKey(0)
    # this might create NaN rows, ignore the warning output
    np.warnings.filterwarnings('ignore')
    depth[depth < KINECT_MIN_DISTANCE] = np.NaN
    depth[depth > KINECT_MAX_DISTANCE] = np.NaN

    removeUpperRange(depth)
    removeFloor(depth)

    # for each column the closest point
    obstacles = np.nanmin(depth, axis=0)

    # enlarge obstacles to account for vehicle size
    for index, i in enumerate(obstacles):
        if not np.isnan(i):
            cv2.circle(screen, (index, int(i/10)), 50, 100, -1)

#    for index, i in enumerate(obstacles):
#        if not np.isnan(i):
#            screen[int((i + OFFSET_KINECT_FROM_CART_CENTER) / 10), index] = 255

    #cv2.imwrite('top obstacle view', screen)
    #cv2.waitKey(0)
    return screen



def startKinect():
    '''
    start the kinect for taking depth images
    '''
    global depth_stream

    try:
        openni2.initialize("C:/Program Files (x86)/OpenNI2/Redist/")

        dev = openni2.Device.open_any()
        depth_stream = dev.create_depth_stream()
        depth_stream.start()
        depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=640, resolutionY=480, fps=30))
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
    '''
    turn off kinect
    '''
    openni2.unload()
    return True


if __name__ == '__main__':

    #startKinect()
    #obstacleMap(0)

    # start the listener for remote calls
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(depthCommands, port=DEPTH_PORT, protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)
    t.start()
