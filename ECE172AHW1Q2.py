import numpy as np
import matplotlib.pyplot as plt

# Set room size
vSize = 9 
hSize = 9

# Initialize position and object locations, then display.
loc = np.array([[0, 6]])


def haveIBeenHereBefore(loc, nextStep):
    ''' 
    This function can be used to determine if the robot has previously been 
    to the location specified in nextStep.
    loc is the set of previous locations traversed by the robot
    nextStep is the new location for which the test is to be performed
    out is a boolean value, True if nextStep has been previosuly
    visited
    '''
    return nextStep.tolist() in loc.tolist()

def displayRoom(loc,obj,vSize,hSize):
    # Create empty room
    room = np.zeros((vSize,hSize))
    
    # Represent objects with gray
    for ob in obj:
        room[ob[0],ob[1]] = 127
    
    # Represent past locations with light gray
    for lo in loc[:-1]:
        room[lo[0], lo[1]] = 191
    
    # Represent current location with white
    room[loc[-1][0], loc[-1][1]] = 255
    

    plt.imshow(room, cmap='gray')
    plt.title('Press \'q\' to continue. Ctrl+C (or Cmd+C) to stop simulation.')
    plt.show()

def detectObject(loc, obj, dir):
    # Check for object in specified direction
    if dir == 'N':
        dirLoc = loc[-1] + np.array([-1, 0])
    elif dir == 'E':
        dirLoc = loc[-1] + np.array([0, 1])
    elif dir == 'S':
        dirLoc = loc[-1] + np.array([1, 0])
    else:
        dirLoc = loc[-1] + np.array([0, -1])
    objectDetected = dirLoc.tolist() in obj.tolist()
    return objectDetected


# Add additional obstacles to this array per instructions:
obj = np.array([[0, 3], [2, 7], [3, 8], [5, 5], [2, 0], [1, 2], [7, 1], [3,6], [4,4], [4,6]]) 

displayRoom(loc, obj, vSize, hSize)

while(True):
    # Make the robot move a certain direction
    nextStep = loc[-1] + np.array([1, 0])
    
    # If there is an object to the South, move a different direction
    # START
    if haveIBeenHereBefore(loc, nextStep):
        if detectObject(loc, obj, 'S'):
            nextStep = loc[-1] + np.array([0, -1])
            if detectObject(loc, obj, 'W'):
                nextStep = loc[-1] + np.array([0, 1])
                if detectObject(loc, obj, 'E'):
                    nextStep = loc[-1] + np.array([-1,0])
        nextStep = loc[-1] + np.array([0,-1])

    if detectObject(loc, obj, 'S'):
        nextStep = loc[-1] + np.array([0, -1])
        if detectObject(loc, obj, 'W'):
            nextStep = loc[-1] + np.array([0, 1])
            if detectObject(loc, obj, 'E'):
                nextStep = loc[-1] + np.array([-1,0])
    # STOP
    
    # Update location if no object is in the way and within bounds
    if (nextStep.tolist() not in obj.tolist()) and nextStep[0] >= 0 and nextStep[0] <= (vSize - 1) and nextStep[1] >= 0:
        loc = np.vstack([loc, nextStep])
    
    # Show new position
    displayRoom(loc, obj, vSize, hSize)
    
    # Check if the South side of the image has been reached
    if loc[-1][0] == vSize-1:
        print("Success!")
        break