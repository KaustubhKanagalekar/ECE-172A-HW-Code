import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import time
import random
import heapq

# initialize parameters
height = 50; # map height
width = 50 # map width
num_bots = 20 # number of bots
max_itr = 2500 # maximum number of iterations

wall = 1 # value for a wall in the map matrix
mapped = .4 # value for a mapped area in map matrix
planned = .2 # value for a bot's planned path in map matrix
unmapped = 0 # value for a unmapped area in map matrix

# define structure to hold all information about the bots and map
bots = [{} for i in range(num_bots)]

# define the provided blueprint map
blueprint = np.pad(np.zeros((height-2,width-2)), (1,1), 'constant', constant_values = (wall, wall)) # a map provided to the bot network
blueprint[9:19,35] = wall
blueprint[9:19,44] = wall
blueprint[9,35:37] = wall
blueprint[9,42:44] = wall
blueprint[19,35:44] = wall

# define the bot's map of explored areas
explore_map = blueprint; # a map the bot network uses to keep track of where bots have visited

# initialize positions, dest, route, and exploration map
for i in range(num_bots):
    bots[i]['current_position'] = [math.floor(height/2), math.floor(width/2)] # current position of bot
    explore_map[bots[i]['current_position'][0], bots[i]['current_position'][1]] = mapped
    bots[i]['destination'] = [] # destination of bot, initialized as empty
    bots[i]['route'] = [] 

plt.imshow(explore_map, cmap='gray')
plt.title('Press \'q\' button to begin.')
plt.show()


def find_unexplored_areas(explore_map, unmapped_value):
    '''
    Write this function so that unexplored_areas is a Nx2 matrix
    where N is the number of locations the bots have not explored, and each
    row of unexplored_areas represents a location (row,col) in the map.
    We define unexplored areas as locations which have values UNMAPPED in
    explore_map. Locations with values PLANNED/MAPPED/WALL are not considered
    unexplored areas in this case. If there are no unexplored areas, then
    unexplored_areas should return empty [].
    '''
    find_unexplored_areas = []
    [row, col] = explore_map.shape
    for i in range(row):
        for j in range(col):
            if explore_map[i,j] == unmapped_value:
                find_unexplored_areas.append([i,j])

    find_unexplored_areas = np.array(find_unexplored_areas)
    return find_unexplored_areas
    

def get_new_dest(current_position, unexplored_areas):
    '''
    Write this function so that it will pick the closest unexplored area
    as the new destination dest. We will keep this function simple by
    ignoring any walls that may block our path to the new destination. Here
    we define "closest" using the euclidean distance measure,
    e.g. sqrt((x1-x2)^2 + (y1-y2)^2).
    '''
    distance = []
    dist2 = -1
    for area in unexplored_areas:
        dist = math.sqrt((current_position[0]-area[0])**2 + (current_position[1]-area[1])**2)
        distance.append(dist)
            
    closest_pt_index = distance.index(min(distance))
    closest_dist = unexplored_areas[closest_pt_index]

    return closest_dist


def update_explore_map(dest, route, explore_map, planned, unmapped):
    '''
    Write this function so that all the locations specified in route and dest
    are marked as PLANNED only if it was previous UNMAPPED in the explore_map
    variable.
    '''
    for itr in route:
        if explore_map[itr[0], itr[1]] == unmapped:
            explore_map[itr[0], itr[1]] = planned
    if explore_map[dest[0], dest[1]] == unmapped:
        explore_map[dest[0], dest[1]]= planned 

    return explore_map

def update_pos(curPos, route, dest, explore_map, mapped):
    '''
    Write this function so that:
    1) update curPos 1 step closer to the destination using route
    2) mark the new location of the bot as MAPPED
    3) if the new location of the bot is at the destination, set destination
       to be empty, i.e. dest = []
    4) update the route by removing the location that the bot was just
       updated to e.g. if route was inputted as an Nx2 matrix, it should
       output as a (N-1)x2 matrix.
    '''
    curPos = route[0]
    explore_map[curPos[0], curPos[1]] = mapped
    if (curPos == dest).all():
        dest = []
        
    route = np.array(route)
    route = np.delete(route,0,0)

    return curPos, route, dest, explore_map


def update_bot_info(curPos, dest, route, explore_map, botNum):
    bots[botNum]['current_position'] = curPos
    bots[botNum]['destination'] = dest
    bots[botNum]['route'] = route
    explore_map = explore_map

# heuristic function for path scoring

def heuristic(a, b):
    return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def a_star(array, start, goal):
    
    start = tuple(start)
    goal = tuple(goal)

    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}

    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
 
    while oheap:

        current = heapq.heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(list(current))
                current = came_from[current]
            return np.array(data)

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            tentative_g_score = gscore[current] + heuristic(current, neighbor)

            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
 
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:

                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
 
    return False

for itr in range(max_itr):
    for botNum in range(num_bots):
        # extract bot's info for convenience/code readability
        curPos = bots[botNum]['current_position']
        dest = bots[botNum]['destination']
        route = bots[botNum]['route']

        # if the bot doesn't have a destination then pick a new destination
        if len(dest) == 0: 
                    
            # get the locations of all areas that are labeled as unmapped 
            # TODO: write this function above.
            unexplored_areas = find_unexplored_areas(explore_map, unmapped)
            
            # if there are no more unexplored areas, then this bot stops moving
            if len(unexplored_areas) == 0:
                dest = []
                route = []
                update_bot_info(curPos, dest, route, explore_map, botNum)
                continue
            
            # calculate bot's new destination (TODO: write this function above)
            dest = get_new_dest(curPos, unexplored_areas)
            
            # calculates bot's route to the destination (you do not need to worry about this function)
            if a_star(explore_map, curPos, dest).size:
                route = a_star(explore_map, curPos, dest)
                route = np.vstack((route, curPos))
                route = route[::-1]

            
            # mark the location in explored_map as planned if it was unmapped (TODO: write this function above)
            explore_map = update_explore_map(dest, route, explore_map, planned, unmapped)

        # Using the calculated route, move the bot 1 step towards the destination bot's destination (TODO: write this function above)
        curPos, route, dest, explore_map = update_pos(curPos, route, dest, explore_map, mapped)
        
        # update bot's curr position, past position, destination, and map
        update_bot_info(curPos, dest, route, explore_map, botNum)

    # update display
    plt.imshow(explore_map, cmap='gray')
    for i in range(num_bots):
        fig = plt.gcf()
        ax = fig.gca()
        p = patches.Rectangle((bots[i]['current_position'][1]-.5, bots[i]['current_position'][0]-.5), 1, 1, edgecolor = 'blue', facecolor = 'red', fill=True)
        ax.add_patch(p)

        #ax.add_patch(patches.Rectangle((bots[i]['current_position'][1]-.5, bots[i]['current_position'][0]-.5), 1, 1, edgecolor = 'blue', facecolor = 'red', fill=True))
        if len(bots[i]['destination']) != 0:
            plt.plot([b[1] for b in bots[i]['route']], [b[0] for b in bots[i]['route']], color='red')
            plt.plot(bots[i]['destination'][1], bots[i]['destination'][0], color='yellow')

    plt.show(block=False)
    plt.pause(.25)
    #ax.patches.clear()
    p.remove()
    mapped_count = sum(sum(np.array(explore_map) == mapped))
    wall_count = sum(sum(np.array(explore_map) == wall))
    explore_size = explore_map.size
    if (mapped_count + wall_count) == explore_size:
        break

print("Number of iterations: "+str(itr))