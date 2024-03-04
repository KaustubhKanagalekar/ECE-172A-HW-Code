import matplotlib.pyplot as plt
import numpy as np
import pickle

def draw_path(final_path_points, other_path_points):
    '''
    final_path_points: the list of points (as tuples or lists) comprising your final maze path.
    other_path_points: the list of points (as tuples or lists) comprising all other explored maze points.
    (0,0) is the start, and (49,49) is the goal.
    Note: the maze template must be in the same folder as this script.
    '''
    im = plt.imread('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/172maze2021.png')
    x_interval = (686-133)/49
    y_interval = (671-122)/49
    plt.imshow(im)
    fig = plt.gcf()
    ax = fig.gca()
    circle_start = plt.Circle((133,800-122), radius=4, color='lime')
    circle_end = plt.Circle((686, 800-671), radius=4, color='red')
    ax.add_patch(circle_start)
    ax.add_patch(circle_end)
    for point in other_path_points:
        if not (point[0]==0 and point[1]==0) and not (point[0]==49 and point[1]==49):
            circle_temp = plt.Circle((133+point[0]*x_interval, 800-(122+point[1]*y_interval)), radius=4, color='blue')
            ax.add_patch(circle_temp)
    for point in final_path_points:
        if not (point[0]==0 and point[1]==0) and not (point[0]==49 and point[1]==49):
            circle_temp = plt.Circle((133+point[0]*x_interval, 800-(122+point[1]*y_interval)), radius=4, color='yellow')
            ax.add_patch(circle_temp)
    plt.show()

def deserialize_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

data = deserialize_pickle('/Users/kaustubhkanagalekar/Desktop/Kaustubh Python/HW Code/172maze2021.p')

def sense_maze(i, maze):
    explore = []
    x, y = i
    if maze[i][0] == True:
        explore.append((x, y+1))
    if maze[i][1] == True:
        explore.append((x+1, y))
    if maze[i][2] == True:
        explore.append((x, y-1))
    if maze[i][3] == True:
        explore.append((x-1, y))
    return explore

def DFS(maze):
    visited = np.zeros((50, 50))
    parent = np.zeros((50, 50, 2))

    start = list(maze.keys())[0]  
    stack = [start]

    path = {}
    while stack:  
        i = stack.pop() 
        x, y = i 

        if not visited[x][y]:
            visited[x][y] = 1
            neighbours = sense_maze(i, maze)  
            
            for neighbour in neighbours: 
                x_n, y_n = neighbour
                if not visited[x_n][y_n]:
                    stack.append(neighbour) 
                    parent[x_n][y_n] = [x, y]
                    path[neighbour]=i

    return_path={}
    end_goal=(49,49)
    while end_goal!=start:
        return_path[path[end_goal]]=end_goal
        end_goal=path[end_goal]
    return return_path

def BFS(maze):
    visited = np.zeros((50, 50))
    parent = np.zeros((50, 50, 2))

    start = list(maze.keys())[0]  
    queue = [start]

    path = {}
    while queue:  
        i = queue.pop(0) 
        x, y = i 

        if not visited[x][y]:
            visited[x][y] = 1
            neighbours = sense_maze(i, maze)  
            
            for neighbour in neighbours: 
                x_n, y_n = neighbour
                if not visited[x_n][y_n]:
                    queue.append(neighbour) 
                    parent[x_n][y_n] = [x, y]
                    path[neighbour]=i
    
    return_path={}
    end_goal=(49,49)
    while end_goal!=start:
        return_path[path[end_goal]]=end_goal
        end_goal=path[end_goal]
    return return_path, visited

final_path = DFS(data)
explored_points = list(set(data.keys()) - set(final_path))
draw_path(final_path, explored_points)

#final_path2 = BFS(data)
#explored_points2 = visited
draw_path(*BFS(data))
