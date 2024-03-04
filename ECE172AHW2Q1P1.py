import numpy as np
import matplotlib.pyplot as plt

initial_loc = np.array([0.0,0.0])
final_loc = np.array([100.0,100.0])
sigma = np.array([[50.0,0.0],[0.0,50.0]])
mu = np.array([[60.0, 50.0], [10.0, 40.0]])

def f(x, y):
    return ((final_loc[0]-x)**2 + (final_loc[1]-y)**2)/20000 + 10000*(1/(2*np.pi*np.linalg.det(sigma)))*np.exp(-.5*(np.matmul(np.array([x-mu[0,0], y-mu[0,1]], dtype=object),np.matmul(np.linalg.pinv(sigma), np.atleast_2d(np.array([x-mu[0,0], y-mu[0,1]], dtype=object)).T)))[0]) + 10000*(1/(2*np.pi*np.linalg.det(sigma)))*np.exp(-.5*(np.matmul(np.array([x-mu[1,0], y-mu[1,1]], dtype=object),np.matmul(np.linalg.pinv(sigma), np.array([x-mu[1,0], y-mu[1,1]], dtype=object)))))

x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
z = f(x[:,None], y[None,:])
z = np.rot90(np.fliplr(z))

#plt.contour(x,y,z)
#dx,dy = np.gradient(z)
#plt.quiver(x,y , dx ,dy)

plt.contour(x, y, z)
dy, dx = np.gradient(z)
plt.quiver(x, y, dx, dy)

plt.plot(initial_loc[0], initial_loc[1], 'b*', markersize=10) 
plt.plot(final_loc[0], final_loc[1], 'r*', markersize=10) 

alpha = 100
error = 0.0001
current_loc = initial_loc.astype(float)
gradient_descent = 1

while gradient_descent == 1:

            dx_new = -dx[int(current_loc[1]), int(current_loc[0])]
            dy_new = -dy[int(current_loc[1]), int(current_loc[0])]

            current_loc[0] = current_loc[0] + (alpha * dx_new)
            current_loc[1] = current_loc[1] + (alpha * dy_new)

            plt.plot(current_loc[0], current_loc[1], marker=".", color="r")

            if np.linalg.norm(np.array([dx_new, dy_new])) < error:
                break


plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, z, 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Contour')
plt.show()




'''
def y_function(x):
    return x ** 2

def f_derivative(x):
    return 2* x

x = np.arrange(-100,100,0.1)
y = y_function(x) 

current_pos = (50, y_function(90))

plt.plot(x,y) 
plt.scatter(current_pos[0], current_pos[1], color = "red")

learning_rate = 0.01

for _ in range(1000):
    new_x = current_pos[0] - learning_rate * f_derivative(current_pos[0])
    new_y = y_function(new_x)
    current_pos = (new_x, new_y)
    plt.pause(0.001)
    plt.clf()
        
'''