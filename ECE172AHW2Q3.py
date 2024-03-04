import numpy as np
import matplotlib.pyplot as plt

import math 
def forwardKinematics(theta0, theta1, theta2, l0, l1, l2):
	J1x = l0*math.cos(theta0)
	J1y = l0*math.sin(theta0)
	J2x = J1x + l1*math.cos(theta0 + theta1)
	J2y = J1y + l1*math.sin(theta0 + theta1)
	JEx = J2x + l2*math.cos(theta0 + theta1 - theta2)
	JEy = J2y + l2*math.sin(theta0 + theta1 - theta2)

	return J1x, J1y, J2x, J2y, JEx, JEy



def Jacobians(theta0, theta1, theta2, l0,l1,l2):
#x_e = l0*math.cos(theta0_target) + l1*math.cos(theta0_target + theta1_target) + l2*math.cos(theta0_target + theta1_target - theta2_target)
#_e = l0*math.sin(theta0_target) + l1*math.sin(theta0_target + theta1_target) + l2*math.sin(theta0_target + theta1_target - theta2_target)


	Jacobian_Matrix = np.array([[-l0*math.sin(theta0) -l1*math.sin(theta0 + theta1) - l2*math.sin(theta0 + theta1 - theta2), 0 -l1*math.sin(theta0 + theta1) -l2*math.sin(theta0 + theta1 - theta2), 0 + 0 +l2*math.sin(theta0 + theta1 - theta2)],
					[l0*math.cos(theta0) + l1*math.cos(theta0 + theta1) + l2*math.cos(theta0 + theta1 + theta2), 0 + l1*math.cos(theta0 + theta1), 0 + 0 -l2*math.cos(theta0 + theta1 -theta2)]])
	return Jacobian_Matrix


def inverseKinematics(l0, l1, l2, x_e_target, y_e_target):
	'''
	This function is supposed to implement inverse kinematics for a robot arm
	with 3 links constrained to move in 2-D. The comments will walk you through
	the algorithm for the Jacobian Method for inverse kinematics.

	INPUTS:
	l0, l1, l2: lengths of the robot links
	x_e_target, y_e_target: Desired final position of the end effector 

	OUTPUTS:
	theta0_target, theta1_target, theta2_target: Joint angles of the robot that
	take the end effector to [x_e_target,y_e_target]
	'''
    # Initialize for the plots: 
	end_effector_positions = []
    # Initialize the thetas to some value 
	theta0_target, theta1_target, theta2_target = [math.pi/4, 0,0]
    # Obtain end effector position x_e, y_e for current thetas: 
	x_e = l0 * math.cos(theta0_target) + l1 * math.cos(theta0_target + theta1_target) + l2 * math.cos(
        theta0_target + theta1_target - theta2_target)
	y_e = l0 * math.sin(theta0_target) + l1 * math.sin(theta0_target + theta1_target) + l2 * math.sin(
        theta0_target + theta1_target - theta2_target)
	initial_pt = np.array([x_e, y_e])

    # HINT: use your ForwardKinematics function
	while math.sqrt((x_e - x_e_target) ** 2 + (y_e - y_e_target) ** 2) > 0.001:
        # Calculate the Jacobian matrix for current values of theta
		jacobian = Jacobians(theta0_target, theta1_target, theta2_target, l0, l1, l2)
        # Calculate the pseudo-inverse of the jacobian (HINT: numpy pinv())
		pseudo = np.linalg.pinv(jacobian)
        # Update the values of the thetas by a small step
		thetas_update = np.dot(pseudo, [x_e_target - x_e, y_e_target - y_e])
		theta0_target += 0.01 * thetas_update[0]
		theta1_target += 0.01 * thetas_update[1]
		theta2_target += 0.01 * thetas_update[2]
        # Obtain end effector position x_e, y_e for the updated thetas
		x_e = l0 * math.cos(theta0_target) + l1 * math.cos(theta0_target + theta1_target) + l2 * math.cos(
            theta0_target + theta1_target - theta2_target)
		y_e = l0 * math.sin(theta0_target) + l1 * math.sin(theta0_target + theta1_target) + l2 * math.sin(
            theta0_target + theta1_target - theta2_target)
		intermediate_step = forwardKinematics(theta0_target, theta1_target, theta2_target, l0, l1, l2)
		end_effector_positions.append([x_e, y_e])

    # Plot the final robot pose
	drawRobot(*intermediate_step)
    # Plot the end effector position through the iterations 
	return theta0_target, theta1_target, theta2_target, x_e, y_e


def drawRobot(x_1,y_1,x_2,y_2,x_e,y_e):
	x_0, y_0 = 0, 0
	plt.plot([x_0, x_1, x_2, x_e], [y_0, y_1, y_2, y_e], lw=4.5)
	plt.scatter([x_0, x_1, x_2, x_e], [y_0, y_1, y_2, y_e], color='r')
	plt.show()
	
#[J1x, J1y, J2x, J2y, JEx, JEy] = forwardKinematics(math.pi/4, math.pi/6, -math.pi/3, 2,4,7)
#drawRobot(J1x, J1y, J2x, J2y, JEx, JEy)
drawRobot(*forwardKinematics(math.pi/4, math.pi/6, -math.pi/3, 2,4,7))
drawRobot(*forwardKinematics(math.pi/4, math.pi/4, math.pi/4, 4,6,3))
inverseKinematics(8,8,8,5,10)
