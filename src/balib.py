import numpy as np
from theano import tensor as T
from collections import namedtuple
# from quaternion import Quaternion as quat
import time


Pose2D = namedtuple('Pose2D', ['x', 'y', 'theta'])
Pose3D = namedtuple('Pose2D', ['x', 'y', 'z', 'q1', 'q2', 'q3', 'q4'])
Landmark2D = namedtuple('Landmark2D', ['x', 'y'])
Landmark3D = namedtuple('Landmark3D', ['x', 'y', 'z'])

dim_lookup = {'R1': 1, 'R2': 2, 'R3': 3, 'SE2': 3, 'QUAT': 6}

constraint_lookup = {'SE2': 2, 'R1': 2, 'R2': 2, 'R3': 2}

def omega_dof(dim):
    """ calculate number of degrees of freedom (dof) of a (square
        inverse-covariance matrix with dimension `dim`
    """
    return (dim * (dim + 1) / 2)


def poseDiff2D(startingPose, endingPose):
    x1, y1, theta1 = endingPose[0], endingPose[1], endingPose[2]
    x2, y2, theta2 = startingPose[0], startingPose[1], startingPose[2]
    dx = (x1 - x2)*T.cos(theta2) + (y1 - y2)*T.sin(theta2)
    dy = -(x1 - x2)*T.sin(theta2) + (y1 - y2)*T.cos(theta2)
    dtheta = normalizeAngle(theta1 - theta2)
    return dx, dy, dtheta


def poseDiff3D(startingPose, endingPose):
    pass


def landmarkDiff2D(startingPose, landmark):
    pass


def landmarkDiff3D(startingPose, landmark):
    pass


def error(measurement, v1, v2, estimationFunction):
    estimate = estimationFunction(v1, v2)
    return estimationFunction(measurement, estimate)


def rotationMatrix2D(theta, module=np):
    costheta = module.cos(theta)
    sintheta = module.sin(theta)
    return np.array([[costheta, -sintheta],
                     [sintheta, costheta]])


def omega(vertex):
    return np.array([[vertex.omega_0_0, vertex.omega_0_1, vertex]])


def rotationMatrix3D(phi, theta, xi, module=np):
    cosphi = module.cos(phi)
    sinphi = module.sin(phi)
    costheta = module.cos(theta)
    sintheta = module.sin(theta)
    cosxi = module.cos(xi)
    sinxi = module.sin(xi)
    A_x = np.array([[1, 0, 0],
                    [0,  cosphi, -sinphi],
                    [0, sinphi, cosphi]])

    A_y = np.array([[costheta, 0, sintheta],
                    [0, 1, 0],
                    [-sintheta, 0, costheta]])

    A_z = np.array([[cosxi, -sinxi, 0],
                    [sinxi, cosxi, 0],
                    [0, 0, 1]])

    return module.dot(A_x, module.dot(A_y, A_z))


def eulerRotation(vector, phi, theta, xi, module=np):
    R = rotationMatrix3D(phi, theta, xi, module)
    return module.dot(R, vector)


def rodriguesMatrix(unit_axis, theta, module=np):
    e_x = np.array([[0, -unit_axis[2], unit_axis[1]],
                    [unit_axis[2], 0, unit_axis[0]],
                    [-unit_axis[1], unit_axis[0], 0]])

    return (module.eye(3) * module.cos(theta) +
           (1 - module.cos(theta))*unit_axis.dot(unit_axis.T) +
            e_x * module.sin(theta))


def normalizeAngle(theta):
    if T.gt(theta, -np.pi) and T.lt(theta, np.pi):
        return theta
    else:
        twopi = 2*np.pi
        mult = np.floor(theta / twopi)
        theta -= mult * twopi
        if T.ge(theta, np.pi):
            theta -= twopi
        elif T.le(theta, -np.pi):
            theta += twopi
    return theta


def identity_map(num, dim):
    return np.asarray([i/dim for i in range(dim*num)], dtype=np.uint32)


class SE2(object):
    def __init__(self, x, y, theta):
        self.t = np.array([x, y])
        self.y = y
        self.theta = theta

    @property
    def R(self):
        return rotationMatrix2D(self.theta, np)

    def __mul__(self, other):
        new_t = self.t + self.R.dot(other.t)
        new_theta = normalizeAngle(self.theta + other.theta)
        return SE2(new_t[0], new_t[1], new_theta)

    def __repr__(self):
        return '%f %f %f' % (self.t[0], self.t[1], self.theta)

# written for testing purposes
if __name__ == '__main__':
    start = SE2(2, -2, -3 * np.pi / 8)
    other = SE2(1, 1, np.pi/2)
    print start*other

    t0 = time.time()
    for _ in range(1000):
        rotationMatrix3D(0, 0, 0)
    print (time.time() - t0) / 1000

    if True:
        from subprocess import Popen, PIPE, STDOUT
        pipe = Popen('swig -version', shell=True, cwd=None, env=None, stdout=PIPE, stderr=STDOUT)

        (output, errout) = pipe.communicate(input=input)
        assert not errout

        status = pipe.returncode

        print status
        print output

