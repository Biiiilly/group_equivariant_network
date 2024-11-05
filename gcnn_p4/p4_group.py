'''Define p4-group'''
import numpy as np

def p4_generator(r, u, v):
    '''
    The function used to generate p4 group.

    :param r: an int representing rotation coefficient between 0 and 4.
    :param u: an int representing horizontal translation.
    :param v: an int representing vertical translation.

    :return g: an 3x3 numpy array representing the element in p4.
    '''

    g = np.zeros((3,3))
    angle = r*np.pi / 2

    g[:, 0] = [np.cos(angle), np.sin(angle), 0]
    g[:, 1] = [-np.sin(angle), np.cos(angle), 0]
    g[:, 2] = [u, v, 1]

    return g
