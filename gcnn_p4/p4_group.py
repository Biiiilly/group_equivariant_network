'''Define p4-group'''
import numpy as np

def p4_matrix_generator(r, u, v):
    '''
    The function used to generate p4 group matrix representation.

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

class P4_Group:

    def __init__(self, r, u, v):
        '''
        :param r: an int representing rotation coefficient between 0 and 4.
        :param u: an int representing horizontal translation.
        :param v: an int representing vertical translation.
        '''
        self.r = r
        self.u = u
        self.v = v

    def to_matrix(self):
        return p4_matrix_generator(self.r, self.u, self.v)

    def __mul__(self, other):
        if not isinstance(other, P4_Group):
            raise TypeError('Can only multiply with P4_Group elements')
        
        self_r, self_u, self_v = self.r, self.u, self.v
        other_r, other_u, other_v = other.r, other.u, other.v

        new_r = (self_r + other_r) % 4
        new_u = self_u + other_u
        new_v = self_v + other_v

        return P4_Group(new_r, new_u, new_v)
    
    def __repr__(self):
        return f"P4Element(rotation={self.r * 90} degrees, translation={(self.u, self.v)})"





