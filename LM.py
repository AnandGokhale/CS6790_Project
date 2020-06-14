
#From the previous module, I want a graph in the form of an adjacency matrix, that defines which egdes are in the clique
import numpy as np

class Feature:

    def __init__(self,j,s,w):
        self.j = j #Feature image coordinate
        self.s = s #Feature descriptor
        self.w = w #Feature world coordinate

class FeatureSet:

    def __init__(self,adjW,features,P):
        self.adjW = adjW
        self.features = features
        self.numFeatures = len(features)
        self.P = P
        self.weights = []
        for i in range(self.numFeatures):
            self.weights.append(self.initialize_weights())

    def generateRT(self,RTParams):

        Rx,Ry,Rz,Tx,Ty,Tz = RTParams
        #Reference Wikipedia https://en.wikipedia.org/wiki/Rotation_matrix
        #Rz == ALPHA
        #Ry == BETA
        #Rx == GAMMA
        
        ca = np.cos(Rz)
        sa = np.sin(Rz)
        cb = np.cos(Ry)
        sb = np.sin(Ry)
        cg = np.cos(Rx)
        sg = np.sin(Rx)

        RT = np.asarray(
            [
                [ca*cb, (ca*sb*sg) - (sa*cg),(ca*sb*cg) + (sa*sg),Tx],
                [sa*cb, (sa*sb*sg) + (ca*cg),(sa*sb*cg) - (ca*sg),Ty],
                [-1*sb, cb *sg , cb*cg,Tz],
                [0,0,0,1]
            ]
        )
        return RT
    
    def initialize_weights(self):
        Rx = 2*np.pi* np.random.rand(1)
        Ry = 2*np.pi* np.random.rand(1)
        Rz = 2*np.pi* np.random.rand(1)

        Tx = 0
        Ty = 0
        Tz = 0

        return [Rx,Ry,Rz,Tx,Ty,Tz]


    def invRT(self,RTParams):
        Rx,Ry,Rz,Tx,Ty,Tz = RTParams

        return [-1*Rx,-1*Ry,-1*Rz,-1*Tx,-1*Ty,-1*Tz]

    def loss(self):
        error = 0

        for a in range(self.numFeatures):
            for b in range(self.numFeatures):
                error += self.adjW[a,b] * np.square(  \
                                        self.features[a].j -  \
                                        self.P @   \
                                        (self.generateRT(self.weights[a]) @ \
                                        self.generateRT(self.invRT(self.weights[b])))@  \
                                        self.features[b].w \
                                        )

        return error

#Steps Ahead
'''
Write a function for calculating the gradient of the loss 
Write a function to update the parameters based on the gradient
Write a function to loop over this, and remove outliers
'''