# -*- coding: utf-8 -*-
import numpy as np
import math

class Integrator:
    def __init__(self, xMin, xMax, N):
        self.mini=xMin
        self.maxi=xMax
        self.n=N
            
    def integrate(self):       
        deltaX=(self.maxi-self.mini)/(self.n-1)
        i=np.arange(self.n)
        xi=self.mini+i*deltaX
        fxi=xi**2*np.exp(-xi)*np.sin(xi)
        fonction=fxi*deltaX
        self.somme = np.sum(fonction)
        
    def show(self):
        print(self.somme)

        

examp = Integrator(1,3,200)
examp.integrate()
examp.show()
