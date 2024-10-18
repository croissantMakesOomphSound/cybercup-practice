# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:37:19 2024

@author: TMpub
"""

class iotsensorvalue:
    def __init__(self,iotSensorEntry=None,iotSensorExit=None):
        self.iotSensorEntry=iotSensorEntry
        self.iotSensorExit=iotSensorExit
    def iotentry(self):
        return self.iotSensorEntry
    def iotexit(self):
        return self.iotSensorExit
    
class gpsvalue:
    def __init__(self,gpsvalue=None):
        self.gpsvalue=gpsvalue
    def getgpsvalue(self):
        return self.gpsvalue
    
class mobileapps:
    def __init__(self,mobileapps=None):
        self.mobileapps=mobileapps
    def getmobilevalue(self):
        return self.mobileapps