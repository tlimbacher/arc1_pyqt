####################################

# (c) Radu Berdan
# ArC Instruments Ltd.

# This code is licensed under GNU v3 license (see LICENSE.txt for details)

####################################

from . import GlobalVars as g
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtGui, QtWidgets
from time import sleep
import numpy as np
import collections
import struct
from virtualArC import virtualarc


###########################################
# Signal Routers
###########################################
# Class which routes signals (socket) where instantiated
class cbAntenna(QObject):
    # Used for highlighting/deselecting devices on the crossbar panel
    selectDeviceSignal=pyqtSignal(int, int)
    deselectOld=pyqtSignal()
    redraw=pyqtSignal()
    reformat=pyqtSignal()
    # and signals updating the current select device
    recolor=pyqtSignal(float, int, int)

    def __init__(self):
        super(cbAntenna,self).__init__()

    def cast(self,w,b):
        self.selectDeviceSignal.emit(w,b)

cbAntenna=cbAntenna()


class displayUpdate(QObject):
    updateSignal_short=pyqtSignal()
    updateSignal=pyqtSignal(int, int, int,  int,    int)
                        #   w,   b,   type, points, slider
    updateLog=pyqtSignal(int)

    def __init__(self):
        super(displayUpdate,self).__init__()

    def cast(self):
        self.updateSignal_short.emit()

displayUpdate=displayUpdate()


# Class which routes signals (socket) where instantiated
class historyTreeAntenna(QObject):
    # used for signaling the device history tree list to update its contents
    updateTree=pyqtSignal(int,int)
    updateTree_short=pyqtSignal()
    clearTree=pyqtSignal()
    changeSessionName=pyqtSignal()

    def __init__(self):
        super(historyTreeAntenna,self).__init__()

historyTreeAntenna=historyTreeAntenna()


class interfaceAntenna(QObject):
    disable=pyqtSignal(bool)
    reformat=pyqtSignal()
    changeArcStatus=pyqtSignal(str)
    changeSessionMode=pyqtSignal(str)
    updateHW=pyqtSignal()
    lastDisplaySignal=pyqtSignal()

    globalDisable=False

    def __init__(self):
        super(interfaceAntenna,self).__init__()

    def wakeUp(self):
        g.waitCondition.wakeAll()
    def cast(self, value):
        # if value==False:
        #   g.waitCondition.wakeAll()
        if self.globalDisable==False:
            self.disable.emit(value)
            #sleep(0.1)
            self.lastDisplaySignal.emit()


    def castArcStatus(self, value):
        if self.globalDisable==False:
            self.changeArcStatus.emit(value)

    def toggleGlobalDisable(self, value):
        self.globalDisable=value
interfaceAntenna=interfaceAntenna()

# updates the range of devices for each pulsing script thread
######################################

###########################################


class SAantenna(QObject):
    disable=pyqtSignal(int, int)
    enable=pyqtSignal(int, int)

    def __init__(self):
        super(SAantenna,self).__init__()

SAantenna=SAantenna()


###########################################
# Update history function
###########################################
def updateHistory(w,b,m,a,pw,tag):
    readTag='R'+str(g.readOption)
    if g.sessionMode==1:
        g.Mnow=m/2
    else:
        g.Mnow=m
    g.Mhistory[w][b].append([g.Mnow,a,pw,tag,readTag,g.Vread])

    g.w=w
    g.b=b
    cbAntenna.recolor.emit(m,w,b)


def updateHistory_CT(w,b,m,a,pw,tag):
    readTag='R2'

    if g.sessionMode==1:
        g.Mnow=m/2
    else:
        g.Mnow=m

    g.Mhistory[w][b].append([g.Mnow,a,pw,tag,readTag,a])

    g.w=w
    g.b=b
    cbAntenna.recolor.emit(m,w,b)


def updateHistory_short(m,a,pw,tag):
    readTag='R'+str(g.readOption)
    if g.sessionMode==1:
        g.Mnow=m/2
    else:
        g.Mnow=m
    g.Mhistory[g.w][g.b].append([g.Mnow,a,pw,tag,readTag, g.Vread])
    cbAntenna.recolor.emit(m,g.w,g.b)


def writeDelimitedData(data, dest, delimiter="\t"):
    try:
        f = open(dest, 'w')
        for line in data:
            if isinstance(line, str) or (not isinstance(line, collections.Iterable)):
                line = [line]
            text = delimiter.join("{0:.5g}".format(x) for x in line)
            f.write(text+"\n")
        f.close()
    except Exception as exc:
        print(exc)

def saveFuncToFilename(func, title="", parent=None):
    fname = QtWidgets.QFileDialog.getSaveFileName(parent, title)

    if fname:
        func(fname)

def gzipFileSize(fname):
    with open(fname, 'rb') as f:
        # gzip uncompressed file size is stored in the
        # last 4 bytes of the file. This will roll over
        # for files > 4 GB
        f.seek(-4,2)
        return struct.unpack('I', f.read(4))[0]


###########################################
# Update Hover panel
###########################################

class hoverAntenna(QObject):
    displayHoverPanel=pyqtSignal(int, int, int, int, int, int)
    hideHoverPanel=pyqtSignal()

    def __init__(self):
        super(hoverAntenna,self).__init__()

hoverAntenna=hoverAntenna()


class addressAntenna(QObject):
    def __init__(self):
        super(addressAntenna,self).__init__()

    def update(self, w,b):
        g.w,g.b=w,b
        cbAntenna.selectDeviceSignal.emit(w, b)

addressAntenna=addressAntenna()


def getFloats(n):
    if not isinstance(g.ser, virtualarc.virtualArC):
        while g.ser.inWaiting()<n*4:
            pass
        # read n * 4 bits of data (n floats) from the input serial
        values=g.ser.read(size=n*4)
        buf = memoryview(values)
        extracted=np.frombuffer(buf, dtype=np.float32)
    else:
        extracted = g.ser.read(n)
    # returns a list of these floats
    return extracted

