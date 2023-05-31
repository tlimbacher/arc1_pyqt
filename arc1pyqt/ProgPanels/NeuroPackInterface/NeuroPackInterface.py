import base64
import importlib
import pathlib
import sys
from importlib import import_module
from math import prod
import tensorflow
from PyQt5 import QtGui, QtCore, QtWidgets
import os
import numpy as np
import json
import subprocess
import pickle

import arc1pyqt.Globals.fonts as fonts
import arc1pyqt.Globals.styles as styles
from arc1pyqt.VirtualArC import VirtualArC
from arc1pyqt.VirtualArC import pulse as VirtualArCPulse
from arc1pyqt.VirtualArC import read as VirtualArCRead
from arc1pyqt.VirtualArC.parametric_device import ParametricDevice as memristor
from arc1pyqt import state
from arc1pyqt import modutils
from arc1pyqt.modutils import BaseThreadWrapper, BaseProgPanel, makeDeviceList, ModTag
from arc1pyqt.ProgPanels.NeuroPackInterface.memristorPulses import memristorPulses as memristorPulses

HW = state.hardware
APP = state.app
CB = state.crossbar
THIS_DIR = os.path.dirname(__file__)


class MemristorInterface:

    def __init__(self, params, mod):
        self.Ap = params.pop("Ap")
        self.An = params.pop("An")
        self.a0p = params.pop("a0p")
        self.a0n = params.pop("a0n")
        self.a1p = params.pop("a1p")
        self.a1n = params.pop("a1n")
        self.tp = params.pop("tp")
        self.tn = params.pop("tn")
        self.dt = params.pop("dt")
        self.pos_voltOfPulseList = params.pop("positive_pulses")
        self.pos_pulsewidthOfPulseList = params.pop("positive_pulse_times")
        self.pos_pulseList = list(zip(self.pos_voltOfPulseList, self.pos_pulsewidthOfPulseList))
        self.neg_voltOfPulseList = params.pop("negative_pulses")
        self.neg_pulsewidthOfPulseList = params.pop("negative_pulse_times")
        self.neg_pulseList = list(zip(self.neg_voltOfPulseList, self.neg_pulsewidthOfPulseList))
        self.RTolerance = params.pop("r_tolerance")
        self.rfloor = params.pop("min_resistance")
        self.rceil = params.pop("max_resistance")
        self.maxSteps = params.pop("max_update_steps")
        self.size = params.pop("size")

        self.Vread = HW.conf.Vread
        self.last_bitline = 0
        self.last_wordline = 0
        self.initR = 11000
        self.mod = mod

        print('Ap:%f, An:%f, a0p:%f, a0n:%f, a1p:%f, a1n:%f, tp:%f, tn:%f' % (
            self.Ap, self.An, self.a0p, self.a0n, self.a1p, self.a1n, self.tp, self.tn))

    def crossbar_init(self):
        if not isinstance(HW.ArC, VirtualArC):
            return
        HW.ArC.crossbar = [[] for _ in range(self.size + 1)]
        for w in range(self.size + 1):
            for b in range(self.size + 1):
                mx = memristor(Ap=self.Ap, An=self.An, tp=self.tp, tn=self.tn, a0p=self.a0p, a0n=self.a0n, a1p=self.a1p,
                               a1n=self.a1n)
                mx.initialise(self.initR)
                HW.ArC.crossbar[w].append(mx)

    def run(self):
        self.crossbar_init()
        self.mod.main(self)
        return

    def read(self, w, b):
        # ArC uses 1 indexing.
        return VirtualArCRead(HW.ArC.crossbar, w+1, b+1)

    def pulse(self, w, b, A, pw):
        # ArC uses 1 indexing.
        VirtualArCPulse(HW.ArC.crossbar, w+1, b+1, A, pw, self.dt)
        return VirtualArCRead(HW.ArC.crossbar, w+1, b+1)

    def request_tensor(self, shape: tuple[int]):
        """
        Allocates a set of memristors in order to fit the provided tensor shape. Then returns an object of type
        NeuroPackTensor that can be used to set the memristor resistances.
        Refer to NeuroPackTensor for further information.
        @param shape: The shape of the needed tensor
        """
        total_size = prod(shape)
        remaining_line = self.size - self.last_bitline
        remaining_bottom = (self.size - self.last_wordline) * self.size
        if total_size > remaining_bottom + remaining_line:
            raise Exception("Not enough space remaining in memristor crossbar.")
        old_wordline = self.last_wordline
        old_bitline = self.last_bitline
        if total_size < remaining_line:
            self.last_bitline += total_size
        else:
            self.last_bitline = (total_size - remaining_line) % self.size
            self.last_wordline += ((total_size - remaining_line) // self.size) + 1

        return NeuroPackTensor(self, shape, old_wordline, old_bitline)


class NeuroPackTensor:
    """
    The class that represents an array of memristors that acts as a tensor.
    """

    def __init__(self, memristor_interface, shape: tuple[int], wordline, bitline):
        self.shape = shape
        self.wordline = wordline
        self.bitline = bitline
        self.memristor_interface = memristor_interface
        self.max_weight = 0
        self.types = np.ones(shape)
        self.last_update_errors = np.full(shape, 0, np.dtype((np.float, 3)))
        self.used_mapping = False

    def de_normalise_resistance(self, w):
        if not self.used_mapping:
            return w
        PCEIL = 1.0 / self.memristor_interface.rfloor
        PFLOOR = 1.0 / self.memristor_interface.rceil

        C = (w / (self.max_weight + 0.2)) * (PCEIL - PFLOOR) + PFLOOR
        R = 1 / C
        return int(R)

    def normalise_weight(self, w):
        if not self.used_mapping:
            return w
        PCEIL = 1.0 / self.memristor_interface.rfloor
        PFLOOR = 1.0 / self.memristor_interface.rceil

        val = ((float(w) - PFLOOR) / (PCEIL - PFLOOR)) * (self.max_weight + 0.2)

        if val > self.max_weight + 0.2:
            val = self.max_weight + 0.2

        return val

    def update(self, d_W):
        """
        Updates the values of the weights according to the provided update matrix d_W. If use mapping was True when
        loading the tensor, the updates are not interpreted as resistance changes, but as weight changes.
        The calculation of the appropriate resistance change is made internally, according to the same linear mapping
        established when first loading the tensor. If use mapping was False when loading the tensor the values in
        the tensor will be interpreted directly as resistance changes, expressed in Ω.
        @param d_W the weight change matrix, should have same shape as the tensor.
        """
        d_W = d_W[0] if d_W.shape[0] == 1 else d_W
        assert d_W.shape == self.shape
        w = self.wordline
        b = self.bitline - 1
        for i in range(prod(self.shape)):
            b += 1
            if b == self.memristor_interface.size:
                w += 1
                b = 0
            index = np.unravel_index(i, self.shape)
            dW = d_W[index] if self.types[index] == 1 else -d_W[index]
            R = self.memristor_interface.read(w, b)  # current R
            p = 1 / R  # new weights
            p_norm = self.normalise_weight(p)
            p_expect = dW + p_norm
            if p_expect < 0:
                p_expect = abs(p_expect)
                self.types[index] *= -1
            R_expect = self.de_normalise_resistance(p_expect)

            for step in range(self.memristor_interface.maxSteps):
                if abs(R - R_expect) / R_expect < self.memristor_interface.RTolerance:
                    break
                if R - R_expect > 0:  # resistance needs to be decreased
                    pulseList = self.memristor_interface.neg_pulseList
                else:  # resistance needs to be increased
                    pulseList = self.memristor_interface.pos_pulseList
                virtualMemristor = memristorPulses(self.memristor_interface.dt,
                                                   self.memristor_interface.Ap,
                                                   self.memristor_interface.An,
                                                   self.memristor_interface.a0p,
                                                   self.memristor_interface.a1p,
                                                   self.memristor_interface.a0n,
                                                   self.memristor_interface.a1n,
                                                   self.memristor_interface.tp,
                                                   self.memristor_interface.tn, R)
                pulseParams = virtualMemristor.BestPulseChoice(R_expect, pulseList)
                del virtualMemristor
                R = self.memristor_interface.pulse(w, b, pulseParams[0], pulseParams[1])

            R_real = self.memristor_interface.read(w, b)
            p_real = 1 / R_real
            self.last_update_errors[index] = (p_norm, abs(self.normalise_weight(p_real) - p_expect), dW)

    def ideal_update(self, d_W):
        """
        Same as update(d_W), only here the right resistance is set directly, without using the voltage pulses for the update.
        @param d_W the weight change matrix, should have same shape as the tensor.
        """
        d_W = d_W[0] if d_W.shape[0] == 1 else d_W
        assert d_W.shape == self.shape
        w = self.wordline
        b = self.bitline - 1
        for i in range(prod(self.shape)):
            b += 1
            if b == self.memristor_interface.size:
                w += 1
                b = 0
            index = np.unravel_index(i, self.shape)
            dW = d_W[index] if self.types[index] == 1 else -d_W[index]
            R = self.memristor_interface.read(w, b)  # current R
            p = 1 / R  # new weights
            p_norm = self.normalise_weight(p)
            p_expect = dW + p_norm
            if p_expect < 0:
                p_expect = abs(p_expect)
                self.types[index] *= -1
            R_expect = self.de_normalise_resistance(p_expect)
            HW.ArC.crossbar[w + 1][b + 1].initialise(R_expect)

    def load(self, tensorflow_tensor, use_mapping=True):
        """
        @param tensorflow_tensor the tensor containing the weights to be loaded.
        @param use_mapping a boolean value indicating if a linear mapping should be established between weights
        and resistances or if the values in the tensor should be used as resistances directly.
        """
        w = self.wordline
        b = self.bitline - 1
        self.used_mapping = use_mapping
        self.max_weight = float(tensorflow.reduce_max(tensorflow.abs(tensorflow_tensor))) if use_mapping else 0.0
        tensorflow_tensor = tensorflow_tensor.numpy()
        for i in range(prod(self.shape)):
            b += 1
            if b == self.memristor_interface.size:
                w += 1
                b = 0
            index = np.unravel_index(i, self.shape)
            self.types[index] = 1 if tensorflow_tensor[index] > 0 else -1
            weight = abs(float(tensorflow_tensor[index]))
            HW.ArC.crossbar[w + 1][b + 1].initialise(self.de_normalise_resistance(weight))

    def get_tensor(self) -> tensorflow.Tensor:
        """
        @return a Tensorflow tensor containing the stored weights.
        The weights are restored from the resistances of the underlying memristors using the same linear mapping that
        was used when loading the tensor, thus this function may only be called after load(tensorflow_tensor).
        """
        tensor = np.zeros(self.shape)
        w = self.wordline
        b = self.bitline - 1
        for i in range(prod(self.shape)):
            b += 1
            if b == self.memristor_interface.size:
                w += 1
                b = 0
            index = np.unravel_index(i, self.shape)
            tensor[index] = self.types[index] * self.normalise_weight(1.0 / self.memristor_interface.read(w, b))
        return tensorflow.convert_to_tensor(tensor, dtype=tensorflow.float32)


class NeuroPackInterface(BaseProgPanel):

    def __init__(self):
        super().__init__(title="NeuroPack Interface", description="Flexible neural nets")
        self.hboxProg = None
        self.gridLayout = None
        self.vW = None
        self.An = 0
        self.Ap = 0
        self.a0p = 0
        self.a0n = 0
        self.a1p = 0
        self.a1n = 0
        self.tp = 0
        self.tn = 0
        self.rightEdits = None
        self.labelCounter = None
        self._analysisWindow = None
        self.params = None
        self.mod = None
        self.path = "arc1pyqt/ProgPanels/NeuroPackInterface/config.json"
        self.main_path = "arc1pyqt/ProgPanels/robot_arm_l2l/main.py"
        self.push_load_base_conf = QtWidgets.QPushButton("Load memristor config")
        self.push_load_base_conf.clicked.connect(self.open_base_conf)
        self.base_conf_filename = QtWidgets.QLabel("Load memristor config")

        self.push_load_main = QtWidgets.QPushButton("Load main")
        self.push_load_main.clicked.connect(self.open_main)
        self.main_filename = QtWidgets.QLabel("Load main")

        self.initUI()
        self.file_name = "config.json"
        self.main_name = "main.py"
        self.load_base_conf()
        self.load_main()

    def initUI(self):
        vbox1 = QtWidgets.QVBoxLayout()
        titleLabel = QtWidgets.QLabel('NeuroPack Interface')
        titleLabel.setFont(fonts.font1)
        descriptionLabel = QtWidgets.QLabel('Flexible neural net application module.')
        descriptionLabel.setFont(fonts.font3)
        descriptionLabel.setWordWrap(True)

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.setColumnStretch(0, 1)

        gridLayout.addWidget(self.push_load_base_conf, 0, 1)
        gridLayout.addWidget(self.base_conf_filename, 0, 0)

        gridLayout.addWidget(self.push_load_main, 1, 1)
        gridLayout.addWidget(self.main_filename, 1, 0)

        vbox1.addWidget(titleLabel)
        vbox1.addWidget(descriptionLabel)

        self.vW = QtWidgets.QWidget()
        self.vW.setLayout(gridLayout)
        self.vW.setContentsMargins(0, 0, 0, 0)

        vbox1.addWidget(self.vW)

        push_train = QtWidgets.QPushButton('Run Simulator')
        push_train.setStyleSheet(styles.btnStyle)
        push_train.clicked.connect(self.runTrain)

        vbox1.addWidget(push_train)

        vbox1.addStretch()

        self.setLayout(vbox1)
        self.gridLayout = gridLayout

    def open_base_conf(self):
        open_path = QtCore.QFileInfo(QtWidgets.QFileDialog()
                                     .getOpenFileName(self, 'Open base configuration', THIS_DIR, filter="*.json")[0])
        self.load_base_conf(open_path.filePath(), open_path.fileName())

    def open_main(self):
        open_path = QtCore.QFileInfo(QtWidgets.QFileDialog()
                                     .getOpenFileName(self, 'Open base configuration', THIS_DIR, filter="*.py")[0])
        self.load_main(open_path.filePath(), open_path.fileName())

    def load_base_conf(self, new_path=None, new_name="config.json"):
        try:
            if new_path == "":
                return
            data = json.load(open(self.path if new_path is None else new_path))
            # TODO maybe some parameter type checking.
            for x in ["Ap", "An", "a0p", "a0n", "a1p", "a1n", "tp", "tn", "min_resistance", "max_resistance", "dt",
                      "positive_pulses", "positive_pulse_times", "negative_pulses", "negative_pulse_times",
                      "max_update_steps", "r_tolerance"]:
                if x not in data.keys():
                    self.showErrorMessage("Missing required parameter %s" % x)
                    return
            self.params = data
            if new_name is not None:
                self.file_name = new_name
            self.path = self.path if new_path is None else new_path
            self.base_conf_filename.setText(self.file_name)
        except Exception as exc:
            if self.params is not None:
                self.base_conf_filename.setText(self.file_name)
            else:
                self.base_conf_filename.setText("Select config file")
            print(exc)
            self.showErrorMessage("An error occurred.")

    def load_main(self, new_path=None, new_name="main.py"):
        try:
            if new_path == "":
                return
            new_path = self.main_path if new_path is None else new_path
            sys.path.append(str(pathlib.Path(new_path).parent))
            spec = importlib.util.spec_from_file_location("mod.neuropack", new_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["mod.neuropack"] = mod
            spec.loader.exec_module(mod)
            if not hasattr(mod, 'main'):
                self.showErrorMessage("Please provide a main function.")
                return

            self.mod = mod
            if new_name is not None:
                self.main_name = new_name
            self.main_path = self.main_path if new_path is None else new_path
            self.main_filename.setText(self.main_name)
        except Exception as exc:
            if self.main_path is not None:
                self.main_filename.setText(self.main_name)
            else:
                self.main_filename.setText("Select main")
            print(exc)
            self.showErrorMessage("An error occurred.")

    @staticmethod
    def showErrorMessage(text):
        errMessage = QtWidgets.QMessageBox()
        errMessage.setText(text)
        errMessage.setIcon(QtWidgets.QMessageBox.Critical)
        errMessage.setWindowTitle("Error")
        errMessage.exec_()

    def eventFilter(self, object_, event):
        if event.type() == QtCore.QEvent.Resize:
            self.vW.setFixedWidth(event.size().width() - object_.verticalScrollBar().width())
        return False

    def disableProgPanel(self, state):
        self.hboxProg.setEnabled(not state)

    def runTrain(self):
        if self.path is None or self.params is None:
            self.showErrorMessage("Please select a valid config file.")
            return
        if self.mod is None:
            self.showErrorMessage("Please select a valid main file.")
            return
        if HW.ArC is None:
            self.showErrorMessage("Please connect to virtual ArC first.")
            return

        self.load_base_conf(new_name=None)
        self.load_main(new_name=None)
        network = MemristorInterface(self.params, self.mod)
        network.run()


tags = {'top': modutils.ModTag("NN2", "NeuroPackInt", None)}
