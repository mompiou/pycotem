# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tiltUI.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Tilt(object):
    def setupUi(self, Tilt):
        Tilt.setObjectName(_fromUtf8("Tilt"))
        Tilt.resize(396, 117)
        self.gridLayout = QtGui.QGridLayout(Tilt)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.theta_signBox = QtGui.QCheckBox(Tilt)
        self.theta_signBox.setText(_fromUtf8(""))
        self.theta_signBox.setObjectName(_fromUtf8("theta_signBox"))
        self.gridLayout.addWidget(self.theta_signBox, 1, 8, 1, 1)
        self.beta_label = QtGui.QLabel(Tilt)
        self.beta_label.setObjectName(_fromUtf8("beta_label"))
        self.gridLayout.addWidget(self.beta_label, 1, 3, 1, 1)
        self.alpha_signBox = QtGui.QCheckBox(Tilt)
        self.alpha_signBox.setText(_fromUtf8(""))
        self.alpha_signBox.setObjectName(_fromUtf8("alpha_signBox"))
        self.gridLayout.addWidget(self.alpha_signBox, 1, 2, 1, 1)
        self.label_3 = QtGui.QLabel(Tilt)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 5)
        self.tilt_z_entry = QtGui.QLineEdit(Tilt)
        self.tilt_z_entry.setObjectName(_fromUtf8("tilt_z_entry"))
        self.gridLayout.addWidget(self.tilt_z_entry, 1, 7, 1, 1)
        self.tilt_a_entry = QtGui.QLineEdit(Tilt)
        self.tilt_a_entry.setObjectName(_fromUtf8("tilt_a_entry"))
        self.gridLayout.addWidget(self.tilt_a_entry, 1, 1, 1, 1)
        self.beta_signBox = QtGui.QCheckBox(Tilt)
        self.beta_signBox.setText(_fromUtf8(""))
        self.beta_signBox.setObjectName(_fromUtf8("beta_signBox"))
        self.gridLayout.addWidget(self.beta_signBox, 1, 5, 1, 1)
        self.alpha_label_2 = QtGui.QLabel(Tilt)
        self.alpha_label_2.setObjectName(_fromUtf8("alpha_label_2"))
        self.gridLayout.addWidget(self.alpha_label_2, 1, 0, 1, 1)
        self.tilt_b_entry = QtGui.QLineEdit(Tilt)
        self.tilt_b_entry.setObjectName(_fromUtf8("tilt_b_entry"))
        self.gridLayout.addWidget(self.tilt_b_entry, 1, 4, 1, 1)
        self.z_label = QtGui.QLabel(Tilt)
        self.z_label.setObjectName(_fromUtf8("z_label"))
        self.gridLayout.addWidget(self.z_label, 1, 6, 1, 1)
        self.t_ang_entry = QtGui.QLineEdit(Tilt)
        self.t_ang_entry.setObjectName(_fromUtf8("t_ang_entry"))
        self.gridLayout.addWidget(self.t_ang_entry, 2, 5, 1, 4)

        self.retranslateUi(Tilt)
        QtCore.QMetaObject.connectSlotsByName(Tilt)

    def retranslateUi(self, Tilt):
        Tilt.setWindowTitle(_translate("Tilt", "Tilt conditions", None))
        self.beta_label.setText(_translate("Tilt", "β (°)", None))
        self.label_3.setText(_translate("Tilt", "Diffraction-α tilt angle", None))
        self.alpha_label_2.setText(_translate("Tilt", "α (°)", None))
        self.z_label.setText(_translate("Tilt", "θ (°)", None))

