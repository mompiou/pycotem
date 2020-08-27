# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tiltUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Tilt(object):
    def setupUi(self, Tilt):
        Tilt.setObjectName("Tilt")
        Tilt.resize(396, 117)
        self.gridLayout = QtWidgets.QGridLayout(Tilt)
        self.gridLayout.setObjectName("gridLayout")
        self.theta_signBox = QtWidgets.QCheckBox(Tilt)
        self.theta_signBox.setText("")
        self.theta_signBox.setObjectName("theta_signBox")
        self.gridLayout.addWidget(self.theta_signBox, 1, 8, 1, 1)
        self.beta_label = QtWidgets.QLabel(Tilt)
        self.beta_label.setObjectName("beta_label")
        self.gridLayout.addWidget(self.beta_label, 1, 3, 1, 1)
        self.alpha_signBox = QtWidgets.QCheckBox(Tilt)
        self.alpha_signBox.setText("")
        self.alpha_signBox.setObjectName("alpha_signBox")
        self.gridLayout.addWidget(self.alpha_signBox, 1, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(Tilt)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 5)
        self.tilt_z_entry = QtWidgets.QLineEdit(Tilt)
        self.tilt_z_entry.setObjectName("tilt_z_entry")
        self.gridLayout.addWidget(self.tilt_z_entry, 1, 7, 1, 1)
        self.tilt_a_entry = QtWidgets.QLineEdit(Tilt)
        self.tilt_a_entry.setObjectName("tilt_a_entry")
        self.gridLayout.addWidget(self.tilt_a_entry, 1, 1, 1, 1)
        self.beta_signBox = QtWidgets.QCheckBox(Tilt)
        self.beta_signBox.setText("")
        self.beta_signBox.setObjectName("beta_signBox")
        self.gridLayout.addWidget(self.beta_signBox, 1, 5, 1, 1)
        self.alpha_label_2 = QtWidgets.QLabel(Tilt)
        self.alpha_label_2.setObjectName("alpha_label_2")
        self.gridLayout.addWidget(self.alpha_label_2, 1, 0, 1, 1)
        self.tilt_b_entry = QtWidgets.QLineEdit(Tilt)
        self.tilt_b_entry.setObjectName("tilt_b_entry")
        self.gridLayout.addWidget(self.tilt_b_entry, 1, 4, 1, 1)
        self.z_label = QtWidgets.QLabel(Tilt)
        self.z_label.setObjectName("z_label")
        self.gridLayout.addWidget(self.z_label, 1, 6, 1, 1)
        self.t_ang_entry = QtWidgets.QLineEdit(Tilt)
        self.t_ang_entry.setObjectName("t_ang_entry")
        self.gridLayout.addWidget(self.t_ang_entry, 2, 5, 1, 4)

        self.retranslateUi(Tilt)
        QtCore.QMetaObject.connectSlotsByName(Tilt)

    def retranslateUi(self, Tilt):
        _translate = QtCore.QCoreApplication.translate
        Tilt.setWindowTitle(_translate("Tilt", "Tilt conditions"))
        self.beta_label.setText(_translate("Tilt", "β (°)"))
        self.label_3.setText(_translate("Tilt", "Diffraction-α tilt angle"))
        self.alpha_label_2.setText(_translate("Tilt", "α (°)"))
        self.z_label.setText(_translate("Tilt", "θ (°)"))
