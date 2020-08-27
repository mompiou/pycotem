# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widthUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Width(object):
    def setupUi(self, Width):
        Width.setObjectName("Width")
        Width.resize(682, 658)
        self.layoutWidget = QtWidgets.QWidget(Width)
        self.layoutWidget.setGeometry(QtCore.QRect(9, 9, 661, 641))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.mplwindow = QtWidgets.QWidget(self.layoutWidget)
        self.mplwindow.setObjectName("mplwindow")
        self.gridLayoutWidget = QtWidgets.QWidget(self.mplwindow)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(-1, -1, 661, 501))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.mplvl = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.mplvl.setContentsMargins(0, 0, 0, 0)
        self.mplvl.setObjectName("mplvl")
        self.gridLayout.addWidget(self.mplwindow, 0, 0, 1, 7)
        self.thickness_label = QtWidgets.QLabel(self.layoutWidget)
        self.thickness_label.setObjectName("thickness_label")
        self.gridLayout.addWidget(self.thickness_label, 3, 0, 1, 1)
        self.clear_button = QtWidgets.QPushButton(self.layoutWidget)
        self.clear_button.setObjectName("clear_button")
        self.gridLayout.addWidget(self.clear_button, 1, 4, 1, 1)
        self.trace_radio_button = QtWidgets.QRadioButton(self.layoutWidget)
        self.trace_radio_button.setObjectName("trace_radio_button")
        self.gridLayout.addWidget(self.trace_radio_button, 1, 2, 1, 2)
        self.thickness_checkBox = QtWidgets.QCheckBox(self.layoutWidget)
        self.thickness_checkBox.setObjectName("thickness_checkBox")
        self.gridLayout.addWidget(self.thickness_checkBox, 3, 2, 1, 2)
        self.plane_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.plane_entry.setObjectName("plane_entry")
        self.gridLayout.addWidget(self.plane_entry, 1, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 3, 1, 4)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.thickness = QtWidgets.QLineEdit(self.layoutWidget)
        self.thickness.setObjectName("thickness")
        self.gridLayout.addWidget(self.thickness, 3, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.foil_surface = QtWidgets.QLineEdit(self.layoutWidget)
        self.foil_surface.setObjectName("foil_surface")
        self.gridLayout.addWidget(self.foil_surface, 2, 1, 1, 1)
        self.surface_box = QtWidgets.QCheckBox(self.layoutWidget)
        self.surface_box.setText("")
        self.surface_box.setObjectName("surface_box")
        self.gridLayout.addWidget(self.surface_box, 2, 2, 1, 2)

        self.retranslateUi(Width)
        QtCore.QMetaObject.connectSlotsByName(Width)
        Width.setTabOrder(self.plane_entry, self.trace_radio_button)
        Width.setTabOrder(self.trace_radio_button, self.foil_surface)
        Width.setTabOrder(self.foil_surface, self.surface_box)
        Width.setTabOrder(self.surface_box, self.thickness)
        Width.setTabOrder(self.thickness, self.thickness_checkBox)
        Width.setTabOrder(self.thickness_checkBox, self.clear_button)
        Width.setTabOrder(self.clear_button, self.buttonBox)

    def retranslateUi(self, Width):
        _translate = QtCore.QCoreApplication.translate
        Width.setWindowTitle(_translate("Width", "Apparent Width"))
        self.thickness_label.setText(_translate("Width", "thickness (nm)"))
        self.clear_button.setText(_translate("Width", "clear"))
        self.trace_radio_button.setText(_translate("Width", "trace dir."))
        self.thickness_checkBox.setText(_translate("Width", "w (nm)"))
        self.label.setText(_translate("Width", "plane"))
        self.label_2.setText(_translate("Width", "foil surface"))
