# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'kikuchi.ui'
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

class Ui_Kikuchi(object):
    def setupUi(self, Kikuchi):
        Kikuchi.setObjectName(_fromUtf8("Kikuchi"))
        Kikuchi.setWindowModality(QtCore.Qt.NonModal)
        Kikuchi.resize(830, 818)
        self.layoutWidget = QtGui.QWidget(Kikuchi)
        self.layoutWidget.setGeometry(QtCore.QRect(9, 9, 811, 801))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.error_label = QtGui.QLabel(self.layoutWidget)
        self.error_label.setObjectName(_fromUtf8("error_label"))
        self.gridLayout.addWidget(self.error_label, 3, 5, 1, 1)
        self.kikuchi_radioButton = QtGui.QRadioButton(self.layoutWidget)
        self.kikuchi_radioButton.setObjectName(_fromUtf8("kikuchi_radioButton"))
        self.gridLayout.addWidget(self.kikuchi_radioButton, 4, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 6, 0, 1, 9)
        self.diff_radioButton = QtGui.QRadioButton(self.layoutWidget)
        self.diff_radioButton.setObjectName(_fromUtf8("diff_radioButton"))
        self.gridLayout.addWidget(self.diff_radioButton, 3, 0, 1, 1)
        self.t_label = QtGui.QLabel(self.layoutWidget)
        self.t_label.setObjectName(_fromUtf8("t_label"))
        self.gridLayout.addWidget(self.t_label, 3, 3, 1, 1)
        self.E_label = QtGui.QLabel(self.layoutWidget)
        self.E_label.setObjectName(_fromUtf8("E_label"))
        self.gridLayout.addWidget(self.E_label, 1, 0, 1, 1)
        self.t_entry = QtGui.QLineEdit(self.layoutWidget)
        self.t_entry.setObjectName(_fromUtf8("t_entry"))
        self.gridLayout.addWidget(self.t_entry, 3, 4, 1, 1)
        self.mplwindow = QtGui.QWidget(self.layoutWidget)
        self.mplwindow.setObjectName(_fromUtf8("mplwindow"))
        self.gridLayoutWidget = QtGui.QWidget(self.mplwindow)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 801, 661))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.mplvl = QtGui.QGridLayout(self.gridLayoutWidget)
        self.mplvl.setMargin(0)
        self.mplvl.setObjectName(_fromUtf8("mplvl"))
        self.gridLayout.addWidget(self.mplwindow, 0, 0, 1, 9)
        self.angle_label = QtGui.QLabel(self.layoutWidget)
        self.angle_label.setObjectName(_fromUtf8("angle_label"))
        self.gridLayout.addWidget(self.angle_label, 1, 3, 1, 1)
        self.spot_size_label = QtGui.QLabel(self.layoutWidget)
        self.spot_size_label.setObjectName(_fromUtf8("spot_size_label"))
        self.gridLayout.addWidget(self.spot_size_label, 1, 5, 1, 1)
        self.angle_entry = QtGui.QLineEdit(self.layoutWidget)
        self.angle_entry.setObjectName(_fromUtf8("angle_entry"))
        self.gridLayout.addWidget(self.angle_entry, 1, 4, 1, 1)
        self.spot_size_entry = QtGui.QLineEdit(self.layoutWidget)
        self.spot_size_entry.setObjectName(_fromUtf8("spot_size_entry"))
        self.gridLayout.addWidget(self.spot_size_entry, 1, 6, 1, 1)
        self.label_checkBox = QtGui.QCheckBox(self.layoutWidget)
        self.label_checkBox.setObjectName(_fromUtf8("label_checkBox"))
        self.gridLayout.addWidget(self.label_checkBox, 1, 7, 1, 2)
        self.error_entry = QtGui.QLineEdit(self.layoutWidget)
        self.error_entry.setObjectName(_fromUtf8("error_entry"))
        self.gridLayout.addWidget(self.error_entry, 3, 6, 1, 1)
        self.indices_label = QtGui.QLabel(self.layoutWidget)
        self.indices_label.setObjectName(_fromUtf8("indices_label"))
        self.gridLayout.addWidget(self.indices_label, 3, 1, 1, 1)
        self.indices_entry = QtGui.QLineEdit(self.layoutWidget)
        self.indices_entry.setObjectName(_fromUtf8("indices_entry"))
        self.gridLayout.addWidget(self.indices_entry, 3, 2, 1, 1)
        self.E_entry = QtGui.QLineEdit(self.layoutWidget)
        self.E_entry.setObjectName(_fromUtf8("E_entry"))
        self.gridLayout.addWidget(self.E_entry, 1, 1, 1, 1)
        self.Diff_button = QtGui.QPushButton(self.layoutWidget)
        self.Diff_button.setObjectName(_fromUtf8("Diff_button"))
        self.gridLayout.addWidget(self.Diff_button, 3, 7, 2, 2)

        self.retranslateUi(Kikuchi)
        QtCore.QMetaObject.connectSlotsByName(Kikuchi)
        Kikuchi.setTabOrder(self.E_entry, self.angle_entry)
        Kikuchi.setTabOrder(self.angle_entry, self.spot_size_entry)
        Kikuchi.setTabOrder(self.spot_size_entry, self.label_checkBox)
        Kikuchi.setTabOrder(self.label_checkBox, self.diff_radioButton)
        Kikuchi.setTabOrder(self.diff_radioButton, self.indices_entry)
        Kikuchi.setTabOrder(self.indices_entry, self.t_entry)
        Kikuchi.setTabOrder(self.t_entry, self.error_entry)
        Kikuchi.setTabOrder(self.error_entry, self.Diff_button)
        Kikuchi.setTabOrder(self.Diff_button, self.kikuchi_radioButton)
        Kikuchi.setTabOrder(self.kikuchi_radioButton, self.buttonBox)

    def retranslateUi(self, Kikuchi):
        Kikuchi.setWindowTitle(_translate("Kikuchi", "Kikuchi lines / Diffraction pattern", None))
        self.error_label.setText(_translate("Kikuchi", "Excitation error (nm-1)", None))
        self.kikuchi_radioButton.setText(_translate("Kikuchi", "Kikuchi", None))
        self.diff_radioButton.setText(_translate("Kikuchi", "Diffraction", None))
        self.t_label.setText(_translate("Kikuchi", "Thickness (nm)", None))
        self.E_label.setText(_translate("Kikuchi", "E (kV)", None))
        self.angle_label.setText(_translate("Kikuchi", "Aperture angle", None))
        self.spot_size_label.setText(_translate("Kikuchi", "Spot size", None))
        self.label_checkBox.setText(_translate("Kikuchi", "labels", None))
        self.indices_label.setText(_translate("Kikuchi", "Max indices", None))
        self.Diff_button.setText(_translate("Kikuchi", "Diff calc.", None))

