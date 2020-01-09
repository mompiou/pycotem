# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widthUI.ui'
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

class Ui_Width(object):
    def setupUi(self, Width):
        Width.setObjectName(_fromUtf8("Width"))
        Width.resize(682, 658)
        self.layoutWidget = QtGui.QWidget(Width)
        self.layoutWidget.setGeometry(QtCore.QRect(9, 9, 661, 641))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.mplwindow = QtGui.QWidget(self.layoutWidget)
        self.mplwindow.setObjectName(_fromUtf8("mplwindow"))
        self.gridLayoutWidget = QtGui.QWidget(self.mplwindow)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(-1, -1, 661, 501))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.mplvl = QtGui.QGridLayout(self.gridLayoutWidget)
        self.mplvl.setMargin(0)
        self.mplvl.setObjectName(_fromUtf8("mplvl"))
        self.gridLayout.addWidget(self.mplwindow, 0, 0, 1, 7)
        self.thickness_label = QtGui.QLabel(self.layoutWidget)
        self.thickness_label.setObjectName(_fromUtf8("thickness_label"))
        self.gridLayout.addWidget(self.thickness_label, 3, 0, 1, 1)
        self.clear_button = QtGui.QPushButton(self.layoutWidget)
        self.clear_button.setObjectName(_fromUtf8("clear_button"))
        self.gridLayout.addWidget(self.clear_button, 1, 4, 1, 1)
        self.trace_radio_button = QtGui.QRadioButton(self.layoutWidget)
        self.trace_radio_button.setObjectName(_fromUtf8("trace_radio_button"))
        self.gridLayout.addWidget(self.trace_radio_button, 1, 2, 1, 2)
        self.thickness_checkBox = QtGui.QCheckBox(self.layoutWidget)
        self.thickness_checkBox.setObjectName(_fromUtf8("thickness_checkBox"))
        self.gridLayout.addWidget(self.thickness_checkBox, 3, 2, 1, 2)
        self.plane_entry = QtGui.QLineEdit(self.layoutWidget)
        self.plane_entry.setObjectName(_fromUtf8("plane_entry"))
        self.gridLayout.addWidget(self.plane_entry, 1, 1, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 4, 3, 1, 4)
        self.label = QtGui.QLabel(self.layoutWidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.thickness = QtGui.QLineEdit(self.layoutWidget)
        self.thickness.setObjectName(_fromUtf8("thickness"))
        self.gridLayout.addWidget(self.thickness, 3, 1, 1, 1)
        self.label_2 = QtGui.QLabel(self.layoutWidget)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.foil_surface = QtGui.QLineEdit(self.layoutWidget)
        self.foil_surface.setObjectName(_fromUtf8("foil_surface"))
        self.gridLayout.addWidget(self.foil_surface, 2, 1, 1, 1)
        self.surface_box = QtGui.QCheckBox(self.layoutWidget)
        self.surface_box.setText(_fromUtf8(""))
        self.surface_box.setObjectName(_fromUtf8("surface_box"))
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
        Width.setWindowTitle(_translate("Width", "Apparent Width", None))
        self.thickness_label.setText(_translate("Width", "thickness (nm)", None))
        self.clear_button.setText(_translate("Width", "clear", None))
        self.trace_radio_button.setText(_translate("Width", "trace dir.", None))
        self.thickness_checkBox.setText(_translate("Width", "w (nm)", None))
        self.label.setText(_translate("Width", "plane", None))
        self.label_2.setText(_translate("Width", "foil surface", None))

