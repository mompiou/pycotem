# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'angleUI.ui'
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

class Ui_Angle(object):
    def setupUi(self, Angle):
        Angle.setObjectName(_fromUtf8("Angle"))
        Angle.resize(288, 220)
        self.layoutWidget = QtGui.QWidget(Angle)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 266, 205))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.angle_label = QtGui.QLabel(self.layoutWidget)
        self.angle_label.setText(_fromUtf8(""))
        self.angle_label.setObjectName(_fromUtf8("angle_label"))
        self.gridLayout.addWidget(self.angle_label, 4, 0, 1, 2)
        self.buttonBox = QtGui.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 5, 0, 1, 2)
        self.n1_entry = QtGui.QLineEdit(self.layoutWidget)
        self.n1_entry.setObjectName(_fromUtf8("n1_entry"))
        self.gridLayout.addWidget(self.n1_entry, 1, 0, 1, 1)
        self.n2_entry = QtGui.QLineEdit(self.layoutWidget)
        self.n2_entry.setObjectName(_fromUtf8("n2_entry"))
        self.gridLayout.addWidget(self.n2_entry, 3, 0, 1, 1)
        self.n1_label = QtGui.QLabel(self.layoutWidget)
        self.n1_label.setObjectName(_fromUtf8("n1_label"))
        self.gridLayout.addWidget(self.n1_label, 0, 0, 1, 1)
        self.n2_label = QtGui.QLabel(self.layoutWidget)
        self.n2_label.setObjectName(_fromUtf8("n2_label"))
        self.gridLayout.addWidget(self.n2_label, 2, 0, 1, 1)

        self.retranslateUi(Angle)
        QtCore.QMetaObject.connectSlotsByName(Angle)
        Angle.setTabOrder(self.n1_entry, self.n2_entry)
        Angle.setTabOrder(self.n2_entry, self.buttonBox)

    def retranslateUi(self, Angle):
        Angle.setWindowTitle(_translate("Angle", "Angle", None))
        self.n1_label.setText(_translate("Angle", "n1", None))
        self.n2_label.setText(_translate("Angle", "n2", None))

