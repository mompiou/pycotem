# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DrawStretchedUI.ui'
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

class Ui_Draw_Stretched(object):
    def setupUi(self, Draw_Stretched):
        Draw_Stretched.setObjectName(_fromUtf8("Draw_Stretched"))
        Draw_Stretched.resize(772, 711)
        self.gridLayout = QtGui.QGridLayout(Draw_Stretched)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.mplvl = QtGui.QGridLayout()
        self.mplvl.setObjectName(_fromUtf8("mplvl"))
        self.gridLayout.addLayout(self.mplvl, 0, 0, 1, 1)

        self.retranslateUi(Draw_Stretched)
        QtCore.QMetaObject.connectSlotsByName(Draw_Stretched)

    def retranslateUi(self, Draw_Stretched):
        Draw_Stretched.setWindowTitle(_translate("Draw_Stretched", "Draw stretched image", None))

