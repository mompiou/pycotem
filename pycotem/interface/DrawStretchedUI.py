# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DrawStretchedUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Draw_Stretched(object):
    def setupUi(self, Draw_Stretched):
        Draw_Stretched.setObjectName("Draw_Stretched")
        Draw_Stretched.resize(772, 711)
        self.gridLayout = QtWidgets.QGridLayout(Draw_Stretched)
        self.gridLayout.setObjectName("gridLayout")
        self.mplvl = QtWidgets.QGridLayout()
        self.mplvl.setObjectName("mplvl")
        self.gridLayout.addLayout(self.mplvl, 0, 0, 1, 1)

        self.retranslateUi(Draw_Stretched)
        QtCore.QMetaObject.connectSlotsByName(Draw_Stretched)

    def retranslateUi(self, Draw_Stretched):
        _translate = QtCore.QCoreApplication.translate
        Draw_Stretched.setWindowTitle(_translate("Draw_Stretched", "Draw stretched image"))
