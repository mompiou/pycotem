# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'refineUI.ui'
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

class Ui_Refine(object):
    def setupUi(self, Refine):
        Refine.setObjectName(_fromUtf8("Refine"))
        Refine.resize(339, 222)
        self.gridLayout_2 = QtGui.QGridLayout(Refine)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.ty_label = QtGui.QLabel(Refine)
        self.ty_label.setObjectName(_fromUtf8("ty_label"))
        self.gridLayout.addWidget(self.ty_label, 1, 4, 1, 1)
        self.Rz_label = QtGui.QLabel(Refine)
        self.Rz_label.setObjectName(_fromUtf8("Rz_label"))
        self.gridLayout.addWidget(self.Rz_label, 2, 0, 1, 1)
        self.Rxp_button = QtGui.QPushButton(Refine)
        self.Rxp_button.setObjectName(_fromUtf8("Rxp_button"))
        self.gridLayout.addWidget(self.Rxp_button, 0, 3, 1, 1)
        self.Rx_entry = QtGui.QLineEdit(Refine)
        self.Rx_entry.setObjectName(_fromUtf8("Rx_entry"))
        self.gridLayout.addWidget(self.Rx_entry, 0, 2, 1, 1)
        self.Rz_entry = QtGui.QLineEdit(Refine)
        self.Rz_entry.setObjectName(_fromUtf8("Rz_entry"))
        self.gridLayout.addWidget(self.Rz_entry, 2, 2, 1, 1)
        self.tx_label = QtGui.QLabel(Refine)
        self.tx_label.setText(_fromUtf8(""))
        self.tx_label.setObjectName(_fromUtf8("tx_label"))
        self.gridLayout.addWidget(self.tx_label, 0, 4, 1, 1)
        self.Rym_button = QtGui.QPushButton(Refine)
        self.Rym_button.setObjectName(_fromUtf8("Rym_button"))
        self.gridLayout.addWidget(self.Rym_button, 1, 1, 1, 1)
        self.tz_label = QtGui.QLabel(Refine)
        self.tz_label.setObjectName(_fromUtf8("tz_label"))
        self.gridLayout.addWidget(self.tz_label, 2, 4, 1, 1)
        self.L_label = QtGui.QLabel(Refine)
        self.L_label.setObjectName(_fromUtf8("L_label"))
        self.gridLayout.addWidget(self.L_label, 3, 0, 1, 1)
        self.Ryp_button = QtGui.QPushButton(Refine)
        self.Ryp_button.setObjectName(_fromUtf8("Ryp_button"))
        self.gridLayout.addWidget(self.Ryp_button, 1, 3, 1, 1)
        self.Ry_entry = QtGui.QLineEdit(Refine)
        self.Ry_entry.setObjectName(_fromUtf8("Ry_entry"))
        self.gridLayout.addWidget(self.Ry_entry, 1, 2, 1, 1)
        self.L_entry = QtGui.QLineEdit(Refine)
        self.L_entry.setObjectName(_fromUtf8("L_entry"))
        self.gridLayout.addWidget(self.L_entry, 3, 1, 1, 1)
        self.Rzm_button = QtGui.QPushButton(Refine)
        self.Rzm_button.setObjectName(_fromUtf8("Rzm_button"))
        self.gridLayout.addWidget(self.Rzm_button, 2, 1, 1, 1)
        self.Rzp_button = QtGui.QPushButton(Refine)
        self.Rzp_button.setObjectName(_fromUtf8("Rzp_button"))
        self.gridLayout.addWidget(self.Rzp_button, 2, 3, 1, 1)
        self.Ry_label = QtGui.QLabel(Refine)
        self.Ry_label.setObjectName(_fromUtf8("Ry_label"))
        self.gridLayout.addWidget(self.Ry_label, 1, 0, 1, 1)
        self.Rxm_button = QtGui.QPushButton(Refine)
        self.Rxm_button.setObjectName(_fromUtf8("Rxm_button"))
        self.gridLayout.addWidget(self.Rxm_button, 0, 1, 1, 1)
        self.Rx_label = QtGui.QLabel(Refine)
        self.Rx_label.setObjectName(_fromUtf8("Rx_label"))
        self.gridLayout.addWidget(self.Rx_label, 0, 0, 1, 1)
        self.euler_label = QtGui.QLabel(Refine)
        self.euler_label.setText(_fromUtf8(""))
        self.euler_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.euler_label.setObjectName(_fromUtf8("euler_label"))
        self.gridLayout.addWidget(self.euler_label, 5, 0, 2, 5)
        self.V_label = QtGui.QLabel(Refine)
        self.V_label.setObjectName(_fromUtf8("V_label"))
        self.gridLayout.addWidget(self.V_label, 4, 0, 1, 1)
        self.V_entry = QtGui.QLineEdit(Refine)
        self.V_entry.setObjectName(_fromUtf8("V_entry"))
        self.gridLayout.addWidget(self.V_entry, 4, 1, 1, 1)
        self.update_button = QtGui.QPushButton(Refine)
        self.update_button.setObjectName(_fromUtf8("update_button"))
        self.gridLayout.addWidget(self.update_button, 4, 3, 1, 2)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)

        self.retranslateUi(Refine)
        QtCore.QMetaObject.connectSlotsByName(Refine)
        Refine.setTabOrder(self.Rxm_button, self.Rx_entry)
        Refine.setTabOrder(self.Rx_entry, self.Rxp_button)
        Refine.setTabOrder(self.Rxp_button, self.Rym_button)
        Refine.setTabOrder(self.Rym_button, self.Ry_entry)
        Refine.setTabOrder(self.Ry_entry, self.Ryp_button)
        Refine.setTabOrder(self.Ryp_button, self.Rzm_button)
        Refine.setTabOrder(self.Rzm_button, self.Rz_entry)
        Refine.setTabOrder(self.Rz_entry, self.Rzp_button)
        Refine.setTabOrder(self.Rzp_button, self.L_entry)

    def retranslateUi(self, Refine):
        Refine.setWindowTitle(_translate("Refine", "Refine orientation", None))
        self.ty_label.setText(_translate("Refine", "                        ", None))
        self.Rz_label.setText(_translate("Refine", "Rz", None))
        self.Rxp_button.setText(_translate("Refine", "+", None))
        self.Rym_button.setText(_translate("Refine", "-", None))
        self.tz_label.setText(_translate("Refine", "                    ", None))
        self.L_label.setText(_translate("Refine", "L", None))
        self.Ryp_button.setText(_translate("Refine", "+", None))
        self.Rzm_button.setText(_translate("Refine", "-", None))
        self.Rzp_button.setText(_translate("Refine", "+", None))
        self.Ry_label.setText(_translate("Refine", "Ry", None))
        self.Rxm_button.setText(_translate("Refine", "-", None))
        self.Rx_label.setText(_translate("Refine", "Rx", None))
        self.V_label.setText(_translate("Refine", "V", None))
        self.update_button.setText(_translate("Refine", "update", None))

