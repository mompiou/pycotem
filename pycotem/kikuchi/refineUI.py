# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'refineUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Refine(object):
    def setupUi(self, Refine):
        Refine.setObjectName("Refine")
        Refine.resize(341, 237)
        self.gridLayout_2 = QtWidgets.QGridLayout(Refine)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.ty_label = QtWidgets.QLabel(Refine)
        self.ty_label.setText("")
        self.ty_label.setObjectName("ty_label")
        self.gridLayout.addWidget(self.ty_label, 1, 4, 1, 1)
        self.Rz_label = QtWidgets.QLabel(Refine)
        self.Rz_label.setObjectName("Rz_label")
        self.gridLayout.addWidget(self.Rz_label, 2, 0, 1, 1)
        self.Rxp_button = QtWidgets.QPushButton(Refine)
        self.Rxp_button.setObjectName("Rxp_button")
        self.gridLayout.addWidget(self.Rxp_button, 0, 3, 1, 1)
        self.Rx_entry = QtWidgets.QLineEdit(Refine)
        self.Rx_entry.setObjectName("Rx_entry")
        self.gridLayout.addWidget(self.Rx_entry, 0, 2, 1, 1)
        self.Rz_entry = QtWidgets.QLineEdit(Refine)
        self.Rz_entry.setObjectName("Rz_entry")
        self.gridLayout.addWidget(self.Rz_entry, 2, 2, 1, 1)
        self.tx_label = QtWidgets.QLabel(Refine)
        self.tx_label.setText("")
        self.tx_label.setObjectName("tx_label")
        self.gridLayout.addWidget(self.tx_label, 0, 4, 1, 1)
        self.Rym_button = QtWidgets.QPushButton(Refine)
        self.Rym_button.setObjectName("Rym_button")
        self.gridLayout.addWidget(self.Rym_button, 1, 1, 1, 1)
        self.tz_label = QtWidgets.QLabel(Refine)
        self.tz_label.setText("")
        self.tz_label.setObjectName("tz_label")
        self.gridLayout.addWidget(self.tz_label, 2, 4, 1, 1)
        self.L_label = QtWidgets.QLabel(Refine)
        self.L_label.setObjectName("L_label")
        self.gridLayout.addWidget(self.L_label, 3, 0, 1, 1)
        self.Ryp_button = QtWidgets.QPushButton(Refine)
        self.Ryp_button.setObjectName("Ryp_button")
        self.gridLayout.addWidget(self.Ryp_button, 1, 3, 1, 1)
        self.Ry_entry = QtWidgets.QLineEdit(Refine)
        self.Ry_entry.setObjectName("Ry_entry")
        self.gridLayout.addWidget(self.Ry_entry, 1, 2, 1, 1)
        self.L_entry = QtWidgets.QLineEdit(Refine)
        self.L_entry.setObjectName("L_entry")
        self.gridLayout.addWidget(self.L_entry, 3, 1, 1, 1)
        self.Rzm_button = QtWidgets.QPushButton(Refine)
        self.Rzm_button.setObjectName("Rzm_button")
        self.gridLayout.addWidget(self.Rzm_button, 2, 1, 1, 1)
        self.Rzp_button = QtWidgets.QPushButton(Refine)
        self.Rzp_button.setObjectName("Rzp_button")
        self.gridLayout.addWidget(self.Rzp_button, 2, 3, 1, 1)
        self.Ry_label = QtWidgets.QLabel(Refine)
        self.Ry_label.setObjectName("Ry_label")
        self.gridLayout.addWidget(self.Ry_label, 1, 0, 1, 1)
        self.Rxm_button = QtWidgets.QPushButton(Refine)
        self.Rxm_button.setObjectName("Rxm_button")
        self.gridLayout.addWidget(self.Rxm_button, 0, 1, 1, 1)
        self.Rx_label = QtWidgets.QLabel(Refine)
        self.Rx_label.setObjectName("Rx_label")
        self.gridLayout.addWidget(self.Rx_label, 0, 0, 1, 1)
        self.euler_label = QtWidgets.QLabel(Refine)
        self.euler_label.setText("")
        self.euler_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.euler_label.setObjectName("euler_label")
        self.gridLayout.addWidget(self.euler_label, 5, 0, 2, 5)
        self.V_label = QtWidgets.QLabel(Refine)
        self.V_label.setObjectName("V_label")
        self.gridLayout.addWidget(self.V_label, 4, 0, 1, 1)
        self.V_entry = QtWidgets.QLineEdit(Refine)
        self.V_entry.setObjectName("V_entry")
        self.gridLayout.addWidget(self.V_entry, 4, 1, 1, 1)
        self.update_button = QtWidgets.QPushButton(Refine)
        self.update_button.setObjectName("update_button")
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
        _translate = QtCore.QCoreApplication.translate
        Refine.setWindowTitle(_translate("Refine", "Refine orientation"))
        self.Rz_label.setText(_translate("Refine", "Rz"))
        self.Rxp_button.setText(_translate("Refine", "+"))
        self.Rym_button.setText(_translate("Refine", "-"))
        self.L_label.setText(_translate("Refine", "rd (pxA)"))
        self.Ryp_button.setText(_translate("Refine", "+"))
        self.Rzm_button.setText(_translate("Refine", "-"))
        self.Rzp_button.setText(_translate("Refine", "+"))
        self.Ry_label.setText(_translate("Refine", "Ry"))
        self.Rxm_button.setText(_translate("Refine", "-"))
        self.Rx_label.setText(_translate("Refine", "Rx"))
        self.V_label.setText(_translate("Refine", "V (kV)"))
        self.update_button.setText(_translate("Refine", "update"))
