# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'schmidUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Schmid(object):
    def setupUi(self, Schmid):
        Schmid.setObjectName("Schmid")
        Schmid.resize(343, 320)
        self.layoutWidget = QtWidgets.QWidget(Schmid)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 318, 298))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.schmid_text = QtWidgets.QTextEdit(self.layoutWidget)
        self.schmid_text.setObjectName("schmid_text")
        self.gridLayout.addWidget(self.schmid_text, 6, 0, 1, 3)
        self.b_label = QtWidgets.QLabel(self.layoutWidget)
        self.b_label.setObjectName("b_label")
        self.gridLayout.addWidget(self.b_label, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 7, 0, 1, 2)
        self.T_label = QtWidgets.QLabel(self.layoutWidget)
        self.T_label.setObjectName("T_label")
        self.gridLayout.addWidget(self.T_label, 3, 0, 1, 1)
        self.b_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.b_entry.setObjectName("b_entry")
        self.gridLayout.addWidget(self.b_entry, 0, 1, 1, 1)
        self.T_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.T_entry.setObjectName("T_entry")
        self.gridLayout.addWidget(self.T_entry, 3, 1, 1, 1)
        self.n_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.n_entry.setObjectName("n_entry")
        self.gridLayout.addWidget(self.n_entry, 2, 1, 1, 1)
        self.n_label = QtWidgets.QLabel(self.layoutWidget)
        self.n_label.setObjectName("n_label")
        self.gridLayout.addWidget(self.n_label, 2, 0, 1, 1)
        self.schmid_factor_label = QtWidgets.QLabel(self.layoutWidget)
        self.schmid_factor_label.setText("")
        self.schmid_factor_label.setObjectName("schmid_factor_label")
        self.gridLayout.addWidget(self.schmid_factor_label, 4, 1, 1, 1)

        self.retranslateUi(Schmid)
        QtCore.QMetaObject.connectSlotsByName(Schmid)
        Schmid.setTabOrder(self.b_entry, self.n_entry)
        Schmid.setTabOrder(self.n_entry, self.T_entry)
        Schmid.setTabOrder(self.T_entry, self.schmid_text)
        Schmid.setTabOrder(self.schmid_text, self.buttonBox)

    def retranslateUi(self, Schmid):
        _translate = QtCore.QCoreApplication.translate
        Schmid.setWindowTitle(_translate("Schmid", "Schmid Factor"))
        self.b_label.setText(_translate("Schmid", "b"))
        self.T_label.setText(_translate("Schmid", "T"))
        self.n_label.setText(_translate("Schmid", "n"))
