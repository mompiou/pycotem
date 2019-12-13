# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'schmidUI.ui'
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

class Ui_Schmid(object):
    def setupUi(self, Schmid):
        Schmid.setObjectName(_fromUtf8("Schmid"))
        Schmid.resize(343, 320)
        self.layoutWidget = QtGui.QWidget(Schmid)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 318, 298))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.schmid_text = QtGui.QTextEdit(self.layoutWidget)
        self.schmid_text.setObjectName(_fromUtf8("schmid_text"))
        self.gridLayout.addWidget(self.schmid_text, 6, 0, 1, 3)
        self.b_label = QtGui.QLabel(self.layoutWidget)
        self.b_label.setObjectName(_fromUtf8("b_label"))
        self.gridLayout.addWidget(self.b_label, 0, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(self.layoutWidget)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 7, 0, 1, 2)
        self.T_label = QtGui.QLabel(self.layoutWidget)
        self.T_label.setObjectName(_fromUtf8("T_label"))
        self.gridLayout.addWidget(self.T_label, 3, 0, 1, 1)
        self.b_entry = QtGui.QLineEdit(self.layoutWidget)
        self.b_entry.setObjectName(_fromUtf8("b_entry"))
        self.gridLayout.addWidget(self.b_entry, 0, 1, 1, 1)
        self.T_entry = QtGui.QLineEdit(self.layoutWidget)
        self.T_entry.setObjectName(_fromUtf8("T_entry"))
        self.gridLayout.addWidget(self.T_entry, 3, 1, 1, 1)
        self.n_entry = QtGui.QLineEdit(self.layoutWidget)
        self.n_entry.setObjectName(_fromUtf8("n_entry"))
        self.gridLayout.addWidget(self.n_entry, 2, 1, 1, 1)
        self.n_label = QtGui.QLabel(self.layoutWidget)
        self.n_label.setObjectName(_fromUtf8("n_label"))
        self.gridLayout.addWidget(self.n_label, 2, 0, 1, 1)
        self.schmid_factor_label = QtGui.QLabel(self.layoutWidget)
        self.schmid_factor_label.setText(_fromUtf8(""))
        self.schmid_factor_label.setObjectName(_fromUtf8("schmid_factor_label"))
        self.gridLayout.addWidget(self.schmid_factor_label, 4, 1, 1, 1)

        self.retranslateUi(Schmid)
        QtCore.QMetaObject.connectSlotsByName(Schmid)
        Schmid.setTabOrder(self.b_entry, self.n_entry)
        Schmid.setTabOrder(self.n_entry, self.T_entry)
        Schmid.setTabOrder(self.T_entry, self.schmid_text)
        Schmid.setTabOrder(self.schmid_text, self.buttonBox)

    def retranslateUi(self, Schmid):
        Schmid.setWindowTitle(_translate("Schmid", "Schmid Factor", None))
        self.b_label.setText(_translate("Schmid", "b", None))
        self.T_label.setText(_translate("Schmid", "T", None))
        self.n_label.setText(_translate("Schmid", "n", None))

