# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'listUI.ui'
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

class Ui_List(object):
    def setupUi(self, List):
        List.setObjectName(_fromUtf8("List"))
        List.resize(452, 557)
        self.gridLayout = QtGui.QGridLayout(List)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.list_button = QtGui.QPushButton(List)
        self.list_button.setObjectName(_fromUtf8("list_button"))
        self.gridLayout.addWidget(self.list_button, 1, 0, 1, 6)
        self.beta_button = QtGui.QRadioButton(List)
        self.beta_button.setObjectName(_fromUtf8("beta_button"))
        self.gridLayout.addWidget(self.beta_button, 4, 3, 1, 1)
        self.z_button = QtGui.QRadioButton(List)
        self.z_button.setObjectName(_fromUtf8("z_button"))
        self.gridLayout.addWidget(self.z_button, 4, 2, 1, 1)
        self.list_table = QtGui.QTableWidget(List)
        self.list_table.setObjectName(_fromUtf8("list_table"))
        self.list_table.setColumnCount(0)
        self.list_table.setRowCount(0)
        self.gridLayout.addWidget(self.list_table, 0, 0, 1, 6)
        self.ZA_button = QtGui.QCheckBox(List)
        self.ZA_button.setObjectName(_fromUtf8("ZA_button"))
        self.gridLayout.addWidget(self.ZA_button, 4, 4, 1, 1)
        self.plot_button = QtGui.QPushButton(List)
        self.plot_button.setObjectName(_fromUtf8("plot_button"))
        self.gridLayout.addWidget(self.plot_button, 2, 0, 1, 6)
        self.tilt_label = QtGui.QLabel(List)
        self.tilt_label.setObjectName(_fromUtf8("tilt_label"))
        self.gridLayout.addWidget(self.tilt_label, 4, 0, 1, 1)
        self.alpha_button = QtGui.QPushButton(List)
        self.alpha_button.setObjectName(_fromUtf8("alpha_button"))
        self.gridLayout.addWidget(self.alpha_button, 10, 0, 1, 6)
        self.tilt_entry = QtGui.QLineEdit(List)
        self.tilt_entry.setObjectName(_fromUtf8("tilt_entry"))
        self.gridLayout.addWidget(self.tilt_entry, 4, 1, 1, 1)

        self.retranslateUi(List)
        QtCore.QMetaObject.connectSlotsByName(List)
        List.setTabOrder(self.list_table, self.list_button)
        List.setTabOrder(self.list_button, self.plot_button)
        List.setTabOrder(self.plot_button, self.tilt_entry)
        List.setTabOrder(self.tilt_entry, self.z_button)
        List.setTabOrder(self.z_button, self.beta_button)
        List.setTabOrder(self.beta_button, self.ZA_button)
        List.setTabOrder(self.ZA_button, self.alpha_button)

    def retranslateUi(self, List):
        List.setWindowTitle(_translate("List", "List", None))
        self.list_button.setText(_translate("List", "Update list", None))
        self.beta_button.setText(_translate("List", "β tilt", None))
        self.z_button.setText(_translate("List", "θ tilt", None))
        self.ZA_button.setText(_translate("List", "Zone Axis", None))
        self.plot_button.setText(_translate("List", "Add/Remove selected", None))
        self.tilt_label.setText(_translate("List", "tilt", None))
        self.alpha_button.setText(_translate("List", "Compute tilt", None))

