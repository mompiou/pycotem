# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'listUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_List(object):
    def setupUi(self, List):
        List.setObjectName("List")
        List.resize(452, 557)
        self.gridLayout = QtWidgets.QGridLayout(List)
        self.gridLayout.setObjectName("gridLayout")
        self.ZA_button = QtWidgets.QCheckBox(List)
        self.ZA_button.setObjectName("ZA_button")
        self.gridLayout.addWidget(self.ZA_button, 4, 5, 1, 1)
        self.list_button = QtWidgets.QPushButton(List)
        self.list_button.setObjectName("list_button")
        self.gridLayout.addWidget(self.list_button, 1, 0, 1, 6)
        self.beta_button = QtWidgets.QRadioButton(List)
        self.beta_button.setObjectName("beta_button")
        self.gridLayout.addWidget(self.beta_button, 4, 3, 1, 1)
        self.z_button = QtWidgets.QRadioButton(List)
        self.z_button.setObjectName("z_button")
        self.gridLayout.addWidget(self.z_button, 4, 2, 1, 1)
        self.list_table = QtWidgets.QTableWidget(List)
        self.list_table.setObjectName("list_table")
        self.list_table.setColumnCount(0)
        self.list_table.setRowCount(0)
        self.gridLayout.addWidget(self.list_table, 0, 0, 1, 6)
        self.tilt_label = QtWidgets.QLabel(List)
        self.tilt_label.setObjectName("tilt_label")
        self.gridLayout.addWidget(self.tilt_label, 4, 0, 1, 1)
        self.alpha_button = QtWidgets.QPushButton(List)
        self.alpha_button.setObjectName("alpha_button")
        self.gridLayout.addWidget(self.alpha_button, 10, 0, 1, 6)
        self.tilt_entry = QtWidgets.QLineEdit(List)
        self.tilt_entry.setObjectName("tilt_entry")
        self.gridLayout.addWidget(self.tilt_entry, 4, 1, 1, 1)
        self.plot_button = QtWidgets.QPushButton(List)
        self.plot_button.setObjectName("plot_button")
        self.gridLayout.addWidget(self.plot_button, 2, 0, 1, 3)
        self.plane_checkBox = QtWidgets.QCheckBox(List)
        self.plane_checkBox.setObjectName("plane_checkBox")
        self.gridLayout.addWidget(self.plane_checkBox, 2, 3, 1, 1)
        self.cone_checkBox = QtWidgets.QCheckBox(List)
        self.cone_checkBox.setObjectName("cone_checkBox")
        self.gridLayout.addWidget(self.cone_checkBox, 2, 5, 1, 1)

        self.retranslateUi(List)
        QtCore.QMetaObject.connectSlotsByName(List)
        List.setTabOrder(self.list_table, self.list_button)
        List.setTabOrder(self.list_button, self.plot_button)
        List.setTabOrder(self.plot_button, self.plane_checkBox)
        List.setTabOrder(self.plane_checkBox, self.cone_checkBox)
        List.setTabOrder(self.cone_checkBox, self.tilt_entry)
        List.setTabOrder(self.tilt_entry, self.z_button)
        List.setTabOrder(self.z_button, self.beta_button)
        List.setTabOrder(self.beta_button, self.ZA_button)
        List.setTabOrder(self.ZA_button, self.alpha_button)

    def retranslateUi(self, List):
        _translate = QtCore.QCoreApplication.translate
        List.setWindowTitle(_translate("List", "List"))
        self.ZA_button.setText(_translate("List", "Zone Axis"))
        self.list_button.setText(_translate("List", "Update list"))
        self.beta_button.setText(_translate("List", "β tilt"))
        self.z_button.setText(_translate("List", "θ tilt"))
        self.tilt_label.setText(_translate("List", "tilt"))
        self.alpha_button.setText(_translate("List", "Compute tilt"))
        self.plot_button.setText(_translate("List", "Add/Remove selected"))
        self.plane_checkBox.setText(_translate("List", "plane"))
        self.cone_checkBox.setText(_translate("List", "cone"))
