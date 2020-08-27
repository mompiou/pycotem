# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hkl_uvwUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_hkl_uvw(object):
    def setupUi(self, hkl_uvw):
        hkl_uvw.setObjectName("hkl_uvw")
        hkl_uvw.resize(300, 341)
        self.layoutWidget = QtWidgets.QWidget(hkl_uvw)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 281, 321))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton_to_hkl = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_to_hkl.setObjectName("pushButton_to_hkl")
        self.gridLayout.addWidget(self.pushButton_to_hkl, 6, 0, 1, 1)
        self.hkl_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.hkl_text_label.setObjectName("hkl_text_label")
        self.gridLayout.addWidget(self.hkl_text_label, 0, 0, 1, 1)
        self.hkl_label = QtWidgets.QLabel(self.layoutWidget)
        self.hkl_label.setText("")
        self.hkl_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.hkl_label.setObjectName("hkl_label")
        self.gridLayout.addWidget(self.hkl_label, 7, 0, 1, 1)
        self.uvw_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.uvw_entry.setObjectName("uvw_entry")
        self.gridLayout.addWidget(self.uvw_entry, 5, 0, 1, 1)
        self.uvw_text_label = QtWidgets.QLabel(self.layoutWidget)
        self.uvw_text_label.setObjectName("uvw_text_label")
        self.gridLayout.addWidget(self.uvw_text_label, 4, 0, 1, 1)
        self.pushButton_to_uvw = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_to_uvw.setObjectName("pushButton_to_uvw")
        self.gridLayout.addWidget(self.pushButton_to_uvw, 2, 0, 1, 1)
        self.uvw_label = QtWidgets.QLabel(self.layoutWidget)
        self.uvw_label.setText("")
        self.uvw_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.uvw_label.setObjectName("uvw_label")
        self.gridLayout.addWidget(self.uvw_label, 3, 0, 1, 1)
        self.hkl_entry = QtWidgets.QLineEdit(self.layoutWidget)
        self.hkl_entry.setObjectName("hkl_entry")
        self.gridLayout.addWidget(self.hkl_entry, 1, 0, 1, 1)
        self.layoutWidget.raise_()
        self.uvw_entry.raise_()

        self.retranslateUi(hkl_uvw)
        QtCore.QMetaObject.connectSlotsByName(hkl_uvw)
        hkl_uvw.setTabOrder(self.hkl_entry, self.pushButton_to_uvw)
        hkl_uvw.setTabOrder(self.pushButton_to_uvw, self.uvw_entry)
        hkl_uvw.setTabOrder(self.uvw_entry, self.pushButton_to_hkl)

    def retranslateUi(self, hkl_uvw):
        _translate = QtCore.QCoreApplication.translate
        hkl_uvw.setWindowTitle(_translate("hkl_uvw", "hkl-uvw"))
        self.pushButton_to_hkl.setText(_translate("hkl_uvw", "To hkl"))
        self.hkl_text_label.setText(_translate("hkl_uvw", "hkl"))
        self.uvw_text_label.setText(_translate("hkl_uvw", "uvw"))
        self.pushButton_to_uvw.setText(_translate("hkl_uvw", "To uvw"))
