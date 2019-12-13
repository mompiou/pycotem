# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hkl_uvwUI.ui'
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

class Ui_hkl_uvw(object):
    def setupUi(self, hkl_uvw):
        hkl_uvw.setObjectName(_fromUtf8("hkl_uvw"))
        hkl_uvw.resize(300, 341)
        self.layoutWidget = QtGui.QWidget(hkl_uvw)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 281, 321))
        self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.layoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.pushButton_to_hkl = QtGui.QPushButton(self.layoutWidget)
        self.pushButton_to_hkl.setObjectName(_fromUtf8("pushButton_to_hkl"))
        self.gridLayout.addWidget(self.pushButton_to_hkl, 6, 0, 1, 1)
        self.hkl_text_label = QtGui.QLabel(self.layoutWidget)
        self.hkl_text_label.setObjectName(_fromUtf8("hkl_text_label"))
        self.gridLayout.addWidget(self.hkl_text_label, 0, 0, 1, 1)
        self.hkl_label = QtGui.QLabel(self.layoutWidget)
        self.hkl_label.setText(_fromUtf8(""))
        self.hkl_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.hkl_label.setObjectName(_fromUtf8("hkl_label"))
        self.gridLayout.addWidget(self.hkl_label, 7, 0, 1, 1)
        self.uvw_entry = QtGui.QLineEdit(self.layoutWidget)
        self.uvw_entry.setObjectName(_fromUtf8("uvw_entry"))
        self.gridLayout.addWidget(self.uvw_entry, 5, 0, 1, 1)
        self.uvw_text_label = QtGui.QLabel(self.layoutWidget)
        self.uvw_text_label.setObjectName(_fromUtf8("uvw_text_label"))
        self.gridLayout.addWidget(self.uvw_text_label, 4, 0, 1, 1)
        self.pushButton_to_uvw = QtGui.QPushButton(self.layoutWidget)
        self.pushButton_to_uvw.setObjectName(_fromUtf8("pushButton_to_uvw"))
        self.gridLayout.addWidget(self.pushButton_to_uvw, 2, 0, 1, 1)
        self.uvw_label = QtGui.QLabel(self.layoutWidget)
        self.uvw_label.setText(_fromUtf8(""))
        self.uvw_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.uvw_label.setObjectName(_fromUtf8("uvw_label"))
        self.gridLayout.addWidget(self.uvw_label, 3, 0, 1, 1)
        self.hkl_entry = QtGui.QLineEdit(self.layoutWidget)
        self.hkl_entry.setObjectName(_fromUtf8("hkl_entry"))
        self.gridLayout.addWidget(self.hkl_entry, 1, 0, 1, 1)
        self.layoutWidget.raise_()
        self.uvw_entry.raise_()

        self.retranslateUi(hkl_uvw)
        QtCore.QMetaObject.connectSlotsByName(hkl_uvw)
        hkl_uvw.setTabOrder(self.hkl_entry, self.pushButton_to_uvw)
        hkl_uvw.setTabOrder(self.pushButton_to_uvw, self.uvw_entry)
        hkl_uvw.setTabOrder(self.uvw_entry, self.pushButton_to_hkl)

    def retranslateUi(self, hkl_uvw):
        hkl_uvw.setWindowTitle(_translate("hkl_uvw", "hkl-uvw", None))
        self.pushButton_to_hkl.setText(_translate("hkl_uvw", "To hkl", None))
        self.hkl_text_label.setText(_translate("hkl_uvw", "hkl", None))
        self.uvw_text_label.setText(_translate("hkl_uvw", "uvw", None))
        self.pushButton_to_uvw.setText(_translate("hkl_uvw", "To uvw", None))

