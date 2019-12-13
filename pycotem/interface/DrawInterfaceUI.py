# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DrawInterfaceUI.ui'
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

class Ui_Draw_plane_directions(object):
    def setupUi(self, Draw_plane_directions):
        Draw_plane_directions.setObjectName(_fromUtf8("Draw_plane_directions"))
        Draw_plane_directions.resize(399, 234)
        self.gridLayoutWidget = QtGui.QWidget(Draw_plane_directions)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 10, 391, 221))
        self.gridLayoutWidget.setObjectName(_fromUtf8("gridLayoutWidget"))
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setMargin(0)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.dir_checkBox = QtGui.QCheckBox(self.gridLayoutWidget)
        self.dir_checkBox.setObjectName(_fromUtf8("dir_checkBox"))
        self.gridLayout.addWidget(self.dir_checkBox, 2, 2, 1, 1)
        self.surf_checkBox = QtGui.QCheckBox(self.gridLayoutWidget)
        self.surf_checkBox.setObjectName(_fromUtf8("surf_checkBox"))
        self.gridLayout.addWidget(self.surf_checkBox, 1, 2, 1, 1)
        self.plane_label = QtGui.QLabel(self.gridLayoutWidget)
        self.plane_label.setObjectName(_fromUtf8("plane_label"))
        self.gridLayout.addWidget(self.plane_label, 2, 0, 1, 1)
        self.surface_entry = QtGui.QLineEdit(self.gridLayoutWidget)
        self.surface_entry.setObjectName(_fromUtf8("surface_entry"))
        self.gridLayout.addWidget(self.surface_entry, 1, 1, 1, 1)
        self.plane_entry = QtGui.QLineEdit(self.gridLayoutWidget)
        self.plane_entry.setObjectName(_fromUtf8("plane_entry"))
        self.gridLayout.addWidget(self.plane_entry, 2, 1, 1, 1)
        self.surface_label = QtGui.QLabel(self.gridLayoutWidget)
        self.surface_label.setObjectName(_fromUtf8("surface_label"))
        self.gridLayout.addWidget(self.surface_label, 1, 0, 1, 1)
        self.thickness_entry = QtGui.QLineEdit(self.gridLayoutWidget)
        self.thickness_entry.setObjectName(_fromUtf8("thickness_entry"))
        self.gridLayout.addWidget(self.thickness_entry, 0, 1, 1, 1)
        self.thickness_label = QtGui.QLabel(self.gridLayoutWidget)
        self.thickness_label.setObjectName(_fromUtf8("thickness_label"))
        self.gridLayout.addWidget(self.thickness_label, 0, 0, 1, 1)
        self.buttonBox = QtGui.QDialogButtonBox(self.gridLayoutWidget)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.gridLayout.addWidget(self.buttonBox, 4, 0, 1, 3)
        self.label_checkBox = QtGui.QCheckBox(self.gridLayoutWidget)
        self.label_checkBox.setObjectName(_fromUtf8("label_checkBox"))
        self.gridLayout.addWidget(self.label_checkBox, 3, 0, 1, 1)

        self.retranslateUi(Draw_plane_directions)
        QtCore.QMetaObject.connectSlotsByName(Draw_plane_directions)
        Draw_plane_directions.setTabOrder(self.thickness_entry, self.surface_entry)
        Draw_plane_directions.setTabOrder(self.surface_entry, self.plane_entry)
        Draw_plane_directions.setTabOrder(self.plane_entry, self.surf_checkBox)
        Draw_plane_directions.setTabOrder(self.surf_checkBox, self.dir_checkBox)

    def retranslateUi(self, Draw_plane_directions):
        Draw_plane_directions.setWindowTitle(_translate("Draw_plane_directions", "Draw planes-directions", None))
        self.dir_checkBox.setText(_translate("Draw_plane_directions", "dir.", None))
        self.surf_checkBox.setText(_translate("Draw_plane_directions", "surf.", None))
        self.plane_label.setText(_translate("Draw_plane_directions", "Plane/Direction", None))
        self.surface_label.setText(_translate("Draw_plane_directions", "Surface norm.", None))
        self.thickness_label.setText(_translate("Draw_plane_directions", "Thickness (nm)", None))
        self.label_checkBox.setText(_translate("Draw_plane_directions", "label", None))

