# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'crystalUI.ui'
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

class Ui_Crystal(object):
    def setupUi(self, Crystal):
        Crystal.setObjectName(_fromUtf8("Crystal"))
        Crystal.resize(1222, 796)
        self.centralwidget = QtGui.QWidget(Crystal)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_11 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_11.setObjectName(_fromUtf8("gridLayout_11"))
        self.mplvl = QtGui.QGridLayout()
        self.mplvl.setObjectName(_fromUtf8("mplvl"))
        self.gridLayout_11.addLayout(self.mplvl, 0, 0, 1, 1)
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(400, 16777215))
        self.groupBox.setTitle(_fromUtf8(""))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.groupBox_2 = QtGui.QGroupBox(self.groupBox)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.alphabetagamma_label = QtGui.QLabel(self.groupBox_2)
        self.alphabetagamma_label.setObjectName(_fromUtf8("alphabetagamma_label"))
        self.gridLayout_2.addWidget(self.alphabetagamma_label, 1, 0, 1, 1)
        self.alphabetagamma_entry = QtGui.QLineEdit(self.groupBox_2)
        self.alphabetagamma_entry.setObjectName(_fromUtf8("alphabetagamma_entry"))
        self.gridLayout_2.addWidget(self.alphabetagamma_entry, 1, 1, 1, 1)
        self.abc_entry = QtGui.QLineEdit(self.groupBox_2)
        self.abc_entry.setObjectName(_fromUtf8("abc_entry"))
        self.gridLayout_2.addWidget(self.abc_entry, 0, 1, 1, 1)
        self.abc_label = QtGui.QLabel(self.groupBox_2)
        self.abc_label.setObjectName(_fromUtf8("abc_label"))
        self.gridLayout_2.addWidget(self.abc_label, 0, 0, 1, 1)
        self.changeStruct_button = QtGui.QPushButton(self.groupBox_2)
        self.changeStruct_button.setObjectName(_fromUtf8("changeStruct_button"))
        self.gridLayout_2.addWidget(self.changeStruct_button, 2, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.groupBox_3 = QtGui.QGroupBox(self.groupBox)
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.gridLayout_3 = QtGui.QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.label_2 = QtGui.QLabel(self.groupBox_3)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_3.addWidget(self.label_2, 2, 0, 1, 1)
        self.label = QtGui.QLabel(self.groupBox_3)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        self.layers_entry = QtGui.QLineEdit(self.groupBox_3)
        self.layers_entry.setObjectName(_fromUtf8("layers_entry"))
        self.gridLayout_3.addWidget(self.layers_entry, 2, 2, 1, 1)
        self.plane_entry = QtGui.QLineEdit(self.groupBox_3)
        self.plane_entry.setObjectName(_fromUtf8("plane_entry"))
        self.gridLayout_3.addWidget(self.plane_entry, 0, 1, 1, 2)
        self.size_entry = QtGui.QLineEdit(self.groupBox_3)
        self.size_entry.setObjectName(_fromUtf8("size_entry"))
        self.gridLayout_3.addWidget(self.size_entry, 1, 2, 1, 1)
        self.label_3 = QtGui.QLabel(self.groupBox_3)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_3, 0, 1, 1, 1)
        self.groupBox_4 = QtGui.QGroupBox(self.groupBox)
        self.groupBox_4.setTitle(_fromUtf8(""))
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.gridLayout_4 = QtGui.QGridLayout(self.groupBox_4)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.label_6 = QtGui.QLabel(self.groupBox_4)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout_4.addWidget(self.label_6, 2, 0, 1, 1)
        self.markers_entry = QtGui.QLineEdit(self.groupBox_4)
        self.markers_entry.setObjectName(_fromUtf8("markers_entry"))
        self.gridLayout_4.addWidget(self.markers_entry, 2, 1, 1, 1)
        self.square_box = QtGui.QCheckBox(self.groupBox_4)
        self.square_box.setObjectName(_fromUtf8("square_box"))
        self.gridLayout_4.addWidget(self.square_box, 3, 0, 1, 2)
        self.labels_box = QtGui.QCheckBox(self.groupBox_4)
        self.labels_box.setObjectName(_fromUtf8("labels_box"))
        self.gridLayout_4.addWidget(self.labels_box, 4, 0, 1, 2)
        self.atoms_box = QtGui.QCheckBox(self.groupBox_4)
        self.atoms_box.setObjectName(_fromUtf8("atoms_box"))
        self.gridLayout_4.addWidget(self.atoms_box, 5, 0, 1, 2)
        self.calculate_button = QtGui.QPushButton(self.groupBox_4)
        self.calculate_button.setObjectName(_fromUtf8("calculate_button"))
        self.gridLayout_4.addWidget(self.calculate_button, 0, 0, 1, 2)
        self.draw_button = QtGui.QPushButton(self.groupBox_4)
        self.draw_button.setObjectName(_fromUtf8("draw_button"))
        self.gridLayout_4.addWidget(self.draw_button, 1, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox_4, 1, 0, 1, 1)
        self.groupBox_5 = QtGui.QGroupBox(self.groupBox)
        self.groupBox_5.setObjectName(_fromUtf8("groupBox_5"))
        self.gridLayout_5 = QtGui.QGridLayout(self.groupBox_5)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.label_5 = QtGui.QLabel(self.groupBox_5)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_5.addWidget(self.label_5, 5, 0, 1, 1)
        self.dsc_box = QtGui.QCheckBox(self.groupBox_5)
        self.dsc_box.setObjectName(_fromUtf8("dsc_box"))
        self.gridLayout_5.addWidget(self.dsc_box, 0, 0, 1, 1)
        self.label_4 = QtGui.QLabel(self.groupBox_5)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_5.addWidget(self.label_4, 1, 0, 1, 1)
        self.angle_entry = QtGui.QLineEdit(self.groupBox_5)
        self.angle_entry.setObjectName(_fromUtf8("angle_entry"))
        self.gridLayout_5.addWidget(self.angle_entry, 1, 1, 1, 1)
        self.coincidence_box = QtGui.QCheckBox(self.groupBox_5)
        self.coincidence_box.setObjectName(_fromUtf8("coincidence_box"))
        self.gridLayout_5.addWidget(self.coincidence_box, 3, 0, 1, 2)
        self.layers_box = QtGui.QCheckBox(self.groupBox_5)
        self.layers_box.setObjectName(_fromUtf8("layers_box"))
        self.gridLayout_5.addWidget(self.layers_box, 4, 0, 1, 2)
        self.precision_entry = QtGui.QLineEdit(self.groupBox_5)
        self.precision_entry.setObjectName(_fromUtf8("precision_entry"))
        self.gridLayout_5.addWidget(self.precision_entry, 5, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_5, 1, 1, 1, 1)
        self.gridLayout_11.addWidget(self.groupBox, 0, 1, 1, 1)
        Crystal.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(Crystal)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1222, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuSave = QtGui.QMenu(self.menubar)
        self.menuSave.setObjectName(_fromUtf8("menuSave"))
        self.menuStructure = QtGui.QMenu(self.menubar)
        self.menuStructure.setObjectName(_fromUtf8("menuStructure"))
        Crystal.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(Crystal)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Crystal.setStatusBar(self.statusbar)
        self.actionSave_figure = QtGui.QAction(Crystal)
        self.actionSave_figure.setObjectName(_fromUtf8("actionSave_figure"))
        self.actionCalculate_Schmid_factor = QtGui.QAction(Crystal)
        self.actionCalculate_Schmid_factor.setObjectName(_fromUtf8("actionCalculate_Schmid_factor"))
        self.actionCalculate_angle = QtGui.QAction(Crystal)
        self.actionCalculate_angle.setObjectName(_fromUtf8("actionCalculate_angle"))
        self.actionCalculate_xyz = QtGui.QAction(Crystal)
        self.actionCalculate_xyz.setObjectName(_fromUtf8("actionCalculate_xyz"))
        self.actionCalculate_apparent_width = QtGui.QAction(Crystal)
        self.actionCalculate_apparent_width.setObjectName(_fromUtf8("actionCalculate_apparent_width"))
        self.actionPlanes = QtGui.QAction(Crystal)
        self.actionPlanes.setObjectName(_fromUtf8("actionPlanes"))
        self.actionProj_directions = QtGui.QAction(Crystal)
        self.actionProj_directions.setObjectName(_fromUtf8("actionProj_directions"))
        self.actionPlane_cone = QtGui.QAction(Crystal)
        self.actionPlane_cone.setObjectName(_fromUtf8("actionPlane_cone"))
        self.actionCalculate_intersections = QtGui.QAction(Crystal)
        self.actionCalculate_intersections.setObjectName(_fromUtf8("actionCalculate_intersections"))
        self.actionHkl_uvw = QtGui.QAction(Crystal)
        self.actionHkl_uvw.setObjectName(_fromUtf8("actionHkl_uvw"))
        self.actionPlot_Kikuchi_lines = QtGui.QAction(Crystal)
        self.actionPlot_Kikuchi_lines.setObjectName(_fromUtf8("actionPlot_Kikuchi_lines"))
        self.menuSave.addAction(self.actionSave_figure)
        self.menubar.addAction(self.menuSave.menuAction())
        self.menubar.addAction(self.menuStructure.menuAction())

        self.retranslateUi(Crystal)
        QtCore.QMetaObject.connectSlotsByName(Crystal)
        Crystal.setTabOrder(self.abc_entry, self.alphabetagamma_entry)
        Crystal.setTabOrder(self.alphabetagamma_entry, self.plane_entry)
        Crystal.setTabOrder(self.plane_entry, self.size_entry)
        Crystal.setTabOrder(self.size_entry, self.layers_entry)
        Crystal.setTabOrder(self.layers_entry, self.changeStruct_button)
        Crystal.setTabOrder(self.changeStruct_button, self.calculate_button)
        Crystal.setTabOrder(self.calculate_button, self.draw_button)
        Crystal.setTabOrder(self.draw_button, self.markers_entry)
        Crystal.setTabOrder(self.markers_entry, self.square_box)
        Crystal.setTabOrder(self.square_box, self.labels_box)
        Crystal.setTabOrder(self.labels_box, self.atoms_box)
        Crystal.setTabOrder(self.atoms_box, self.dsc_box)
        Crystal.setTabOrder(self.dsc_box, self.angle_entry)
        Crystal.setTabOrder(self.angle_entry, self.coincidence_box)
        Crystal.setTabOrder(self.coincidence_box, self.layers_box)
        Crystal.setTabOrder(self.layers_box, self.precision_entry)

    def retranslateUi(self, Crystal):
        Crystal.setWindowTitle(_translate("Crystal", "Crystal", None))
        self.groupBox_2.setTitle(_translate("Crystal", "Crystal Parameters", None))
        self.alphabetagamma_label.setText(_translate("Crystal", "<p>&alpha;, &beta;, &gamma;</p>", None))
        self.abc_label.setText(_translate("Crystal", "a,b,c", None))
        self.changeStruct_button.setText(_translate("Crystal", "Change structure", None))
        self.groupBox_3.setTitle(_translate("Crystal", "Layers", None))
        self.label_2.setText(_translate("Crystal", "Layers numb.", None))
        self.label.setText(_translate("Crystal", "Plane", None))
        self.label_3.setText(_translate("Crystal", "Size", None))
        self.label_6.setText(_translate("Crystal", "Marker size", None))
        self.square_box.setText(_translate("Crystal", "Square/circle", None))
        self.labels_box.setText(_translate("Crystal", "Atom positions", None))
        self.atoms_box.setText(_translate("Crystal", "Atom type", None))
        self.calculate_button.setText(_translate("Crystal", "Calculate", None))
        self.draw_button.setText(_translate("Crystal", "Draw", None))
        self.groupBox_5.setTitle(_translate("Crystal", "Dichromatic", None))
        self.label_5.setText(_translate("Crystal", "Precision", None))
        self.dsc_box.setText(_translate("Crystal", "enable", None))
        self.label_4.setText(_translate("Crystal", " angle", None))
        self.coincidence_box.setText(_translate("Crystal", "coincidence", None))
        self.layers_box.setText(_translate("Crystal", "layers", None))
        self.menuSave.setTitle(_translate("Crystal", "Save", None))
        self.menuStructure.setTitle(_translate("Crystal", "Structure", None))
        self.actionSave_figure.setText(_translate("Crystal", "Save figure", None))
        self.actionCalculate_Schmid_factor.setText(_translate("Crystal", "calculate Schmid factor", None))
        self.actionCalculate_angle.setText(_translate("Crystal", "Calculate angle", None))
        self.actionCalculate_xyz.setText(_translate("Crystal", "calculate xyz directions", None))
        self.actionCalculate_apparent_width.setText(_translate("Crystal", "Calculate apparent width", None))
        self.actionPlanes.setText(_translate("Crystal", "planes", None))
        self.actionProj_directions.setText(_translate("Crystal", "proj. directions", None))
        self.actionPlane_cone.setText(_translate("Crystal", "plane-cone", None))
        self.actionCalculate_intersections.setText(_translate("Crystal", "Calculate intersections", None))
        self.actionHkl_uvw.setText(_translate("Crystal", "hkl <> uvw", None))
        self.actionPlot_Kikuchi_lines.setText(_translate("Crystal", "plot Kikuchi lines or diffraction pattern", None))
