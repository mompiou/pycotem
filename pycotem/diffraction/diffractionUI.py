# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'diffractionUI.ui'
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

class Ui_Diffraction(object):
    def setupUi(self, Diffraction):
        Diffraction.setObjectName(_fromUtf8("Diffraction"))
        Diffraction.resize(1200, 782)
        self.centralwidget = QtGui.QWidget(Diffraction)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.mplvl = QtGui.QGridLayout()
        self.mplvl.setObjectName(_fromUtf8("mplvl"))
        self.gridLayout_2.addLayout(self.mplvl, 0, 0, 1, 1)
        self.groupBox = QtGui.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(350, 16777215))
        self.groupBox.setTitle(_fromUtf8(""))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.alpha_signBox = QtGui.QCheckBox(self.groupBox)
        self.alpha_signBox.setText(_fromUtf8(""))
        self.alpha_signBox.setObjectName(_fromUtf8("alpha_signBox"))
        self.gridLayout.addWidget(self.alpha_signBox, 12, 2, 1, 1)
        self.Button_reset = QtGui.QPushButton(self.groupBox)
        self.Button_reset.setObjectName(_fromUtf8("Button_reset"))
        self.gridLayout.addWidget(self.Button_reset, 7, 0, 1, 5)
        self.Calib_label = QtGui.QLabel(self.groupBox)
        self.Calib_label.setObjectName(_fromUtf8("Calib_label"))
        self.gridLayout.addWidget(self.Calib_label, 5, 0, 1, 2)
        self.label_4 = QtGui.QLabel(self.groupBox)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 15, 0, 1, 2)
        self.alpha_label_2 = QtGui.QLabel(self.groupBox)
        self.alpha_label_2.setObjectName(_fromUtf8("alpha_label_2"))
        self.gridLayout.addWidget(self.alpha_label_2, 12, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 8, 5, 1, 4)
        self.alpha_label = QtGui.QLabel(self.groupBox)
        self.alpha_label.setObjectName(_fromUtf8("alpha_label"))
        self.gridLayout.addWidget(self.alpha_label, 1, 0, 1, 1)
        self.abc_label = QtGui.QLabel(self.groupBox)
        self.abc_label.setObjectName(_fromUtf8("abc_label"))
        self.gridLayout.addWidget(self.abc_label, 0, 0, 1, 1)
        self.n_entry = QtGui.QLineEdit(self.groupBox)
        self.n_entry.setObjectName(_fromUtf8("n_entry"))
        self.gridLayout.addWidget(self.n_entry, 3, 5, 1, 4)
        self.indice_label = QtGui.QLabel(self.groupBox)
        self.indice_label.setObjectName(_fromUtf8("indice_label"))
        self.gridLayout.addWidget(self.indice_label, 2, 0, 1, 2)
        self.orientation_button = QtGui.QPushButton(self.groupBox)
        self.orientation_button.setObjectName(_fromUtf8("orientation_button"))
        self.gridLayout.addWidget(self.orientation_button, 17, 6, 1, 5)
        self.space_group_label = QtGui.QLabel(self.groupBox)
        self.space_group_label.setObjectName(_fromUtf8("space_group_label"))
        self.gridLayout.addWidget(self.space_group_label, 4, 0, 1, 3)
        self.label_3 = QtGui.QLabel(self.groupBox)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 11, 0, 1, 5)
        self.euler_listbox = QtGui.QListWidget(self.groupBox)
        self.euler_listbox.setObjectName(_fromUtf8("euler_listbox"))
        self.gridLayout.addWidget(self.euler_listbox, 19, 0, 1, 11)
        self.Button_reset_all = QtGui.QPushButton(self.groupBox)
        self.Button_reset_all.setObjectName(_fromUtf8("Button_reset_all"))
        self.gridLayout.addWidget(self.Button_reset_all, 7, 6, 1, 5)
        self.diff_spot_Listbox = QtGui.QListWidget(self.groupBox)
        self.diff_spot_Listbox.setObjectName(_fromUtf8("diff_spot_Listbox"))
        self.gridLayout.addWidget(self.diff_spot_Listbox, 16, 0, 1, 11)
        self.abc_entry = QtGui.QLineEdit(self.groupBox)
        self.abc_entry.setObjectName(_fromUtf8("abc_entry"))
        self.gridLayout.addWidget(self.abc_entry, 0, 5, 1, 6)
        self.tilt_b_entry = QtGui.QLineEdit(self.groupBox)
        self.tilt_b_entry.setObjectName(_fromUtf8("tilt_b_entry"))
        self.gridLayout.addWidget(self.tilt_b_entry, 12, 5, 1, 1)
        self.tilt_axis_angle_label = QtGui.QLabel(self.groupBox)
        self.tilt_axis_angle_label.setObjectName(_fromUtf8("tilt_axis_angle_label"))
        self.gridLayout.addWidget(self.tilt_axis_angle_label, 6, 0, 1, 6)
        self.ListBox_d_2 = QtGui.QListWidget(self.groupBox)
        self.ListBox_d_2.setObjectName(_fromUtf8("ListBox_d_2"))
        self.gridLayout.addWidget(self.ListBox_d_2, 9, 0, 1, 5)
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 8, 0, 1, 5)
        self.tilt_a_entry = QtGui.QLineEdit(self.groupBox)
        self.tilt_a_entry.setObjectName(_fromUtf8("tilt_a_entry"))
        self.gridLayout.addWidget(self.tilt_a_entry, 12, 1, 1, 1)
        self.distance_button = QtGui.QPushButton(self.groupBox)
        self.distance_button.setObjectName(_fromUtf8("distance_button"))
        self.gridLayout.addWidget(self.distance_button, 10, 0, 1, 6)
        self.SpaceGroup_box = QtGui.QComboBox(self.groupBox)
        self.SpaceGroup_box.setObjectName(_fromUtf8("SpaceGroup_box"))
        self.gridLayout.addWidget(self.SpaceGroup_box, 4, 4, 1, 7)
        self.indice_entry = QtGui.QLineEdit(self.groupBox)
        self.indice_entry.setObjectName(_fromUtf8("indice_entry"))
        self.gridLayout.addWidget(self.indice_entry, 2, 5, 1, 4)
        self.ListBox_theo = QtGui.QListWidget(self.groupBox)
        self.ListBox_theo.setObjectName(_fromUtf8("ListBox_theo"))
        self.gridLayout.addWidget(self.ListBox_theo, 9, 5, 1, 6)
        self.beta_label = QtGui.QLabel(self.groupBox)
        self.beta_label.setObjectName(_fromUtf8("beta_label"))
        self.gridLayout.addWidget(self.beta_label, 12, 3, 1, 1)
        self.beta_signBox = QtGui.QCheckBox(self.groupBox)
        self.beta_signBox.setText(_fromUtf8(""))
        self.beta_signBox.setObjectName(_fromUtf8("beta_signBox"))
        self.gridLayout.addWidget(self.beta_signBox, 12, 6, 1, 1)
        self.tilt_axis_angle_entry = QtGui.QLineEdit(self.groupBox)
        self.tilt_axis_angle_entry.setObjectName(_fromUtf8("tilt_axis_angle_entry"))
        self.gridLayout.addWidget(self.tilt_axis_angle_entry, 6, 6, 1, 3)
        self.add_spot_button = QtGui.QPushButton(self.groupBox)
        self.add_spot_button.setObjectName(_fromUtf8("add_spot_button"))
        self.gridLayout.addWidget(self.add_spot_button, 17, 0, 1, 3)
        self.n_label = QtGui.QLabel(self.groupBox)
        self.n_label.setObjectName(_fromUtf8("n_label"))
        self.gridLayout.addWidget(self.n_label, 3, 0, 1, 5)
        self.Calib_box = QtGui.QComboBox(self.groupBox)
        self.Calib_box.setObjectName(_fromUtf8("Calib_box"))
        self.gridLayout.addWidget(self.Calib_box, 5, 4, 1, 7)
        self.alphabetagamma_entry = QtGui.QLineEdit(self.groupBox)
        self.alphabetagamma_entry.setObjectName(_fromUtf8("alphabetagamma_entry"))
        self.gridLayout.addWidget(self.alphabetagamma_entry, 1, 5, 1, 6)
        self.z_label = QtGui.QLabel(self.groupBox)
        self.z_label.setObjectName(_fromUtf8("z_label"))
        self.gridLayout.addWidget(self.z_label, 12, 7, 1, 1)
        self.tilt_z_entry = QtGui.QLineEdit(self.groupBox)
        self.tilt_z_entry.setObjectName(_fromUtf8("tilt_z_entry"))
        self.gridLayout.addWidget(self.tilt_z_entry, 12, 8, 1, 1)
        self.theta_signBox = QtGui.QCheckBox(self.groupBox)
        self.theta_signBox.setText(_fromUtf8(""))
        self.theta_signBox.setObjectName(_fromUtf8("theta_signBox"))
        self.gridLayout.addWidget(self.theta_signBox, 12, 9, 1, 1)
        self.remove_spot_button = QtGui.QPushButton(self.groupBox)
        self.remove_spot_button.setObjectName(_fromUtf8("remove_spot_button"))
        self.gridLayout.addWidget(self.remove_spot_button, 17, 3, 1, 3)
        self.abc_label.raise_()
        self.indice_label.raise_()
        self.n_label.raise_()
        self.space_group_label.raise_()
        self.Calib_label.raise_()
        self.Button_reset.raise_()
        self.label.raise_()
        self.distance_button.raise_()
        self.label_3.raise_()
        self.diff_spot_Listbox.raise_()
        self.add_spot_button.raise_()
        self.ListBox_d_2.raise_()
        self.euler_listbox.raise_()
        self.alpha_label_2.raise_()
        self.tilt_a_entry.raise_()
        self.alpha_signBox.raise_()
        self.label_4.raise_()
        self.tilt_b_entry.raise_()
        self.beta_signBox.raise_()
        self.remove_spot_button.raise_()
        self.orientation_button.raise_()
        self.SpaceGroup_box.raise_()
        self.Calib_box.raise_()
        self.n_entry.raise_()
        self.indice_entry.raise_()
        self.tilt_axis_angle_entry.raise_()
        self.tilt_axis_angle_label.raise_()
        self.ListBox_theo.raise_()
        self.alpha_label.raise_()
        self.alphabetagamma_entry.raise_()
        self.abc_entry.raise_()
        self.Button_reset_all.raise_()
        self.label_2.raise_()
        self.beta_label.raise_()
        self.z_label.raise_()
        self.tilt_z_entry.raise_()
        self.theta_signBox.raise_()
        self.gridLayout_2.addWidget(self.groupBox, 0, 1, 1, 1)
        Diffraction.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(Diffraction)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuSave = QtGui.QMenu(self.menubar)
        self.menuSave.setObjectName(_fromUtf8("menuSave"))
        self.menuStructure = QtGui.QMenu(self.menubar)
        self.menuStructure.setObjectName(_fromUtf8("menuStructure"))
        self.menuSpectrum = QtGui.QMenu(self.menubar)
        self.menuSpectrum.setObjectName(_fromUtf8("menuSpectrum"))
        Diffraction.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(Diffraction)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Diffraction.setStatusBar(self.statusbar)
        self.actionSave_figure = QtGui.QAction(Diffraction)
        self.actionSave_figure.setObjectName(_fromUtf8("actionSave_figure"))
        self.actionCalculate_Schmid_factor = QtGui.QAction(Diffraction)
        self.actionCalculate_Schmid_factor.setObjectName(_fromUtf8("actionCalculate_Schmid_factor"))
        self.actionCalculate_spectrum = QtGui.QAction(Diffraction)
        self.actionCalculate_spectrum.setObjectName(_fromUtf8("actionCalculate_spectrum"))
        self.menuSave.addAction(self.actionSave_figure)
        self.menuSpectrum.addAction(self.actionCalculate_spectrum)
        self.menubar.addAction(self.menuSave.menuAction())
        self.menubar.addAction(self.menuStructure.menuAction())
        self.menubar.addAction(self.menuSpectrum.menuAction())

        self.retranslateUi(Diffraction)
        QtCore.QMetaObject.connectSlotsByName(Diffraction)
        Diffraction.setTabOrder(self.abc_entry, self.alphabetagamma_entry)
        Diffraction.setTabOrder(self.alphabetagamma_entry, self.indice_entry)
        Diffraction.setTabOrder(self.indice_entry, self.n_entry)
        Diffraction.setTabOrder(self.n_entry, self.SpaceGroup_box)
        Diffraction.setTabOrder(self.SpaceGroup_box, self.Calib_box)
        Diffraction.setTabOrder(self.Calib_box, self.tilt_axis_angle_entry)
        Diffraction.setTabOrder(self.tilt_axis_angle_entry, self.Button_reset)
        Diffraction.setTabOrder(self.Button_reset, self.Button_reset_all)
        Diffraction.setTabOrder(self.Button_reset_all, self.ListBox_d_2)
        Diffraction.setTabOrder(self.ListBox_d_2, self.ListBox_theo)
        Diffraction.setTabOrder(self.ListBox_theo, self.distance_button)
        Diffraction.setTabOrder(self.distance_button, self.tilt_a_entry)
        Diffraction.setTabOrder(self.tilt_a_entry, self.alpha_signBox)
        Diffraction.setTabOrder(self.alpha_signBox, self.tilt_b_entry)
        Diffraction.setTabOrder(self.tilt_b_entry, self.beta_signBox)
        Diffraction.setTabOrder(self.beta_signBox, self.tilt_z_entry)
        Diffraction.setTabOrder(self.tilt_z_entry, self.theta_signBox)
        Diffraction.setTabOrder(self.theta_signBox, self.diff_spot_Listbox)
        Diffraction.setTabOrder(self.diff_spot_Listbox, self.add_spot_button)
        Diffraction.setTabOrder(self.add_spot_button, self.remove_spot_button)
        Diffraction.setTabOrder(self.remove_spot_button, self.orientation_button)
        Diffraction.setTabOrder(self.orientation_button, self.euler_listbox)

    def retranslateUi(self, Diffraction):
        Diffraction.setWindowTitle(_translate("Diffraction", "Diffraction pattern", None))
        self.Button_reset.setText(_translate("Diffraction", "Reset points", None))
        self.Calib_label.setText(_translate("Diffraction", "Calibrations", None))
        self.label_4.setText(_translate("Diffraction", "<html><head/><body><p><span style=\" font-weight:600;\">Data</span></p></body></html>", None))
        self.alpha_label_2.setText(_translate("Diffraction", "α (°)", None))
        self.label_2.setText(_translate("Diffraction", "<html><head/><body><p><span style=\" font-weight:600;\">g vectors</span></p></body></html>", None))
        self.alpha_label.setText(_translate("Diffraction", "<html><head/><body><p>α,β,γ</p></body></html>", None))
        self.abc_label.setText(_translate("Diffraction", "a,b,c", None))
        self.indice_label.setText(_translate("Diffraction", "Max Indices", None))
        self.orientation_button.setText(_translate("Diffraction", "Get orientation", None))
        self.space_group_label.setText(_translate("Diffraction", "Space group", None))
        self.label_3.setText(_translate("Diffraction", "<html><head/><body><p><span style=\" font-weight:600;\">Tilt angles/AC</span></p></body></html>", None))
        self.Button_reset_all.setText(_translate("Diffraction", "Reset all", None))
        self.tilt_axis_angle_label.setText(_translate("Diffraction", "Diffraction-α tilt angle", None))
        self.label.setText(_translate("Diffraction", "<html><head/><body><p><span style=\" font-weight:600;\">Dist., inclination</span></p></body></html>", None))
        self.distance_button.setText(_translate("Diffraction", "Find diffraction spots", None))
        self.beta_label.setText(_translate("Diffraction", "β (°)", None))
        self.add_spot_button.setText(_translate("Diffraction", "Add data", None))
        self.n_label.setText(_translate("Diffraction", "Number of spots", None))
        self.z_label.setText(_translate("Diffraction", "z (°)", None))
        self.remove_spot_button.setText(_translate("Diffraction", "Remove", None))
        self.menuSave.setTitle(_translate("Diffraction", "Open", None))
        self.menuStructure.setTitle(_translate("Diffraction", "Structure", None))
        self.menuSpectrum.setTitle(_translate("Diffraction", "Spectrum", None))
        self.actionSave_figure.setText(_translate("Diffraction", "Open image", None))
        self.actionCalculate_Schmid_factor.setText(_translate("Diffraction", "calculate Schmid factor", None))
        self.actionCalculate_spectrum.setText(_translate("Diffraction", "Plot spectrum", None))

