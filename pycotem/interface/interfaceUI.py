# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interfaceUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import os
fn = os.path.join(os.path.dirname(__file__), "interface-icon.png")

class Ui_Interface(object):
    def setupUi(self, Interface):
        Interface.setObjectName("Interface")
        Interface.resize(1200, 755)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(fn), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Interface.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(Interface)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.mplvl = QtWidgets.QGridLayout()
        self.mplvl.setObjectName("mplvl")
        self.gridLayout_7.addLayout(self.mplvl, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(400, 16777215))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.Button_reset = QtWidgets.QPushButton(self.groupBox)
        self.Button_reset.setObjectName("Button_reset")
        self.gridLayout.addWidget(self.Button_reset, 37, 0, 1, 3)
        self.alphabetagamma_label = QtWidgets.QLabel(self.groupBox)
        self.alphabetagamma_label.setObjectName("alphabetagamma_label")
        self.gridLayout.addWidget(self.alphabetagamma_label, 2, 0, 1, 1)
        self.abc_label = QtWidgets.QLabel(self.groupBox)
        self.abc_label.setObjectName("abc_label")
        self.gridLayout.addWidget(self.abc_label, 0, 0, 1, 1)
        self.magnification_entry = QtWidgets.QLineEdit(self.groupBox)
        self.magnification_entry.setObjectName("magnification_entry")
        self.gridLayout.addWidget(self.magnification_entry, 24, 2, 1, 2)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 26, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 8, 0, 1, 1)
        self.remove_condition_button = QtWidgets.QPushButton(self.groupBox)
        self.remove_condition_button.setObjectName("remove_condition_button")
        self.gridLayout.addWidget(self.remove_condition_button, 27, 3, 1, 1)
        self.conditions_Listbox = QtWidgets.QListWidget(self.groupBox)
        self.conditions_Listbox.setMaximumSize(QtCore.QSize(16777215, 100))
        self.conditions_Listbox.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.conditions_Listbox.setObjectName("conditions_Listbox")
        self.gridLayout.addWidget(self.conditions_Listbox, 28, 0, 1, 5)
        self.z_label = QtWidgets.QLabel(self.groupBox)
        self.z_label.setObjectName("z_label")
        self.gridLayout.addWidget(self.z_label, 23, 0, 1, 1)
        self.tilt_y_entry = QtWidgets.QLineEdit(self.groupBox)
        self.tilt_y_entry.setObjectName("tilt_y_entry")
        self.gridLayout.addWidget(self.tilt_y_entry, 19, 2, 1, 2)
        self.add_condition_button = QtWidgets.QPushButton(self.groupBox)
        self.add_condition_button.setObjectName("add_condition_button")
        self.gridLayout.addWidget(self.add_condition_button, 27, 0, 1, 3)
        self.tilt_z_entry = QtWidgets.QLineEdit(self.groupBox)
        self.tilt_z_entry.setObjectName("tilt_z_entry")
        self.gridLayout.addWidget(self.tilt_z_entry, 23, 2, 1, 2)
        self.double_button = QtWidgets.QRadioButton(self.groupBox)
        self.double_button.setObjectName("double_button")
        self.gridLayout.addWidget(self.double_button, 15, 3, 1, 1)
        self.image_angle_entry = QtWidgets.QLineEdit(self.groupBox)
        self.image_angle_entry.setObjectName("image_angle_entry")
        self.gridLayout.addWidget(self.image_angle_entry, 12, 3, 1, 2)
        self.abc_entry = QtWidgets.QLineEdit(self.groupBox)
        self.abc_entry.setObjectName("abc_entry")
        self.gridLayout.addWidget(self.abc_entry, 0, 2, 1, 3)
        self.euler_label = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.euler_label.setFont(font)
        self.euler_label.setObjectName("euler_label")
        self.gridLayout.addWidget(self.euler_label, 3, 0, 1, 1)
        self.image_angle_label = QtWidgets.QLabel(self.groupBox)
        self.image_angle_label.setObjectName("image_angle_label")
        self.gridLayout.addWidget(self.image_angle_label, 12, 0, 1, 3)
        self.alphabetagamma_entry = QtWidgets.QLineEdit(self.groupBox)
        self.alphabetagamma_entry.setObjectName("alphabetagamma_entry")
        self.gridLayout.addWidget(self.alphabetagamma_entry, 2, 2, 1, 3)
        self.micro_box = QtWidgets.QComboBox(self.groupBox)
        self.micro_box.setObjectName("micro_box")
        self.gridLayout.addWidget(self.micro_box, 8, 2, 1, 3)
        self.pxtonm_label = QtWidgets.QLabel(self.groupBox)
        self.pxtonm_label.setObjectName("pxtonm_label")
        self.gridLayout.addWidget(self.pxtonm_label, 24, 0, 1, 1)
        self.reset_view_button = QtWidgets.QPushButton(self.groupBox)
        self.reset_view_button.setObjectName("reset_view_button")
        self.gridLayout.addWidget(self.reset_view_button, 37, 3, 1, 1)
        self.tilt_rot_button = QtWidgets.QRadioButton(self.groupBox)
        self.tilt_rot_button.setObjectName("tilt_rot_button")
        self.gridLayout.addWidget(self.tilt_rot_button, 15, 4, 1, 1)
        self.beta_label_2 = QtWidgets.QLabel(self.groupBox)
        self.beta_label_2.setObjectName("beta_label_2")
        self.gridLayout.addWidget(self.beta_label_2, 21, 0, 1, 1)
        self.crystal_checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.crystal_checkBox.setObjectName("crystal_checkBox")
        self.gridLayout.addWidget(self.crystal_checkBox, 7, 0, 1, 5)
        self.euler_entry = QtWidgets.QLineEdit(self.groupBox)
        self.euler_entry.setObjectName("euler_entry")
        self.gridLayout.addWidget(self.euler_entry, 3, 2, 1, 3)
        self.normal_button = QtWidgets.QPushButton(self.groupBox)
        self.normal_button.setObjectName("normal_button")
        self.gridLayout.addWidget(self.normal_button, 39, 0, 1, 5)
        self.euler_Listbox = QtWidgets.QListWidget(self.groupBox)
        self.euler_Listbox.setMaximumSize(QtCore.QSize(16777215, 100))
        self.euler_Listbox.setObjectName("euler_Listbox")
        self.gridLayout.addWidget(self.euler_Listbox, 40, 0, 1, 5)
        self.theta_signBox = QtWidgets.QCheckBox(self.groupBox)
        self.theta_signBox.setText("")
        self.theta_signBox.setObjectName("theta_signBox")
        self.gridLayout.addWidget(self.theta_signBox, 23, 4, 1, 1)
        self.single_button = QtWidgets.QRadioButton(self.groupBox)
        self.single_button.setObjectName("single_button")
        self.gridLayout.addWidget(self.single_button, 15, 0, 1, 1)
        self.alpha_signBox = QtWidgets.QCheckBox(self.groupBox)
        self.alpha_signBox.setObjectName("alpha_signBox")
        self.gridLayout.addWidget(self.alpha_signBox, 19, 4, 1, 1)
        self.beta_signBox = QtWidgets.QCheckBox(self.groupBox)
        self.beta_signBox.setText("")
        self.beta_signBox.setObjectName("beta_signBox")
        self.gridLayout.addWidget(self.beta_signBox, 21, 4, 1, 1)
        self.tilt_x_entry = QtWidgets.QLineEdit(self.groupBox)
        self.tilt_x_entry.setObjectName("tilt_x_entry")
        self.gridLayout.addWidget(self.tilt_x_entry, 21, 2, 1, 2)
        self.alpha_label_2 = QtWidgets.QLabel(self.groupBox)
        self.alpha_label_2.setObjectName("alpha_label_2")
        self.gridLayout.addWidget(self.alpha_label_2, 19, 0, 1, 1)
        self.direction_button = QtWidgets.QCheckBox(self.groupBox)
        self.direction_button.setObjectName("direction_button")
        self.gridLayout.addWidget(self.direction_button, 26, 2, 1, 2)
        self.gridLayout_7.addWidget(self.groupBox, 0, 1, 1, 1)
        Interface.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Interface)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 22))
        self.menubar.setObjectName("menubar")
        self.menuSave = QtWidgets.QMenu(self.menubar)
        self.menuSave.setObjectName("menuSave")
        self.menuStructure = QtWidgets.QMenu(self.menubar)
        self.menuStructure.setObjectName("menuStructure")
        self.menuDraw = QtWidgets.QMenu(self.menubar)
        self.menuDraw.setObjectName("menuDraw")
        self.menuData = QtWidgets.QMenu(self.menubar)
        self.menuData.setObjectName("menuData")
        Interface.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Interface)
        self.statusbar.setObjectName("statusbar")
        Interface.setStatusBar(self.statusbar)
        self.actionSave_figure = QtWidgets.QAction(Interface)
        self.actionSave_figure.setObjectName("actionSave_figure")
        self.actionCalculate_Schmid_factor = QtWidgets.QAction(Interface)
        self.actionCalculate_Schmid_factor.setObjectName("actionCalculate_Schmid_factor")
        self.actionCalculate_spectrum = QtWidgets.QAction(Interface)
        self.actionCalculate_spectrum.setObjectName("actionCalculate_spectrum")
        self.actionDirections_planes = QtWidgets.QAction(Interface)
        self.actionDirections_planes.setObjectName("actionDirections_planes")
        self.actionImport = QtWidgets.QAction(Interface)
        self.actionImport.setObjectName("actionImport")
        self.actionExport = QtWidgets.QAction(Interface)
        self.actionExport.setObjectName("actionExport")
        self.menuSave.addAction(self.actionSave_figure)
        self.menuDraw.addAction(self.actionDirections_planes)
        self.menuData.addAction(self.actionImport)
        self.menuData.addAction(self.actionExport)
        self.menubar.addAction(self.menuSave.menuAction())
        self.menubar.addAction(self.menuStructure.menuAction())
        self.menubar.addAction(self.menuDraw.menuAction())
        self.menubar.addAction(self.menuData.menuAction())

        self.retranslateUi(Interface)
        QtCore.QMetaObject.connectSlotsByName(Interface)
        Interface.setTabOrder(self.abc_entry, self.alphabetagamma_entry)
        Interface.setTabOrder(self.alphabetagamma_entry, self.euler_entry)
        Interface.setTabOrder(self.euler_entry, self.crystal_checkBox)
        Interface.setTabOrder(self.crystal_checkBox, self.micro_box)
        Interface.setTabOrder(self.micro_box, self.image_angle_entry)
        Interface.setTabOrder(self.image_angle_entry, self.single_button)
        Interface.setTabOrder(self.single_button, self.double_button)
        Interface.setTabOrder(self.double_button, self.tilt_rot_button)
        Interface.setTabOrder(self.tilt_rot_button, self.tilt_y_entry)
        Interface.setTabOrder(self.tilt_y_entry, self.alpha_signBox)
        Interface.setTabOrder(self.alpha_signBox, self.tilt_x_entry)
        Interface.setTabOrder(self.tilt_x_entry, self.beta_signBox)
        Interface.setTabOrder(self.beta_signBox, self.tilt_z_entry)
        Interface.setTabOrder(self.tilt_z_entry, self.theta_signBox)
        Interface.setTabOrder(self.theta_signBox, self.magnification_entry)
        Interface.setTabOrder(self.magnification_entry, self.direction_button)
        Interface.setTabOrder(self.direction_button, self.add_condition_button)
        Interface.setTabOrder(self.add_condition_button, self.remove_condition_button)
        Interface.setTabOrder(self.remove_condition_button, self.conditions_Listbox)
        Interface.setTabOrder(self.conditions_Listbox, self.Button_reset)
        Interface.setTabOrder(self.Button_reset, self.reset_view_button)
        Interface.setTabOrder(self.reset_view_button, self.normal_button)
        Interface.setTabOrder(self.normal_button, self.euler_Listbox)

    def retranslateUi(self, Interface):
        _translate = QtCore.QCoreApplication.translate
        Interface.setWindowTitle(_translate("Interface", "interface determination"))
        self.Button_reset.setText(_translate("Interface", "Reset points"))
        self.alphabetagamma_label.setText(_translate("Interface", "<html><head/><body><p>α, β, γ</p></body></html>"))
        self.abc_label.setText(_translate("Interface", "a,b,c"))
        self.label.setText(_translate("Interface", "<html><head/><body><p><span style=\" font-weight:600;\">Measure</span></p></body></html>"))
        self.label_2.setText(_translate("Interface", "<html><head/><body><p><span style=\" font-weight:600;\">Microscope</span></p></body></html>"))
        self.remove_condition_button.setText(_translate("Interface", "Remove"))
        self.z_label.setText(_translate("Interface", "θ  (°)"))
        self.add_condition_button.setText(_translate("Interface", "Add data"))
        self.double_button.setText(_translate("Interface", "double (α,β)"))
        self.euler_label.setText(_translate("Interface", "<html><head/><body><p><span style=\" font-weight:400;\">Euler angles</span></p></body></html>"))
        self.image_angle_label.setText(_translate("Interface", "<html><head/><body><p>Image-α tilt angle</p></body></html>"))
        self.pxtonm_label.setText(_translate("Interface", "Magnification"))
        self.reset_view_button.setText(_translate("Interface", "Reset view"))
        self.tilt_rot_button.setText(_translate("Interface", "tilt rot (α,θ)"))
        self.beta_label_2.setText(_translate("Interface", "<html><head/><body><p>β (°)</p></body></html>"))
        self.crystal_checkBox.setText(_translate("Interface", "Use lattice coordinates"))
        self.normal_button.setText(_translate("Interface", "Get normal/direction"))
        self.single_button.setText(_translate("Interface", "single (α)"))
        self.alpha_signBox.setText(_translate("Interface", "AC"))
        self.alpha_label_2.setText(_translate("Interface", "<html><head/><body><p>α (°)</p></body></html>"))
        self.direction_button.setText(_translate("Interface", "direction"))
        self.menuSave.setTitle(_translate("Interface", "Open"))
        self.menuStructure.setTitle(_translate("Interface", "Structure"))
        self.menuDraw.setTitle(_translate("Interface", "Draw"))
        self.menuData.setTitle(_translate("Interface", "Data"))
        self.actionSave_figure.setText(_translate("Interface", "Open image"))
        self.actionCalculate_Schmid_factor.setText(_translate("Interface", "calculate Schmid factor"))
        self.actionCalculate_spectrum.setText(_translate("Interface", "Plot spectrum"))
        self.actionDirections_planes.setText(_translate("Interface", "directions/planes"))
        self.actionImport.setText(_translate("Interface", "Import"))
        self.actionExport.setText(_translate("Interface", "Export"))
