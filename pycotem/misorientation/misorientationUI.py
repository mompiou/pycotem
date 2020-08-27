# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'misorientationUI.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Misorientation(object):
    def setupUi(self, Misorientation):
        Misorientation.setObjectName("Misorientation")
        Misorientation.resize(1212, 887)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("misorientation-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Misorientation.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(Misorientation)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.settings = QtWidgets.QWidget()
        self.settings.setObjectName("settings")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.settings)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.groupBox = QtWidgets.QGroupBox(self.settings)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.alphabetagamma_entry = QtWidgets.QLineEdit(self.groupBox)
        self.alphabetagamma_entry.setObjectName("alphabetagamma_entry")
        self.gridLayout_2.addWidget(self.alphabetagamma_entry, 2, 1, 1, 5)
        self.alphabetagamma_label = QtWidgets.QLabel(self.groupBox)
        self.alphabetagamma_label.setObjectName("alphabetagamma_label")
        self.gridLayout_2.addWidget(self.alphabetagamma_label, 2, 0, 1, 1)
        self.abc_label = QtWidgets.QLabel(self.groupBox)
        self.abc_label.setObjectName("abc_label")
        self.gridLayout_2.addWidget(self.abc_label, 1, 0, 1, 1)
        self.style_box = QtWidgets.QCheckBox(self.groupBox)
        self.style_box.setObjectName("style_box")
        self.gridLayout_2.addWidget(self.style_box, 8, 0, 1, 4)
        self.e_label = QtWidgets.QLabel(self.groupBox)
        self.e_label.setObjectName("e_label")
        self.gridLayout_2.addWidget(self.e_label, 3, 0, 1, 2)
        self.color_trace_rouge = QtWidgets.QRadioButton(self.groupBox)
        self.color_trace_rouge.setObjectName("color_trace_rouge")
        self.gridLayout_2.addWidget(self.color_trace_rouge, 6, 2, 1, 1)
        self.structure_box = QtWidgets.QComboBox(self.groupBox)
        self.structure_box.setObjectName("structure_box")
        self.gridLayout_2.addWidget(self.structure_box, 0, 0, 1, 6)
        self.d_label = QtWidgets.QLabel(self.groupBox)
        self.d_label.setObjectName("d_label")
        self.gridLayout_2.addWidget(self.d_label, 13, 0, 1, 1)
        self.abc_entry = QtWidgets.QLineEdit(self.groupBox)
        self.abc_entry.setObjectName("abc_entry")
        self.gridLayout_2.addWidget(self.abc_entry, 1, 1, 1, 5)
        self.color_trace_vert = QtWidgets.QRadioButton(self.groupBox)
        self.color_trace_vert.setObjectName("color_trace_vert")
        self.gridLayout_2.addWidget(self.color_trace_vert, 6, 1, 1, 1)
        self.color_trace_bleu = QtWidgets.QRadioButton(self.groupBox)
        self.color_trace_bleu.setObjectName("color_trace_bleu")
        self.gridLayout_2.addWidget(self.color_trace_bleu, 6, 0, 1, 1)
        self.e_entry = QtWidgets.QLineEdit(self.groupBox)
        self.e_entry.setObjectName("e_entry")
        self.gridLayout_2.addWidget(self.e_entry, 3, 3, 1, 3)
        self.hexa_button = QtWidgets.QCheckBox(self.groupBox)
        self.hexa_button.setObjectName("hexa_button")
        self.gridLayout_2.addWidget(self.hexa_button, 4, 0, 1, 6)
        self.d1_Slider = QtWidgets.QSlider(self.groupBox)
        self.d1_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.d1_Slider.setObjectName("d1_Slider")
        self.gridLayout_2.addWidget(self.d1_Slider, 13, 1, 1, 3)
        self.d_label_var = QtWidgets.QLabel(self.groupBox)
        self.d_label_var.setText("")
        self.d_label_var.setObjectName("d_label_var")
        self.gridLayout_2.addWidget(self.d_label_var, 13, 4, 1, 2)
        self.gridLayout_7.addWidget(self.groupBox, 0, 2, 2, 1)
        self.crystal2_box = QtWidgets.QGroupBox(self.settings)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.crystal2_box.sizePolicy().hasHeightForWidth())
        self.crystal2_box.setSizePolicy(sizePolicy)
        self.crystal2_box.setObjectName("crystal2_box")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.crystal2_box)
        self.gridLayout_4.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_4.setSpacing(5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.d_label_var_2 = QtWidgets.QLabel(self.crystal2_box)
        self.d_label_var_2.setText("")
        self.d_label_var_2.setObjectName("d_label_var_2")
        self.gridLayout_4.addWidget(self.d_label_var_2, 12, 5, 1, 2)
        self.style_box_2 = QtWidgets.QCheckBox(self.crystal2_box)
        self.style_box_2.setObjectName("style_box_2")
        self.gridLayout_4.addWidget(self.style_box_2, 8, 0, 1, 7)
        self.abc_label_2 = QtWidgets.QLabel(self.crystal2_box)
        self.abc_label_2.setObjectName("abc_label_2")
        self.gridLayout_4.addWidget(self.abc_label_2, 1, 0, 1, 1)
        self.color_trace_rouge_2 = QtWidgets.QRadioButton(self.crystal2_box)
        self.color_trace_rouge_2.setObjectName("color_trace_rouge_2")
        self.gridLayout_4.addWidget(self.color_trace_rouge_2, 7, 2, 1, 1)
        self.alphabetagamma_label_2 = QtWidgets.QLabel(self.crystal2_box)
        self.alphabetagamma_label_2.setObjectName("alphabetagamma_label_2")
        self.gridLayout_4.addWidget(self.alphabetagamma_label_2, 3, 0, 1, 1)
        self.e_label_2 = QtWidgets.QLabel(self.crystal2_box)
        self.e_label_2.setObjectName("e_label_2")
        self.gridLayout_4.addWidget(self.e_label_2, 5, 0, 1, 4)
        self.e_entry_2 = QtWidgets.QLineEdit(self.crystal2_box)
        self.e_entry_2.setObjectName("e_entry_2")
        self.gridLayout_4.addWidget(self.e_entry_2, 5, 4, 1, 3)
        self.structure2_box = QtWidgets.QComboBox(self.crystal2_box)
        self.structure2_box.setObjectName("structure2_box")
        self.gridLayout_4.addWidget(self.structure2_box, 0, 0, 1, 7)
        self.hexa_button_2 = QtWidgets.QCheckBox(self.crystal2_box)
        self.hexa_button_2.setObjectName("hexa_button_2")
        self.gridLayout_4.addWidget(self.hexa_button_2, 6, 0, 1, 7)
        self.color_trace_bleu_2 = QtWidgets.QRadioButton(self.crystal2_box)
        self.color_trace_bleu_2.setObjectName("color_trace_bleu_2")
        self.gridLayout_4.addWidget(self.color_trace_bleu_2, 7, 0, 1, 1)
        self.alphabetagamma_entry_2 = QtWidgets.QLineEdit(self.crystal2_box)
        self.alphabetagamma_entry_2.setObjectName("alphabetagamma_entry_2")
        self.gridLayout_4.addWidget(self.alphabetagamma_entry_2, 3, 1, 1, 6)
        self.abc_entry_2 = QtWidgets.QLineEdit(self.crystal2_box)
        self.abc_entry_2.setObjectName("abc_entry_2")
        self.gridLayout_4.addWidget(self.abc_entry_2, 1, 1, 1, 6)
        self.d_label_2 = QtWidgets.QLabel(self.crystal2_box)
        self.d_label_2.setObjectName("d_label_2")
        self.gridLayout_4.addWidget(self.d_label_2, 12, 0, 1, 1)
        self.color_trace_vert_2 = QtWidgets.QRadioButton(self.crystal2_box)
        self.color_trace_vert_2.setObjectName("color_trace_vert_2")
        self.gridLayout_4.addWidget(self.color_trace_vert_2, 7, 1, 1, 1)
        self.d2_Slider = QtWidgets.QSlider(self.crystal2_box)
        self.d2_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.d2_Slider.setObjectName("d2_Slider")
        self.gridLayout_4.addWidget(self.d2_Slider, 12, 1, 1, 4)
        self.gridLayout_7.addWidget(self.crystal2_box, 0, 3, 2, 1)
        self.groupBox_7 = QtWidgets.QGroupBox(self.settings)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.image_angle_entry = QtWidgets.QLineEdit(self.groupBox_7)
        self.image_angle_entry.setObjectName("image_angle_entry")
        self.gridLayout_12.addWidget(self.image_angle_entry, 1, 1, 1, 1)
        self.real_space_checkBox = QtWidgets.QCheckBox(self.groupBox_7)
        self.real_space_checkBox.setObjectName("real_space_checkBox")
        self.gridLayout_12.addWidget(self.real_space_checkBox, 2, 0, 1, 1)
        self.image_tilt_y_label = QtWidgets.QLabel(self.groupBox_7)
        self.image_tilt_y_label.setObjectName("image_tilt_y_label")
        self.gridLayout_12.addWidget(self.image_tilt_y_label, 1, 0, 1, 1)
        self.tilt_angle_entry = QtWidgets.QLineEdit(self.groupBox_7)
        self.tilt_angle_entry.setObjectName("tilt_angle_entry")
        self.gridLayout_12.addWidget(self.tilt_angle_entry, 0, 1, 1, 1)
        self.tilt_angle_label = QtWidgets.QLabel(self.groupBox_7)
        self.tilt_angle_label.setObjectName("tilt_angle_label")
        self.gridLayout_12.addWidget(self.tilt_angle_label, 0, 0, 1, 1)
        self.uvw_button = QtWidgets.QCheckBox(self.groupBox_7)
        self.uvw_button.setObjectName("uvw_button")
        self.gridLayout_12.addWidget(self.uvw_button, 3, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_7, 0, 0, 1, 2)
        self.groupBox_10 = QtWidgets.QGroupBox(self.settings)
        self.groupBox_10.setObjectName("groupBox_10")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_10)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.text_size_entry = QtWidgets.QLineEdit(self.groupBox_10)
        self.text_size_entry.setObjectName("text_size_entry")
        self.gridLayout_3.addWidget(self.text_size_entry, 3, 1, 1, 1)
        self.wulff_button = QtWidgets.QCheckBox(self.groupBox_10)
        self.wulff_button.setObjectName("wulff_button")
        self.gridLayout_3.addWidget(self.wulff_button, 0, 0, 1, 2)
        self.text_size_label = QtWidgets.QLabel(self.groupBox_10)
        self.text_size_label.setObjectName("text_size_label")
        self.gridLayout_3.addWidget(self.text_size_label, 3, 0, 1, 1)
        self.size_var = QtWidgets.QLineEdit(self.groupBox_10)
        self.size_var.setObjectName("size_var")
        self.gridLayout_3.addWidget(self.size_var, 2, 1, 1, 1)
        self.reset_view_button = QtWidgets.QPushButton(self.groupBox_10)
        self.reset_view_button.setObjectName("reset_view_button")
        self.gridLayout_3.addWidget(self.reset_view_button, 4, 0, 1, 2)
        self.size_var_label = QtWidgets.QLabel(self.groupBox_10)
        self.size_var_label.setObjectName("size_var_label")
        self.gridLayout_3.addWidget(self.size_var_label, 2, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_10, 1, 0, 1, 1)
        self.tabWidget.addTab(self.settings, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_10.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_10.setSpacing(0)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.groupBox_6 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_6.setMaximumSize(QtCore.QSize(250, 16777215))
        self.groupBox_6.setTitle("")
        self.groupBox_6.setFlat(False)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.lab_coord = QtWidgets.QLabel(self.groupBox_5)
        self.lab_coord.setObjectName("lab_coord")
        self.gridLayout_6.addWidget(self.lab_coord, 6, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label_2.setObjectName("label_2")
        self.gridLayout_6.addWidget(self.label_2, 3, 0, 1, 1)
        self.angle_euler_label = QtWidgets.QLabel(self.groupBox_5)
        self.angle_euler_label.setText("")
        self.angle_euler_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.angle_euler_label.setObjectName("angle_euler_label")
        self.gridLayout_6.addWidget(self.angle_euler_label, 1, 1, 1, 2)
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setObjectName("label")
        self.gridLayout_6.addWidget(self.label, 0, 0, 1, 1)
        self.angle_euler_label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.angle_euler_label_2.setText("")
        self.angle_euler_label_2.setObjectName("angle_euler_label_2")
        self.gridLayout_6.addWidget(self.angle_euler_label_2, 4, 1, 1, 2)
        self.phi1phiphi2_entry = QtWidgets.QLineEdit(self.groupBox_5)
        self.phi1phiphi2_entry.setObjectName("phi1phiphi2_entry")
        self.gridLayout_6.addWidget(self.phi1phiphi2_entry, 0, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.groupBox_5)
        self.label_3.setObjectName("label_3")
        self.gridLayout_6.addWidget(self.label_3, 4, 0, 1, 1)
        self.button_trace2 = QtWidgets.QPushButton(self.groupBox_5)
        self.button_trace2.setObjectName("button_trace2")
        self.gridLayout_6.addWidget(self.button_trace2, 5, 1, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem, 7, 1, 1, 4)
        self.coord_label = QtWidgets.QLabel(self.groupBox_5)
        self.coord_label.setText("")
        self.coord_label.setObjectName("coord_label")
        self.gridLayout_6.addWidget(self.coord_label, 6, 1, 1, 1)
        self.phi1phiphi2_2_entry = QtWidgets.QLineEdit(self.groupBox_5)
        self.phi1phiphi2_2_entry.setObjectName("phi1phiphi2_2_entry")
        self.gridLayout_6.addWidget(self.phi1phiphi2_2_entry, 3, 1, 1, 2)
        self.lab_euler2 = QtWidgets.QLabel(self.groupBox_5)
        self.lab_euler2.setObjectName("lab_euler2")
        self.gridLayout_6.addWidget(self.lab_euler2, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_5, 3, 0, 1, 2)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.pole_entry = QtWidgets.QLineEdit(self.groupBox_4)
        self.pole_entry.setObjectName("pole_entry")
        self.gridLayout_5.addWidget(self.pole_entry, 0, 0, 1, 3)
        self.undo_trace_plan_sym_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_trace_plan_sym_button.setObjectName("undo_trace_plan_sym_button")
        self.gridLayout_5.addWidget(self.undo_trace_plan_sym_button, 4, 2, 1, 1)
        self.trace_plan_sym_button = QtWidgets.QPushButton(self.groupBox_4)
        self.trace_plan_sym_button.setObjectName("trace_plan_sym_button")
        self.gridLayout_5.addWidget(self.trace_plan_sym_button, 4, 0, 1, 2)
        self.addpole_button = QtWidgets.QPushButton(self.groupBox_4)
        self.addpole_button.setObjectName("addpole_button")
        self.gridLayout_5.addWidget(self.addpole_button, 1, 0, 1, 2)
        self.undo_addpole_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_addpole_button.setObjectName("undo_addpole_button")
        self.gridLayout_5.addWidget(self.undo_addpole_button, 1, 2, 1, 1)
        self.undo_sym_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_sym_button.setObjectName("undo_sym_button")
        self.gridLayout_5.addWidget(self.undo_sym_button, 2, 2, 1, 1)
        self.sym_button = QtWidgets.QPushButton(self.groupBox_4)
        self.sym_button.setObjectName("sym_button")
        self.gridLayout_5.addWidget(self.sym_button, 2, 0, 1, 2)
        self.trace_plan_button = QtWidgets.QPushButton(self.groupBox_4)
        self.trace_plan_button.setObjectName("trace_plan_button")
        self.gridLayout_5.addWidget(self.trace_plan_button, 3, 0, 1, 2)
        self.undo_trace_plan_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_trace_plan_button.setObjectName("undo_trace_plan_button")
        self.gridLayout_5.addWidget(self.undo_trace_plan_button, 3, 2, 1, 1)
        self.gridLayout.addWidget(self.groupBox_4, 2, 0, 1, 2)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.crystal1_radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.crystal1_radioButton.setObjectName("crystal1_radioButton")
        self.gridLayout_8.addWidget(self.crystal1_radioButton, 0, 0, 1, 1)
        self.crystal2_radioButton = QtWidgets.QRadioButton(self.groupBox_2)
        self.crystal2_radioButton.setObjectName("crystal2_radioButton")
        self.gridLayout_8.addWidget(self.crystal2_radioButton, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.angle_alpha_label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.angle_alpha_label_2.setText("")
        self.angle_alpha_label_2.setObjectName("angle_alpha_label_2")
        self.gridLayout_9.addWidget(self.angle_alpha_label_2, 2, 3, 1, 1)
        self.angle_beta_label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.angle_beta_label_2.setText("")
        self.angle_beta_label_2.setObjectName("angle_beta_label_2")
        self.gridLayout_9.addWidget(self.angle_beta_label_2, 4, 3, 1, 1)
        self.angle_z_label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.angle_z_label_2.setText("")
        self.angle_z_label_2.setObjectName("angle_z_label_2")
        self.gridLayout_9.addWidget(self.angle_z_label_2, 6, 3, 1, 1)
        self.angle_z_buttonp = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_z_buttonp.setObjectName("angle_z_buttonp")
        self.gridLayout_9.addWidget(self.angle_z_buttonp, 6, 2, 1, 1)
        self.angle_alpha_buttonm = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_alpha_buttonm.setObjectName("angle_alpha_buttonm")
        self.gridLayout_9.addWidget(self.angle_alpha_buttonm, 2, 0, 1, 1)
        self.angle_alpha_label = QtWidgets.QLabel(self.groupBox_3)
        self.angle_alpha_label.setObjectName("angle_alpha_label")
        self.gridLayout_9.addWidget(self.angle_alpha_label, 1, 0, 1, 2)
        self.lock_checkButton = QtWidgets.QCheckBox(self.groupBox_3)
        self.lock_checkButton.setObjectName("lock_checkButton")
        self.gridLayout_9.addWidget(self.lock_checkButton, 0, 0, 1, 3)
        self.angle_z_label = QtWidgets.QLabel(self.groupBox_3)
        self.angle_z_label.setObjectName("angle_z_label")
        self.gridLayout_9.addWidget(self.angle_z_label, 5, 0, 1, 2)
        self.angle_z_entry = QtWidgets.QLineEdit(self.groupBox_3)
        self.angle_z_entry.setObjectName("angle_z_entry")
        self.gridLayout_9.addWidget(self.angle_z_entry, 6, 1, 1, 1)
        self.angle_z_buttonm = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_z_buttonm.setObjectName("angle_z_buttonm")
        self.gridLayout_9.addWidget(self.angle_z_buttonm, 6, 0, 1, 1)
        self.theta_signBox = QtWidgets.QCheckBox(self.groupBox_3)
        self.theta_signBox.setText("")
        self.theta_signBox.setObjectName("theta_signBox")
        self.gridLayout_9.addWidget(self.theta_signBox, 5, 2, 1, 1)
        self.angle_beta_buttonp = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_beta_buttonp.setObjectName("angle_beta_buttonp")
        self.gridLayout_9.addWidget(self.angle_beta_buttonp, 4, 2, 1, 1)
        self.angle_beta_entry = QtWidgets.QLineEdit(self.groupBox_3)
        self.angle_beta_entry.setObjectName("angle_beta_entry")
        self.gridLayout_9.addWidget(self.angle_beta_entry, 4, 1, 1, 1)
        self.angle_beta_buttonm = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_beta_buttonm.setObjectName("angle_beta_buttonm")
        self.gridLayout_9.addWidget(self.angle_beta_buttonm, 4, 0, 1, 1)
        self.beta_signBox = QtWidgets.QCheckBox(self.groupBox_3)
        self.beta_signBox.setText("")
        self.beta_signBox.setObjectName("beta_signBox")
        self.gridLayout_9.addWidget(self.beta_signBox, 3, 2, 1, 1)
        self.alpha_signBox = QtWidgets.QCheckBox(self.groupBox_3)
        self.alpha_signBox.setText("")
        self.alpha_signBox.setObjectName("alpha_signBox")
        self.gridLayout_9.addWidget(self.alpha_signBox, 1, 2, 1, 1)
        self.angle_alpha_buttonp = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_alpha_buttonp.setObjectName("angle_alpha_buttonp")
        self.gridLayout_9.addWidget(self.angle_alpha_buttonp, 2, 2, 1, 1)
        self.angle_alpha_entry = QtWidgets.QLineEdit(self.groupBox_3)
        self.angle_alpha_entry.setObjectName("angle_alpha_entry")
        self.gridLayout_9.addWidget(self.angle_alpha_entry, 2, 1, 1, 1)
        self.angle_beta_label = QtWidgets.QLabel(self.groupBox_3)
        self.angle_beta_label.setObjectName("angle_beta_label")
        self.gridLayout_9.addWidget(self.angle_beta_label, 3, 0, 1, 2)
        self.rot_gp_button = QtWidgets.QPushButton(self.groupBox_3)
        self.rot_gp_button.setObjectName("rot_gp_button")
        self.gridLayout_9.addWidget(self.rot_gp_button, 8, 2, 1, 1)
        self.rot_g_entry = QtWidgets.QLineEdit(self.groupBox_3)
        self.rot_g_entry.setObjectName("rot_g_entry")
        self.gridLayout_9.addWidget(self.rot_g_entry, 8, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setObjectName("label_4")
        self.gridLayout_9.addWidget(self.label_4, 7, 0, 1, 1)
        self.rot_gm_button = QtWidgets.QPushButton(self.groupBox_3)
        self.rot_gm_button.setObjectName("rot_gm_button")
        self.gridLayout_9.addWidget(self.rot_gm_button, 8, 0, 1, 1)
        self.rg_label = QtWidgets.QLabel(self.groupBox_3)
        self.rg_label.setText("")
        self.rg_label.setObjectName("rg_label")
        self.gridLayout_9.addWidget(self.rg_label, 8, 3, 1, 1)
        self.gridLayout.addWidget(self.groupBox_3, 1, 0, 1, 1)
        self.gridLayout_10.addWidget(self.groupBox_6, 0, 1, 1, 1)
        self.mplvl = QtWidgets.QGridLayout()
        self.mplvl.setObjectName("mplvl")
        self.gridLayout_10.addLayout(self.mplvl, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.misorientation = QtWidgets.QWidget()
        self.misorientation.setObjectName("misorientation")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.misorientation)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.misorientation_button = QtWidgets.QPushButton(self.misorientation)
        self.misorientation_button.setObjectName("misorientation_button")
        self.gridLayout_13.addWidget(self.misorientation_button, 0, 0, 1, 1)
        self.numbers_checkBox = QtWidgets.QCheckBox(self.misorientation)
        self.numbers_checkBox.setObjectName("numbers_checkBox")
        self.gridLayout_13.addWidget(self.numbers_checkBox, 2, 0, 1, 1)
        self.axis_checkBox = QtWidgets.QCheckBox(self.misorientation)
        self.axis_checkBox.setObjectName("axis_checkBox")
        self.gridLayout_13.addWidget(self.axis_checkBox, 3, 0, 1, 1)
        self.clear_misorientation_button = QtWidgets.QPushButton(self.misorientation)
        self.clear_misorientation_button.setObjectName("clear_misorientation_button")
        self.gridLayout_13.addWidget(self.clear_misorientation_button, 1, 0, 1, 1)
        self.misorientation_list = QtWidgets.QTableWidget(self.misorientation)
        self.misorientation_list.setObjectName("misorientation_list")
        self.misorientation_list.setColumnCount(0)
        self.misorientation_list.setRowCount(0)
        self.gridLayout_13.addWidget(self.misorientation_list, 4, 0, 1, 1)
        self.tabWidget.addTab(self.misorientation, "")
        self.gridLayout_11.addWidget(self.tabWidget, 0, 0, 1, 1)
        Misorientation.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Misorientation)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1212, 23))
        self.menubar.setObjectName("menubar")
        self.menuSave = QtWidgets.QMenu(self.menubar)
        self.menuSave.setObjectName("menuSave")
        Misorientation.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Misorientation)
        self.statusbar.setObjectName("statusbar")
        Misorientation.setStatusBar(self.statusbar)
        self.actionSave_figure = QtWidgets.QAction(Misorientation)
        self.actionSave_figure.setObjectName("actionSave_figure")
        self.actionCalculate_Schmid_factor = QtWidgets.QAction(Misorientation)
        self.actionCalculate_Schmid_factor.setObjectName("actionCalculate_Schmid_factor")
        self.actionCalculate_angle = QtWidgets.QAction(Misorientation)
        self.actionCalculate_angle.setObjectName("actionCalculate_angle")
        self.actionCalculate_xyz = QtWidgets.QAction(Misorientation)
        self.actionCalculate_xyz.setObjectName("actionCalculate_xyz")
        self.actionCalculate_apparent_width = QtWidgets.QAction(Misorientation)
        self.actionCalculate_apparent_width.setObjectName("actionCalculate_apparent_width")
        self.actionPlanes = QtWidgets.QAction(Misorientation)
        self.actionPlanes.setObjectName("actionPlanes")
        self.actionProj_directions = QtWidgets.QAction(Misorientation)
        self.actionProj_directions.setObjectName("actionProj_directions")
        self.actionPlane_cone = QtWidgets.QAction(Misorientation)
        self.actionPlane_cone.setObjectName("actionPlane_cone")
        self.actionCalculate_intersections = QtWidgets.QAction(Misorientation)
        self.actionCalculate_intersections.setObjectName("actionCalculate_intersections")
        self.actionHkl_uvw = QtWidgets.QAction(Misorientation)
        self.actionHkl_uvw.setObjectName("actionHkl_uvw")
        self.actionPlot_Kikuchi_lines = QtWidgets.QAction(Misorientation)
        self.actionPlot_Kikuchi_lines.setObjectName("actionPlot_Kikuchi_lines")
        self.menuSave.addAction(self.actionSave_figure)
        self.menubar.addAction(self.menuSave.menuAction())

        self.retranslateUi(Misorientation)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Misorientation)
        Misorientation.setTabOrder(self.tabWidget, self.tilt_angle_entry)
        Misorientation.setTabOrder(self.tilt_angle_entry, self.image_angle_entry)
        Misorientation.setTabOrder(self.image_angle_entry, self.real_space_checkBox)
        Misorientation.setTabOrder(self.real_space_checkBox, self.uvw_button)
        Misorientation.setTabOrder(self.uvw_button, self.wulff_button)
        Misorientation.setTabOrder(self.wulff_button, self.size_var)
        Misorientation.setTabOrder(self.size_var, self.text_size_entry)
        Misorientation.setTabOrder(self.text_size_entry, self.reset_view_button)
        Misorientation.setTabOrder(self.reset_view_button, self.structure_box)
        Misorientation.setTabOrder(self.structure_box, self.abc_entry)
        Misorientation.setTabOrder(self.abc_entry, self.alphabetagamma_entry)
        Misorientation.setTabOrder(self.alphabetagamma_entry, self.e_entry)
        Misorientation.setTabOrder(self.e_entry, self.hexa_button)
        Misorientation.setTabOrder(self.hexa_button, self.color_trace_bleu)
        Misorientation.setTabOrder(self.color_trace_bleu, self.color_trace_vert)
        Misorientation.setTabOrder(self.color_trace_vert, self.color_trace_rouge)
        Misorientation.setTabOrder(self.color_trace_rouge, self.style_box)
        Misorientation.setTabOrder(self.style_box, self.structure2_box)
        Misorientation.setTabOrder(self.structure2_box, self.abc_entry_2)
        Misorientation.setTabOrder(self.abc_entry_2, self.alphabetagamma_entry_2)
        Misorientation.setTabOrder(self.alphabetagamma_entry_2, self.e_entry_2)
        Misorientation.setTabOrder(self.e_entry_2, self.hexa_button_2)
        Misorientation.setTabOrder(self.hexa_button_2, self.color_trace_bleu_2)
        Misorientation.setTabOrder(self.color_trace_bleu_2, self.color_trace_vert_2)
        Misorientation.setTabOrder(self.color_trace_vert_2, self.color_trace_rouge_2)
        Misorientation.setTabOrder(self.color_trace_rouge_2, self.style_box_2)
        Misorientation.setTabOrder(self.style_box_2, self.crystal1_radioButton)
        Misorientation.setTabOrder(self.crystal1_radioButton, self.crystal2_radioButton)
        Misorientation.setTabOrder(self.crystal2_radioButton, self.lock_checkButton)
        Misorientation.setTabOrder(self.lock_checkButton, self.angle_alpha_buttonm)
        Misorientation.setTabOrder(self.angle_alpha_buttonm, self.angle_alpha_entry)
        Misorientation.setTabOrder(self.angle_alpha_entry, self.alpha_signBox)
        Misorientation.setTabOrder(self.alpha_signBox, self.angle_alpha_buttonp)
        Misorientation.setTabOrder(self.angle_alpha_buttonp, self.angle_beta_buttonm)
        Misorientation.setTabOrder(self.angle_beta_buttonm, self.angle_beta_entry)
        Misorientation.setTabOrder(self.angle_beta_entry, self.beta_signBox)
        Misorientation.setTabOrder(self.beta_signBox, self.angle_beta_buttonp)
        Misorientation.setTabOrder(self.angle_beta_buttonp, self.angle_z_buttonm)
        Misorientation.setTabOrder(self.angle_z_buttonm, self.angle_z_entry)
        Misorientation.setTabOrder(self.angle_z_entry, self.theta_signBox)
        Misorientation.setTabOrder(self.theta_signBox, self.angle_z_buttonp)
        Misorientation.setTabOrder(self.angle_z_buttonp, self.rot_gm_button)
        Misorientation.setTabOrder(self.rot_gm_button, self.rot_g_entry)
        Misorientation.setTabOrder(self.rot_g_entry, self.rot_gp_button)
        Misorientation.setTabOrder(self.rot_gp_button, self.pole_entry)
        Misorientation.setTabOrder(self.pole_entry, self.addpole_button)
        Misorientation.setTabOrder(self.addpole_button, self.undo_addpole_button)
        Misorientation.setTabOrder(self.undo_addpole_button, self.sym_button)
        Misorientation.setTabOrder(self.sym_button, self.undo_sym_button)
        Misorientation.setTabOrder(self.undo_sym_button, self.trace_plan_button)
        Misorientation.setTabOrder(self.trace_plan_button, self.undo_trace_plan_button)
        Misorientation.setTabOrder(self.undo_trace_plan_button, self.trace_plan_sym_button)
        Misorientation.setTabOrder(self.trace_plan_sym_button, self.undo_trace_plan_sym_button)
        Misorientation.setTabOrder(self.undo_trace_plan_sym_button, self.phi1phiphi2_entry)
        Misorientation.setTabOrder(self.phi1phiphi2_entry, self.phi1phiphi2_2_entry)
        Misorientation.setTabOrder(self.phi1phiphi2_2_entry, self.button_trace2)
        Misorientation.setTabOrder(self.button_trace2, self.misorientation_button)
        Misorientation.setTabOrder(self.misorientation_button, self.clear_misorientation_button)
        Misorientation.setTabOrder(self.clear_misorientation_button, self.numbers_checkBox)
        Misorientation.setTabOrder(self.numbers_checkBox, self.axis_checkBox)
        Misorientation.setTabOrder(self.axis_checkBox, self.misorientation_list)

    def retranslateUi(self, Misorientation):
        _translate = QtCore.QCoreApplication.translate
        Misorientation.setWindowTitle(_translate("Misorientation", "Misorientation"))
        self.groupBox.setTitle(_translate("Misorientation", "Crystal 1"))
        self.alphabetagamma_label.setText(_translate("Misorientation", "<p>&alpha;, &beta;, &gamma;</p>"))
        self.abc_label.setText(_translate("Misorientation", "a,b,c"))
        self.style_box.setText(_translate("Misorientation", "open/filled"))
        self.e_label.setText(_translate("Misorientation", "max indices"))
        self.color_trace_rouge.setText(_translate("Misorientation", "red"))
        self.d_label.setText(_translate("Misorientation", "d"))
        self.color_trace_vert.setText(_translate("Misorientation", "green"))
        self.color_trace_bleu.setText(_translate("Misorientation", "blue"))
        self.hexa_button.setText(_translate("Misorientation", "hexa"))
        self.crystal2_box.setTitle(_translate("Misorientation", "Crystal 2"))
        self.style_box_2.setText(_translate("Misorientation", "open/filled"))
        self.abc_label_2.setText(_translate("Misorientation", "a,b,c"))
        self.color_trace_rouge_2.setText(_translate("Misorientation", "red"))
        self.alphabetagamma_label_2.setText(_translate("Misorientation", "<p>&alpha;, &beta;, &gamma;</p>"))
        self.e_label_2.setText(_translate("Misorientation", "max indices"))
        self.hexa_button_2.setText(_translate("Misorientation", "hexa"))
        self.color_trace_bleu_2.setText(_translate("Misorientation", "blue"))
        self.d_label_2.setText(_translate("Misorientation", "d"))
        self.color_trace_vert_2.setText(_translate("Misorientation", "green"))
        self.groupBox_7.setTitle(_translate("Misorientation", "Settings"))
        self.real_space_checkBox.setText(_translate("Misorientation", "work in real space"))
        self.image_tilt_y_label.setText(_translate("Misorientation", "Image α-tilt/y angle"))
        self.tilt_angle_label.setText(_translate("Misorientation", "Diffraction α-tilt/y angle"))
        self.uvw_button.setText(_translate("Misorientation", "uvw"))
        self.groupBox_10.setTitle(_translate("Misorientation", "Layout"))
        self.wulff_button.setText(_translate("Misorientation", "Wulff net"))
        self.text_size_label.setText(_translate("Misorientation", "Text size"))
        self.reset_view_button.setText(_translate("Misorientation", "Update/Reset view"))
        self.size_var_label.setText(_translate("Misorientation", "Marker size"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.settings), _translate("Misorientation", "settings"))
        self.groupBox_5.setTitle(_translate("Misorientation", "Euler Angles"))
        self.lab_coord.setText(_translate("Misorientation", "Tilt, Inclin."))
        self.label_2.setText(_translate("Misorientation", "Crystal 2"))
        self.label.setText(_translate("Misorientation", "Crystal 1"))
        self.label_3.setText(_translate("Misorientation", "φ 1 , Φ , φ2"))
        self.button_trace2.setText(_translate("Misorientation", "PLOT"))
        self.lab_euler2.setText(_translate("Misorientation", "<html><head/><body><p>φ<span style=\" vertical-align:sub;\"> 1</span> , Φ , φ<span style=\" vertical-align:sub;\">2</span></p></body></html>"))
        self.groupBox_4.setTitle(_translate("Misorientation", "Pole/Direction"))
        self.undo_trace_plan_sym_button.setText(_translate("Misorientation", "-"))
        self.trace_plan_sym_button.setText(_translate("Misorientation", "Sym Plane"))
        self.addpole_button.setText(_translate("Misorientation", "Add"))
        self.undo_addpole_button.setText(_translate("Misorientation", "-"))
        self.undo_sym_button.setText(_translate("Misorientation", "-"))
        self.sym_button.setText(_translate("Misorientation", "Symmetry"))
        self.trace_plan_button.setText(_translate("Misorientation", " Plane"))
        self.undo_trace_plan_button.setText(_translate("Misorientation", "-"))
        self.groupBox_2.setTitle(_translate("Misorientation", "Switch"))
        self.crystal1_radioButton.setText(_translate("Misorientation", "crystal 1"))
        self.crystal2_radioButton.setText(_translate("Misorientation", "crystal 2"))
        self.groupBox_3.setTitle(_translate("Misorientation", "Rotation"))
        self.angle_z_buttonp.setText(_translate("Misorientation", "+"))
        self.angle_alpha_buttonm.setText(_translate("Misorientation", "-"))
        self.angle_alpha_label.setText(_translate("Misorientation", "<html><head/><body><p>α (AC)</p></body></html>"))
        self.lock_checkButton.setText(_translate("Misorientation", "Lock Axes"))
        self.angle_z_label.setText(_translate("Misorientation", "<html><head/><body><p>θ (AC)</p></body></html>"))
        self.angle_z_buttonm.setText(_translate("Misorientation", "-"))
        self.angle_beta_buttonp.setText(_translate("Misorientation", "+"))
        self.angle_beta_buttonm.setText(_translate("Misorientation", "-"))
        self.angle_alpha_buttonp.setText(_translate("Misorientation", "+"))
        self.angle_beta_label.setText(_translate("Misorientation", "<html><head/><body><p>β (AC)</p></body></html>"))
        self.rot_gp_button.setText(_translate("Misorientation", "+"))
        self.label_4.setText(_translate("Misorientation", "g"))
        self.rot_gm_button.setText(_translate("Misorientation", "-"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Misorientation", "stereo-proj"))
        self.misorientation_button.setText(_translate("Misorientation", "Find misorientation & Plot Angle/Axes"))
        self.numbers_checkBox.setText(_translate("Misorientation", "show numbers"))
        self.axis_checkBox.setText(_translate("Misorientation", "show axes"))
        self.clear_misorientation_button.setText(_translate("Misorientation", "Clear plot"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.misorientation), _translate("Misorientation", "misorientation"))
        self.menuSave.setTitle(_translate("Misorientation", "Save"))
        self.actionSave_figure.setText(_translate("Misorientation", "Save figure"))
        self.actionCalculate_Schmid_factor.setText(_translate("Misorientation", "calculate Schmid factor"))
        self.actionCalculate_angle.setText(_translate("Misorientation", "Calculate angle"))
        self.actionCalculate_xyz.setText(_translate("Misorientation", "calculate xyz directions"))
        self.actionCalculate_apparent_width.setText(_translate("Misorientation", "Calculate apparent width"))
        self.actionPlanes.setText(_translate("Misorientation", "planes"))
        self.actionProj_directions.setText(_translate("Misorientation", "proj. directions"))
        self.actionPlane_cone.setText(_translate("Misorientation", "plane-cone"))
        self.actionCalculate_intersections.setText(_translate("Misorientation", "Calculate intersections"))
        self.actionHkl_uvw.setText(_translate("Misorientation", "hkl <> uvw"))
        self.actionPlot_Kikuchi_lines.setText(_translate("Misorientation", "plot Kikuchi lines or diffraction pattern"))
