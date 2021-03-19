# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'stereoprojUI.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import os
fn = os.path.join(os.path.dirname(__file__), "stereo-icon.png")

class Ui_StereoProj(object):
    def setupUi(self, StereoProj):
        StereoProj.setObjectName("StereoProj")
        StereoProj.resize(1403, 884)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(fn), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        StereoProj.setWindowIcon(icon)
        StereoProj.setIconSize(QtCore.QSize(24, 24))
        self.centralwidget = QtWidgets.QWidget(StereoProj)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.mplvl = QtWidgets.QGridLayout()
        self.mplvl.setObjectName("mplvl")
        self.gridLayout_7.addLayout(self.mplvl, 0, 0, 1, 1)
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setMaximumSize(QtCore.QSize(460, 16777215))
        self.groupBox_6.setTitle("")
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout.setContentsMargins(-1, 0, -1, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setMaximumSize(QtCore.QSize(230, 16777215))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_3.setSpacing(5)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.rot_gm_button = QtWidgets.QPushButton(self.groupBox_2)
        self.rot_gm_button.setObjectName("rot_gm_button")
        self.gridLayout_3.addWidget(self.rot_gm_button, 9, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem, 11, 1, 1, 1)
        self.diff_entry = QtWidgets.QLineEdit(self.groupBox_2)
        self.diff_entry.setObjectName("diff_entry")
        self.gridLayout_3.addWidget(self.diff_entry, 0, 1, 1, 3)
        self.diff_label = QtWidgets.QLabel(self.groupBox_2)
        self.diff_label.setObjectName("diff_label")
        self.gridLayout_3.addWidget(self.diff_label, 0, 0, 1, 1)
        self.inclinaison_entry = QtWidgets.QLineEdit(self.groupBox_2)
        self.inclinaison_entry.setObjectName("inclinaison_entry")
        self.gridLayout_3.addWidget(self.inclinaison_entry, 4, 1, 1, 1)
        self.rot_g_entry = QtWidgets.QLineEdit(self.groupBox_2)
        self.rot_g_entry.setObjectName("rot_g_entry")
        self.gridLayout_3.addWidget(self.rot_g_entry, 9, 1, 1, 1)
        self.inclination_label = QtWidgets.QLabel(self.groupBox_2)
        self.inclination_label.setObjectName("inclination_label")
        self.gridLayout_3.addWidget(self.inclination_label, 4, 0, 1, 1)
        self.rot_diff_label = QtWidgets.QLabel(self.groupBox_2)
        self.rot_diff_label.setObjectName("rot_diff_label")
        self.gridLayout_3.addWidget(self.rot_diff_label, 8, 0, 1, 3)
        self.button_trace = QtWidgets.QPushButton(self.groupBox_2)
        self.button_trace.setObjectName("button_trace")
        self.gridLayout_3.addWidget(self.button_trace, 5, 0, 1, 2)
        self.rot_gp_button = QtWidgets.QPushButton(self.groupBox_2)
        self.rot_gp_button.setObjectName("rot_gp_button")
        self.gridLayout_3.addWidget(self.rot_gp_button, 9, 2, 1, 1)
        self.tilt_entry = QtWidgets.QLineEdit(self.groupBox_2)
        self.tilt_entry.setObjectName("tilt_entry")
        self.gridLayout_3.addWidget(self.tilt_entry, 1, 1, 1, 3)
        self.rg_label = QtWidgets.QLabel(self.groupBox_2)
        self.rg_label.setText("")
        self.rg_label.setObjectName("rg_label")
        self.gridLayout_3.addWidget(self.rg_label, 10, 1, 1, 1)
        self.tilt_label = QtWidgets.QLabel(self.groupBox_2)
        self.tilt_label.setObjectName("tilt_label")
        self.gridLayout_3.addWidget(self.tilt_label, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 2, 0, 1, 1)
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_5.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.coord_label = QtWidgets.QLabel(self.groupBox_5)
        self.coord_label.setText("")
        self.coord_label.setObjectName("coord_label")
        self.gridLayout_6.addWidget(self.coord_label, 5, 0, 1, 3)
        self.lab_euler2 = QtWidgets.QLabel(self.groupBox_5)
        self.lab_euler2.setObjectName("lab_euler2")
        self.gridLayout_6.addWidget(self.lab_euler2, 2, 0, 1, 3)
        self.button_trace2 = QtWidgets.QPushButton(self.groupBox_5)
        self.button_trace2.setObjectName("button_trace2")
        self.gridLayout_6.addWidget(self.button_trace2, 1, 0, 1, 3)
        self.lab_coord = QtWidgets.QLabel(self.groupBox_5)
        self.lab_coord.setObjectName("lab_coord")
        self.gridLayout_6.addWidget(self.lab_coord, 4, 0, 1, 3)
        self.phi1phiphi2_entry = QtWidgets.QLineEdit(self.groupBox_5)
        self.phi1phiphi2_entry.setObjectName("phi1phiphi2_entry")
        self.gridLayout_6.addWidget(self.phi1phiphi2_entry, 0, 0, 1, 3)
        self.angle_euler_label = QtWidgets.QLabel(self.groupBox_5)
        self.angle_euler_label.setText("")
        self.angle_euler_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.angle_euler_label.setObjectName("angle_euler_label")
        self.gridLayout_6.addWidget(self.angle_euler_label, 3, 0, 1, 3)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem1, 6, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_5, 2, 1, 1, 1)
        self.groupBox_9 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_9.setMaximumSize(QtCore.QSize(230, 16777215))
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout_10.setObjectName("gridLayout_10")
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_10.addItem(spacerItem2, 11, 0, 1, 1)
        self.real_space_checkBox = QtWidgets.QCheckBox(self.groupBox_9)
        self.real_space_checkBox.setObjectName("real_space_checkBox")
        self.gridLayout_10.addWidget(self.real_space_checkBox, 3, 2, 1, 2)
        self.tilt_angle_entry = QtWidgets.QLineEdit(self.groupBox_9)
        self.tilt_angle_entry.setObjectName("tilt_angle_entry")
        self.gridLayout_10.addWidget(self.tilt_angle_entry, 7, 3, 1, 1)
        self.reset_view_button = QtWidgets.QPushButton(self.groupBox_9)
        self.reset_view_button.setObjectName("reset_view_button")
        self.gridLayout_10.addWidget(self.reset_view_button, 10, 0, 1, 4)
        self.style_box = QtWidgets.QCheckBox(self.groupBox_9)
        self.style_box.setObjectName("style_box")
        self.gridLayout_10.addWidget(self.style_box, 2, 0, 1, 3)
        self.wulff_button = QtWidgets.QCheckBox(self.groupBox_9)
        self.wulff_button.setObjectName("wulff_button")
        self.gridLayout_10.addWidget(self.wulff_button, 3, 0, 1, 2)
        self.color_trace_bleu = QtWidgets.QRadioButton(self.groupBox_9)
        self.color_trace_bleu.setObjectName("color_trace_bleu")
        self.gridLayout_10.addWidget(self.color_trace_bleu, 0, 0, 1, 1)
        self.image_angle_entry = QtWidgets.QLineEdit(self.groupBox_9)
        self.image_angle_entry.setObjectName("image_angle_entry")
        self.gridLayout_10.addWidget(self.image_angle_entry, 8, 3, 1, 1)
        self.tilt_angle_label = QtWidgets.QLabel(self.groupBox_9)
        self.tilt_angle_label.setObjectName("tilt_angle_label")
        self.gridLayout_10.addWidget(self.tilt_angle_label, 7, 0, 1, 3)
        self.color_trace_vert = QtWidgets.QRadioButton(self.groupBox_9)
        self.color_trace_vert.setObjectName("color_trace_vert")
        self.gridLayout_10.addWidget(self.color_trace_vert, 0, 1, 1, 2)
        self.size_var = QtWidgets.QLineEdit(self.groupBox_9)
        self.size_var.setObjectName("size_var")
        self.gridLayout_10.addWidget(self.size_var, 5, 3, 1, 1)
        self.text_size_label = QtWidgets.QLabel(self.groupBox_9)
        self.text_size_label.setObjectName("text_size_label")
        self.gridLayout_10.addWidget(self.text_size_label, 6, 0, 1, 2)
        self.color_trace_rouge = QtWidgets.QRadioButton(self.groupBox_9)
        self.color_trace_rouge.setObjectName("color_trace_rouge")
        self.gridLayout_10.addWidget(self.color_trace_rouge, 0, 3, 1, 1)
        self.hexa_button = QtWidgets.QCheckBox(self.groupBox_9)
        self.hexa_button.setObjectName("hexa_button")
        self.gridLayout_10.addWidget(self.hexa_button, 9, 0, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.groupBox_9)
        self.label_2.setObjectName("label_2")
        self.gridLayout_10.addWidget(self.label_2, 8, 0, 1, 3)
        self.uvw_button = QtWidgets.QCheckBox(self.groupBox_9)
        self.uvw_button.setObjectName("uvw_button")
        self.gridLayout_10.addWidget(self.uvw_button, 9, 3, 1, 1)
        self.text_size_entry = QtWidgets.QLineEdit(self.groupBox_9)
        self.text_size_entry.setObjectName("text_size_entry")
        self.gridLayout_10.addWidget(self.text_size_entry, 6, 3, 1, 1)
        self.size_var_label = QtWidgets.QLabel(self.groupBox_9)
        self.size_var_label.setObjectName("size_var_label")
        self.gridLayout_10.addWidget(self.size_var_label, 5, 0, 1, 3)
        self.gridLayout.addWidget(self.groupBox_9, 1, 0, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_4.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.trace_plan_sym_button = QtWidgets.QPushButton(self.groupBox_4)
        self.trace_plan_sym_button.setObjectName("trace_plan_sym_button")
        self.gridLayout_5.addWidget(self.trace_plan_sym_button, 4, 0, 1, 3)
        self.sym_button = QtWidgets.QPushButton(self.groupBox_4)
        self.sym_button.setObjectName("sym_button")
        self.gridLayout_5.addWidget(self.sym_button, 2, 0, 1, 3)
        self.undo_trace_cone_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_trace_cone_button.setObjectName("undo_trace_cone_button")
        self.gridLayout_5.addWidget(self.undo_trace_cone_button, 6, 3, 1, 1)
        self.trace_plan_button = QtWidgets.QPushButton(self.groupBox_4)
        self.trace_plan_button.setObjectName("trace_plan_button")
        self.gridLayout_5.addWidget(self.trace_plan_button, 3, 0, 1, 3)
        self.undo_trace_plan_sym_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_trace_plan_sym_button.setObjectName("undo_trace_plan_sym_button")
        self.gridLayout_5.addWidget(self.undo_trace_plan_sym_button, 4, 3, 1, 1)
        self.inclination_entry = QtWidgets.QLineEdit(self.groupBox_4)
        self.inclination_entry.setObjectName("inclination_entry")
        self.gridLayout_5.addWidget(self.inclination_entry, 6, 2, 1, 1)
        self.trace_cone_button = QtWidgets.QPushButton(self.groupBox_4)
        self.trace_cone_button.setObjectName("trace_cone_button")
        self.gridLayout_5.addWidget(self.trace_cone_button, 6, 0, 1, 2)
        self.undo_trace_schmid = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_trace_schmid.setObjectName("undo_trace_schmid")
        self.gridLayout_5.addWidget(self.undo_trace_schmid, 5, 3, 1, 1)
        self.norm_button = QtWidgets.QPushButton(self.groupBox_4)
        self.norm_button.setObjectName("norm_button")
        self.gridLayout_5.addWidget(self.norm_button, 7, 0, 1, 1)
        self.pole_entry = QtWidgets.QLineEdit(self.groupBox_4)
        self.pole_entry.setObjectName("pole_entry")
        self.gridLayout_5.addWidget(self.pole_entry, 0, 0, 1, 4)
        self.undo_trace_plan_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_trace_plan_button.setObjectName("undo_trace_plan_button")
        self.gridLayout_5.addWidget(self.undo_trace_plan_button, 3, 3, 1, 1)
        self.undo_addpole_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_addpole_button.setObjectName("undo_addpole_button")
        self.gridLayout_5.addWidget(self.undo_addpole_button, 1, 3, 1, 1)
        self.addpole_button = QtWidgets.QPushButton(self.groupBox_4)
        self.addpole_button.setObjectName("addpole_button")
        self.gridLayout_5.addWidget(self.addpole_button, 1, 0, 1, 3)
        self.dhkl_label = QtWidgets.QLabel(self.groupBox_4)
        self.dhkl_label.setText("")
        self.dhkl_label.setObjectName("dhkl_label")
        self.gridLayout_5.addWidget(self.dhkl_label, 7, 2, 1, 2)
        self.undo_sym_button = QtWidgets.QPushButton(self.groupBox_4)
        self.undo_sym_button.setObjectName("undo_sym_button")
        self.gridLayout_5.addWidget(self.undo_sym_button, 2, 3, 1, 1)
        self.trace_schmid_button = QtWidgets.QPushButton(self.groupBox_4)
        self.trace_schmid_button.setObjectName("trace_schmid_button")
        self.gridLayout_5.addWidget(self.trace_schmid_button, 5, 0, 1, 3)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_5.addItem(spacerItem3, 8, 2, 1, 1)
        self.gridLayout.addWidget(self.groupBox_4, 1, 1, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_3.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_4.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_4.setSpacing(5)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.angle_beta_entry = QtWidgets.QLineEdit(self.groupBox_3)
        self.angle_beta_entry.setObjectName("angle_beta_entry")
        self.gridLayout_4.addWidget(self.angle_beta_entry, 8, 2, 1, 1)
        self.angle_beta_label = QtWidgets.QLabel(self.groupBox_3)
        self.angle_beta_label.setObjectName("angle_beta_label")
        self.gridLayout_4.addWidget(self.angle_beta_label, 8, 0, 1, 1)
        self.angle_z_buttonm = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_z_buttonm.setObjectName("angle_z_buttonm")
        self.gridLayout_4.addWidget(self.angle_z_buttonm, 11, 1, 1, 1)
        self.angle_alpha_label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.angle_alpha_label_2.setText("")
        self.angle_alpha_label_2.setObjectName("angle_alpha_label_2")
        self.gridLayout_4.addWidget(self.angle_alpha_label_2, 2, 5, 1, 1)
        self.angle_alpha_entry = QtWidgets.QLineEdit(self.groupBox_3)
        self.angle_alpha_entry.setObjectName("angle_alpha_entry")
        self.gridLayout_4.addWidget(self.angle_alpha_entry, 2, 2, 1, 1)
        self.angle_beta_buttonp = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_beta_buttonp.setObjectName("angle_beta_buttonp")
        self.gridLayout_4.addWidget(self.angle_beta_buttonp, 8, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 4, 1, 2)
        self.angle_beta_buttonm = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_beta_buttonm.setObjectName("angle_beta_buttonm")
        self.gridLayout_4.addWidget(self.angle_beta_buttonm, 8, 1, 1, 1)
        self.angle_z_label = QtWidgets.QLabel(self.groupBox_3)
        self.angle_z_label.setObjectName("angle_z_label")
        self.gridLayout_4.addWidget(self.angle_z_label, 11, 0, 1, 1)
        self.angle_z_entry = QtWidgets.QLineEdit(self.groupBox_3)
        self.angle_z_entry.setObjectName("angle_z_entry")
        self.gridLayout_4.addWidget(self.angle_z_entry, 11, 2, 1, 1)
        self.lock_checkButton = QtWidgets.QCheckBox(self.groupBox_3)
        self.lock_checkButton.setObjectName("lock_checkButton")
        self.gridLayout_4.addWidget(self.lock_checkButton, 0, 0, 1, 4)
        self.angle_alpha_buttonm = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_alpha_buttonm.setObjectName("angle_alpha_buttonm")
        self.gridLayout_4.addWidget(self.angle_alpha_buttonm, 2, 1, 1, 1)
        self.beta_signBox = QtWidgets.QCheckBox(self.groupBox_3)
        self.beta_signBox.setText("")
        self.beta_signBox.setObjectName("beta_signBox")
        self.gridLayout_4.addWidget(self.beta_signBox, 8, 4, 1, 1)
        self.angle_z_buttonp = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_z_buttonp.setObjectName("angle_z_buttonp")
        self.gridLayout_4.addWidget(self.angle_z_buttonp, 11, 3, 1, 1)
        self.alpha_signBox = QtWidgets.QCheckBox(self.groupBox_3)
        self.alpha_signBox.setText("")
        self.alpha_signBox.setObjectName("alpha_signBox")
        self.gridLayout_4.addWidget(self.alpha_signBox, 2, 4, 1, 1)
        self.theta_signBox = QtWidgets.QCheckBox(self.groupBox_3)
        self.theta_signBox.setText("")
        self.theta_signBox.setObjectName("theta_signBox")
        self.gridLayout_4.addWidget(self.theta_signBox, 11, 4, 1, 1)
        self.angle_alpha_buttonp = QtWidgets.QPushButton(self.groupBox_3)
        self.angle_alpha_buttonp.setObjectName("angle_alpha_buttonp")
        self.gridLayout_4.addWidget(self.angle_alpha_buttonp, 2, 3, 1, 1)
        self.angle_beta_label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.angle_beta_label_2.setText("")
        self.angle_beta_label_2.setObjectName("angle_beta_label_2")
        self.gridLayout_4.addWidget(self.angle_beta_label_2, 8, 5, 1, 1)
        self.angle_alpha_label = QtWidgets.QLabel(self.groupBox_3)
        self.angle_alpha_label.setObjectName("angle_alpha_label")
        self.gridLayout_4.addWidget(self.angle_alpha_label, 2, 0, 1, 1)
        self.angle_z_label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.angle_z_label_2.setText("")
        self.angle_z_label_2.setObjectName("angle_z_label_2")
        self.gridLayout_4.addWidget(self.angle_z_label_2, 11, 5, 1, 1)
        self.reset_angle_button = QtWidgets.QPushButton(self.groupBox_3)
        self.reset_angle_button.setObjectName("reset_angle_button")
        self.gridLayout_4.addWidget(self.reset_angle_button, 12, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox_3, 0, 1, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMaximumSize(QtCore.QSize(230, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_2.setSpacing(5)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.abc_entry = QtWidgets.QLineEdit(self.groupBox)
        self.abc_entry.setObjectName("abc_entry")
        self.gridLayout_2.addWidget(self.abc_entry, 0, 3, 1, 3)
        self.alphabetagamma_entry = QtWidgets.QLineEdit(self.groupBox)
        self.alphabetagamma_entry.setObjectName("alphabetagamma_entry")
        self.gridLayout_2.addWidget(self.alphabetagamma_entry, 2, 3, 1, 3)
        self.alphabetagamma_label = QtWidgets.QLabel(self.groupBox)
        self.alphabetagamma_label.setObjectName("alphabetagamma_label")
        self.gridLayout_2.addWidget(self.alphabetagamma_label, 2, 0, 1, 1)
        self.e_entry = QtWidgets.QLineEdit(self.groupBox)
        self.e_entry.setObjectName("e_entry")
        self.gridLayout_2.addWidget(self.e_entry, 4, 4, 1, 1)
        self.abc_label = QtWidgets.QLabel(self.groupBox)
        self.abc_label.setObjectName("abc_label")
        self.gridLayout_2.addWidget(self.abc_label, 0, 0, 1, 1)
        self.e_label = QtWidgets.QLabel(self.groupBox)
        self.e_label.setObjectName("e_label")
        self.gridLayout_2.addWidget(self.e_label, 4, 0, 1, 4)
        self.d_label = QtWidgets.QLabel(self.groupBox)
        self.d_label.setObjectName("d_label")
        self.gridLayout_2.addWidget(self.d_label, 9, 0, 1, 1)
        self.d_label_var = QtWidgets.QLabel(self.groupBox)
        self.d_label_var.setText("")
        self.d_label_var.setObjectName("d_label_var")
        self.gridLayout_2.addWidget(self.d_label_var, 10, 2, 1, 3)
        self.reciprocal_checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.reciprocal_checkBox.setObjectName("reciprocal_checkBox")
        self.gridLayout_2.addWidget(self.reciprocal_checkBox, 5, 0, 1, 5)
        self.space_group_Box = QtWidgets.QComboBox(self.groupBox)
        self.space_group_Box.setObjectName("space_group_Box")
        self.gridLayout_2.addWidget(self.space_group_Box, 7, 0, 1, 6)
        self.d_Slider = QtWidgets.QSlider(self.groupBox)
        self.d_Slider.setOrientation(QtCore.Qt.Horizontal)
        self.d_Slider.setObjectName("d_Slider")
        self.gridLayout_2.addWidget(self.d_Slider, 9, 2, 1, 4)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 1)
        self.gridLayout_7.addWidget(self.groupBox_6, 0, 1, 1, 1)
        StereoProj.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(StereoProj)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1403, 23))
        self.menubar.setObjectName("menubar")
        self.menuSave = QtWidgets.QMenu(self.menubar)
        self.menuSave.setObjectName("menuSave")
        self.menuStructure = QtWidgets.QMenu(self.menubar)
        self.menuStructure.setObjectName("menuStructure")
        self.menuAngle = QtWidgets.QMenu(self.menubar)
        self.menuAngle.setObjectName("menuAngle")
        self.menuSchmid_factor = QtWidgets.QMenu(self.menubar)
        self.menuSchmid_factor.setObjectName("menuSchmid_factor")
        self.menuXyz_directions = QtWidgets.QMenu(self.menubar)
        self.menuXyz_directions.setObjectName("menuXyz_directions")
        self.menuWidth = QtWidgets.QMenu(self.menubar)
        self.menuWidth.setObjectName("menuWidth")
        self.menuIntersections = QtWidgets.QMenu(self.menubar)
        self.menuIntersections.setObjectName("menuIntersections")
        self.menuDiffraction = QtWidgets.QMenu(self.menubar)
        self.menuDiffraction.setObjectName("menuDiffraction")
        self.menuList = QtWidgets.QMenu(self.menubar)
        self.menuList.setObjectName("menuList")
        self.menuIPF = QtWidgets.QMenu(self.menubar)
        self.menuIPF.setObjectName("menuIPF")
        StereoProj.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(StereoProj)
        self.statusbar.setObjectName("statusbar")
        StereoProj.setStatusBar(self.statusbar)
        self.actionSave_figure = QtWidgets.QAction(StereoProj)
        self.actionSave_figure.setObjectName("actionSave_figure")
        self.actionCalculate_Schmid_factor = QtWidgets.QAction(StereoProj)
        self.actionCalculate_Schmid_factor.setObjectName("actionCalculate_Schmid_factor")
        self.actionCalculate_angle = QtWidgets.QAction(StereoProj)
        self.actionCalculate_angle.setObjectName("actionCalculate_angle")
        self.actionCalculate_xyz = QtWidgets.QAction(StereoProj)
        self.actionCalculate_xyz.setObjectName("actionCalculate_xyz")
        self.actionCalculate_apparent_width = QtWidgets.QAction(StereoProj)
        self.actionCalculate_apparent_width.setObjectName("actionCalculate_apparent_width")
        self.actionPlanes = QtWidgets.QAction(StereoProj)
        self.actionPlanes.setObjectName("actionPlanes")
        self.actionProj_directions = QtWidgets.QAction(StereoProj)
        self.actionProj_directions.setObjectName("actionProj_directions")
        self.actionPlane_cone = QtWidgets.QAction(StereoProj)
        self.actionPlane_cone.setObjectName("actionPlane_cone")
        self.actionCalculate_intersections = QtWidgets.QAction(StereoProj)
        self.actionCalculate_intersections.setObjectName("actionCalculate_intersections")
        self.actionHkl_uvw = QtWidgets.QAction(StereoProj)
        self.actionHkl_uvw.setObjectName("actionHkl_uvw")
        self.actionPlot_Kikuchi_lines = QtWidgets.QAction(StereoProj)
        self.actionPlot_Kikuchi_lines.setObjectName("actionPlot_Kikuchi_lines")
        self.actionShow_list_of_poles_directions = QtWidgets.QAction(StereoProj)
        self.actionShow_list_of_poles_directions.setObjectName("actionShow_list_of_poles_directions")
        self.actiondraw_IPF = QtWidgets.QAction(StereoProj)
        self.actiondraw_IPF.setObjectName("actiondraw_IPF")
        self.menuSave.addAction(self.actionSave_figure)
        self.menuAngle.addAction(self.actionCalculate_angle)
        self.menuSchmid_factor.addAction(self.actionCalculate_Schmid_factor)
        self.menuXyz_directions.addAction(self.actionCalculate_xyz)
        self.menuXyz_directions.addAction(self.actionHkl_uvw)
        self.menuWidth.addAction(self.actionCalculate_apparent_width)
        self.menuIntersections.addAction(self.actionCalculate_intersections)
        self.menuDiffraction.addAction(self.actionPlot_Kikuchi_lines)
        self.menuList.addAction(self.actionShow_list_of_poles_directions)
        self.menuIPF.addAction(self.actiondraw_IPF)
        self.menubar.addAction(self.menuSave.menuAction())
        self.menubar.addAction(self.menuStructure.menuAction())
        self.menubar.addAction(self.menuAngle.menuAction())
        self.menubar.addAction(self.menuSchmid_factor.menuAction())
        self.menubar.addAction(self.menuXyz_directions.menuAction())
        self.menubar.addAction(self.menuList.menuAction())
        self.menubar.addAction(self.menuWidth.menuAction())
        self.menubar.addAction(self.menuIntersections.menuAction())
        self.menubar.addAction(self.menuDiffraction.menuAction())
        self.menubar.addAction(self.menuIPF.menuAction())

        self.retranslateUi(StereoProj)
        QtCore.QMetaObject.connectSlotsByName(StereoProj)
        StereoProj.setTabOrder(self.abc_entry, self.alphabetagamma_entry)
        StereoProj.setTabOrder(self.alphabetagamma_entry, self.e_entry)
        StereoProj.setTabOrder(self.e_entry, self.reciprocal_checkBox)
        StereoProj.setTabOrder(self.reciprocal_checkBox, self.space_group_Box)
        StereoProj.setTabOrder(self.space_group_Box, self.color_trace_bleu)
        StereoProj.setTabOrder(self.color_trace_bleu, self.color_trace_vert)
        StereoProj.setTabOrder(self.color_trace_vert, self.color_trace_rouge)
        StereoProj.setTabOrder(self.color_trace_rouge, self.style_box)
        StereoProj.setTabOrder(self.style_box, self.wulff_button)
        StereoProj.setTabOrder(self.wulff_button, self.real_space_checkBox)
        StereoProj.setTabOrder(self.real_space_checkBox, self.size_var)
        StereoProj.setTabOrder(self.size_var, self.text_size_entry)
        StereoProj.setTabOrder(self.text_size_entry, self.tilt_angle_entry)
        StereoProj.setTabOrder(self.tilt_angle_entry, self.image_angle_entry)
        StereoProj.setTabOrder(self.image_angle_entry, self.uvw_button)
        StereoProj.setTabOrder(self.uvw_button, self.hexa_button)
        StereoProj.setTabOrder(self.hexa_button, self.reset_view_button)
        StereoProj.setTabOrder(self.reset_view_button, self.diff_entry)
        StereoProj.setTabOrder(self.diff_entry, self.tilt_entry)
        StereoProj.setTabOrder(self.tilt_entry, self.inclinaison_entry)
        StereoProj.setTabOrder(self.inclinaison_entry, self.button_trace)
        StereoProj.setTabOrder(self.button_trace, self.rot_gm_button)
        StereoProj.setTabOrder(self.rot_gm_button, self.rot_g_entry)
        StereoProj.setTabOrder(self.rot_g_entry, self.rot_gp_button)
        StereoProj.setTabOrder(self.rot_gp_button, self.lock_checkButton)
        StereoProj.setTabOrder(self.lock_checkButton, self.angle_alpha_buttonm)
        StereoProj.setTabOrder(self.angle_alpha_buttonm, self.angle_alpha_entry)
        StereoProj.setTabOrder(self.angle_alpha_entry, self.angle_alpha_buttonp)
        StereoProj.setTabOrder(self.angle_alpha_buttonp, self.alpha_signBox)
        StereoProj.setTabOrder(self.alpha_signBox, self.angle_beta_buttonm)
        StereoProj.setTabOrder(self.angle_beta_buttonm, self.angle_beta_entry)
        StereoProj.setTabOrder(self.angle_beta_entry, self.angle_beta_buttonp)
        StereoProj.setTabOrder(self.angle_beta_buttonp, self.beta_signBox)
        StereoProj.setTabOrder(self.beta_signBox, self.angle_z_buttonm)
        StereoProj.setTabOrder(self.angle_z_buttonm, self.angle_z_entry)
        StereoProj.setTabOrder(self.angle_z_entry, self.angle_z_buttonp)
        StereoProj.setTabOrder(self.angle_z_buttonp, self.theta_signBox)
        StereoProj.setTabOrder(self.theta_signBox, self.reset_angle_button)
        StereoProj.setTabOrder(self.reset_angle_button, self.pole_entry)
        StereoProj.setTabOrder(self.pole_entry, self.addpole_button)
        StereoProj.setTabOrder(self.addpole_button, self.undo_addpole_button)
        StereoProj.setTabOrder(self.undo_addpole_button, self.sym_button)
        StereoProj.setTabOrder(self.sym_button, self.undo_sym_button)
        StereoProj.setTabOrder(self.undo_sym_button, self.trace_plan_button)
        StereoProj.setTabOrder(self.trace_plan_button, self.undo_trace_plan_button)
        StereoProj.setTabOrder(self.undo_trace_plan_button, self.trace_plan_sym_button)
        StereoProj.setTabOrder(self.trace_plan_sym_button, self.undo_trace_plan_sym_button)
        StereoProj.setTabOrder(self.undo_trace_plan_sym_button, self.trace_schmid_button)
        StereoProj.setTabOrder(self.trace_schmid_button, self.undo_trace_schmid)
        StereoProj.setTabOrder(self.undo_trace_schmid, self.trace_cone_button)
        StereoProj.setTabOrder(self.trace_cone_button, self.inclination_entry)
        StereoProj.setTabOrder(self.inclination_entry, self.undo_trace_cone_button)
        StereoProj.setTabOrder(self.undo_trace_cone_button, self.norm_button)
        StereoProj.setTabOrder(self.norm_button, self.phi1phiphi2_entry)
        StereoProj.setTabOrder(self.phi1phiphi2_entry, self.button_trace2)

    def retranslateUi(self, StereoProj):
        _translate = QtCore.QCoreApplication.translate
        StereoProj.setWindowTitle(_translate("StereoProj", "Stereo-Proj"))
        StereoProj.setAccessibleName(_translate("StereoProj", "Stereoproj"))
        self.centralwidget.setAccessibleName(_translate("StereoProj", "Stereoproj"))
        self.groupBox_2.setTitle(_translate("StereoProj", "Axis/Rotation"))
        self.rot_gm_button.setText(_translate("StereoProj", "-"))
        self.diff_label.setText(_translate("StereoProj", "g-vector"))
        self.inclination_label.setText(_translate("StereoProj", "Inclination"))
        self.rot_diff_label.setText(_translate("StereoProj", "Rotation along g"))
        self.button_trace.setText(_translate("StereoProj", "PLOT"))
        self.rot_gp_button.setText(_translate("StereoProj", "+"))
        self.tilt_label.setText(_translate("StereoProj", "Tilt α,β,θ"))
        self.groupBox_5.setTitle(_translate("StereoProj", "Euler Angles"))
        self.lab_euler2.setText(_translate("StereoProj", "φ 1 , Φ , φ2"))
        self.button_trace2.setText(_translate("StereoProj", "PLOT"))
        self.lab_coord.setText(_translate("StereoProj", "Tilt, Inclination"))
        self.groupBox_9.setTitle(_translate("StereoProj", "Layout"))
        self.real_space_checkBox.setText(_translate("StereoProj", "img/diff"))
        self.reset_view_button.setText(_translate("StereoProj", "Update/Reset view"))
        self.style_box.setText(_translate("StereoProj", "open/filled"))
        self.wulff_button.setText(_translate("StereoProj", "Wulff net"))
        self.color_trace_bleu.setText(_translate("StereoProj", "blue"))
        self.tilt_angle_label.setText(_translate("StereoProj", "diff α-tilt/y angle"))
        self.color_trace_vert.setText(_translate("StereoProj", "green"))
        self.text_size_label.setText(_translate("StereoProj", "Text size"))
        self.color_trace_rouge.setText(_translate("StereoProj", "red"))
        self.hexa_button.setText(_translate("StereoProj", "hexa"))
        self.label_2.setText(_translate("StereoProj", "img α-tilt/y angle"))
        self.uvw_button.setText(_translate("StereoProj", "uvw"))
        self.size_var_label.setText(_translate("StereoProj", "Marker size"))
        self.groupBox_4.setTitle(_translate("StereoProj", "Pole/Plane"))
        self.trace_plan_sym_button.setText(_translate("StereoProj", "Sym Plane"))
        self.sym_button.setText(_translate("StereoProj", "Symmetry"))
        self.undo_trace_cone_button.setText(_translate("StereoProj", "-"))
        self.trace_plan_button.setText(_translate("StereoProj", " Plane"))
        self.undo_trace_plan_sym_button.setText(_translate("StereoProj", "-"))
        self.trace_cone_button.setText(_translate("StereoProj", "Cone"))
        self.undo_trace_schmid.setText(_translate("StereoProj", "-"))
        self.norm_button.setText(_translate("StereoProj", "dhkl"))
        self.undo_trace_plan_button.setText(_translate("StereoProj", "-"))
        self.undo_addpole_button.setText(_translate("StereoProj", "-"))
        self.addpole_button.setText(_translate("StereoProj", "Add"))
        self.undo_sym_button.setText(_translate("StereoProj", "-"))
        self.trace_schmid_button.setText(_translate("StereoProj", "Schmid"))
        self.groupBox_3.setTitle(_translate("StereoProj", "Rotation"))
        self.angle_beta_label.setText(_translate("StereoProj", "<html><head/><body><p>β</p></body></html>"))
        self.angle_z_buttonm.setText(_translate("StereoProj", "-"))
        self.angle_beta_buttonp.setText(_translate("StereoProj", "+"))
        self.label.setText(_translate("StereoProj", "AC"))
        self.angle_beta_buttonm.setText(_translate("StereoProj", "-"))
        self.angle_z_label.setText(_translate("StereoProj", "<html><head/><body><p>θ</p></body></html>"))
        self.lock_checkButton.setText(_translate("StereoProj", "Lock Axes"))
        self.angle_alpha_buttonm.setText(_translate("StereoProj", "-"))
        self.angle_z_buttonp.setText(_translate("StereoProj", "+"))
        self.angle_alpha_buttonp.setText(_translate("StereoProj", "+"))
        self.angle_alpha_label.setText(_translate("StereoProj", "<html><head/><body><p>α</p></body></html>"))
        self.reset_angle_button.setText(_translate("StereoProj", "Reset"))
        self.groupBox.setTitle(_translate("StereoProj", "Crystal Parameters"))
        self.alphabetagamma_label.setText(_translate("StereoProj", "<p>&alpha;, &beta;, &gamma;</p>"))
        self.abc_label.setText(_translate("StereoProj", "a,b,c"))
        self.e_label.setText(_translate("StereoProj", "max indices"))
        self.d_label.setText(_translate("StereoProj", "d"))
        self.reciprocal_checkBox.setText(_translate("StereoProj", "Reciprocal indices"))
        self.menuSave.setTitle(_translate("StereoProj", "Save"))
        self.menuStructure.setTitle(_translate("StereoProj", "Structure"))
        self.menuAngle.setTitle(_translate("StereoProj", "Angle"))
        self.menuSchmid_factor.setTitle(_translate("StereoProj", "Schmid factor"))
        self.menuXyz_directions.setTitle(_translate("StereoProj", "xyz directions"))
        self.menuWidth.setTitle(_translate("StereoProj", "Width"))
        self.menuIntersections.setTitle(_translate("StereoProj", "intersections"))
        self.menuDiffraction.setTitle(_translate("StereoProj", "diffraction"))
        self.menuList.setTitle(_translate("StereoProj", "list"))
        self.menuIPF.setTitle(_translate("StereoProj", "PF/IPF"))
        self.actionSave_figure.setText(_translate("StereoProj", "Save figure"))
        self.actionCalculate_Schmid_factor.setText(_translate("StereoProj", "calculate Schmid factor"))
        self.actionCalculate_angle.setText(_translate("StereoProj", "Calculate angle"))
        self.actionCalculate_xyz.setText(_translate("StereoProj", "calculate xyz directions"))
        self.actionCalculate_apparent_width.setText(_translate("StereoProj", "Calculate apparent width"))
        self.actionPlanes.setText(_translate("StereoProj", "planes"))
        self.actionProj_directions.setText(_translate("StereoProj", "proj. directions"))
        self.actionPlane_cone.setText(_translate("StereoProj", "plane-cone"))
        self.actionCalculate_intersections.setText(_translate("StereoProj", "Calculate intersections"))
        self.actionHkl_uvw.setText(_translate("StereoProj", "hkl <> uvw"))
        self.actionPlot_Kikuchi_lines.setText(_translate("StereoProj", "plot Kikuchi lines or diffraction pattern"))
        self.actionShow_list_of_poles_directions.setText(_translate("StereoProj", "show list of poles/directions"))
        self.actiondraw_IPF.setText(_translate("StereoProj", "draw PF/IPF"))

