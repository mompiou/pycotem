from __future__ import division
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import os
import functools
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
import matplotlib as mpl
from . import misorientationUI


################
#       Misc
################

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def GCD(a, b, rtol=1e-05, atol=1e-08):
    t = min(abs(a), abs(b))
    while abs(b) > rtol * t + atol:
        a, b = b, a % b
    return a


# "
# Projection
####################################################################

def proj(x, y, z):

    if z == 1:
        X = 0
        Y = 0
    elif z < -0.000001:
        X = 'nan'
        Y = 'nan'
    else:

        X = x / (1 + z)
        Y = y / (1 + z)

    return np.array([X, Y], float)


def proj2(x, y, z):

    if z == 1:
        X = 0
        Y = 0
    elif z < -0.000001:
        X = -x / (1 - z)
        Y = -y / (1 - z)
    else:

        X = x / (1 + z)
        Y = y / (1 + z)

    return np.array([X, Y, z], float)


def proj_gnomonic(x, y, z):

    if z == 0:
        X = x
        Y = y

    else:

        X = x / z
        Y = y / z

    return np.array([X, Y], float)

# def proj_ortho(x,y,z):
#
#    return np.array([x,y],float)


###################################################################
# Rotation Euler
#
##################################################################

def rotation(phi1, phi, phi2):
    phi1 = phi1 * np.pi / 180
    phi = phi * np.pi / 180
    phi2 = phi2 * np.pi / 180
    R = np.array([[np.cos(phi1) * np.cos(phi2) - np.cos(phi) * np.sin(phi1) * np.sin(phi2),
                   -np.cos(phi) * np.cos(phi2) * np.sin(phi1) - np.cos(phi1) *
                   np.sin(phi2), np.sin(phi) * np.sin(phi1)], [np.cos(phi2) * np.sin(phi1)
                                                               + np.cos(phi) * np.cos(phi1) * np.sin(phi2), np.cos(phi) * np.cos(phi1)
                                                               * np.cos(phi2) - np.sin(phi1) * np.sin(phi2), -np.cos(phi1) * np.sin(phi)],
                  [np.sin(phi) * np.sin(phi2), np.cos(phi2) * np.sin(phi), np.cos(phi)]], float)
    return R

###################################################################

# Rotation around a given axis
#
##################################################################


def Rot(th, a, b, c):
    th = th * np.pi / 180
    no = np.linalg.norm([a, b, c])
    aa = a / no
    bb = b / no
    cc = c / no
    c1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    c2 = np.array([[aa**2, aa * bb, aa * cc], [bb * aa, bb**2, bb * cc], [cc * aa,
                                                                          cc * bb, cc**2]], float)
    c3 = np.array([[0, -cc, bb], [cc, 0, -aa], [-bb, aa, 0]], float)
    R = np.cos(th) * c1 + (1 - np.cos(th)) * c2 + np.sin(th) * c3

    return R

#######################
#
# Layout functions
#
#######################


def color_trace(i_c):

    if i_c == 2:
        if ui.color_trace_bleu_2.isChecked():
            color_trace = 1
        if ui.color_trace_vert_2.isChecked():
            color_trace = 2
        if ui.color_trace_rouge_2.isChecked():
            color_trace = 3
    else:
        if ui.color_trace_bleu.isChecked():
            color_trace = 1
        if ui.color_trace_vert.isChecked():
            color_trace = 2
        if ui.color_trace_rouge.isChecked():
            color_trace = 3
    return color_trace


def var_uvw():
    var_uvw = 0
    if ui.uvw_button.isChecked():
        var_uvw = 1

    return var_uvw


def var_hexa(i_c):
    if i_c == 2:
        var_hexa = 0
        if ui.hexa_button_2.isChecked():
            var_hexa = 1
    else:
        var_hexa = 0
        if ui.hexa_button.isChecked():
            var_hexa = 1
    return var_hexa


def var_carre(i_c):
    if i_c == 2:
        var_carre = 0
        if ui.style_box.isChecked():
            var_carre = 1
    else:
        var_carre = 0
        if ui.style_box_2.isChecked():
            var_carre = 1

    return var_carre

####################################################################
#
#  Crystal definition
#
##################################################################


def crist():
    global i_c, axes, axesh, D, Dstar, V, G, hexa_cryst1, hexa_cryst2, e, e2

    abc2 = ui.abc_entry_2.text().split(",")
    alphabetagamma2 = ui.alphabetagamma_entry_2.text().split(",")
    e2 = np.int(ui.e_entry_2.text())
    abc = ui.abc_entry.text().split(",")
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    e = np.int(ui.e_entry.text())
    hexa_cryst1 = 0

    if np.float(alphabetagamma[0]) == 90 and np.float(alphabetagamma[1]) == 90 and np.float(alphabetagamma[2]) == 120:
        hexa_cryst1 = 1
        dmip1 = np.float(abc[0]) / 2 - 0.0001e-10
    else:
        dmip1 = 0

    hexa_cryst2 = 0
    if np.float(alphabetagamma2[0]) == 90 and np.float(alphabetagamma2[1]) == 90 and np.float(alphabetagamma2[2]) == 120:
        hexa_cryst2 = 1
        dmip2 = np.float(abc2[0]) / 2 - 0.0001e-10
    else:
        dmip2 = 0

    ui.d1_Slider.setMinimum(0)
    ui.d1_Slider.setMaximum(np.amax([np.float(abc[0]), np.float(abc[1]), np.float(abc[2])]) * 100)
    ui.d1_Slider.setSingleStep(100)
    ui.d1_Slider.setValue(dmip1 * 100)
    ui.d_label_var.setText(str(np.around(dmip1, decimals=3)))

    ui.d2_Slider.setMinimum(0)
    ui.d2_Slider.setMaximum(np.amax([np.float(abc2[0]), np.float(abc2[1]), np.float(abc2[2])]) * 100)
    ui.d2_Slider.setSingleStep(100)
    ui.d2_Slider.setValue(dmip2 * 100)

    ui.d_label_var_2.setText(str(np.around(dmip2, decimals=3)))

    axes1, axesh1, D1, Dstar1, V1, G1 = crist_axes(abc, alphabetagamma, e, dmip1, hexa_cryst1, 1)
    axes2, axesh2, D2, Dstar2, V2, G2 = crist_axes(abc2, alphabetagamma2, e2, dmip2, hexa_cryst2, 2)
    if ui.crystal1_radioButton.isChecked():
        D, Dstar, V, G = D1, Dstar1, V1, G1
    if ui.crystal2_radioButton.isChecked():
        D, Dstar, V, G = D2, Dstar2, V2, G2
    axes = np.vstack((axes1, axes2))
    axesh = np.vstack((axesh1, axesh2))


def crist_mat(abc, alphabetagamma):
    a = np.float(abc[0]) * 1e-10
    b = np.float(abc[1]) * 1e-10
    c = np.float(abc[2]) * 1e-10
    alpha = np.float(alphabetagamma[0])
    beta = np.float(alphabetagamma[1])
    gamma = np.float(alphabetagamma[2])
    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180
    V = a * b * c * np.sqrt(1 - (np.cos(alpha)**2) - (np.cos(beta))**2 - (np.cos(gamma))**2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
    D = np.array([[a, b * np.cos(gamma), c * np.cos(beta)], [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)], [0, 0, V / (a * b * np.sin(gamma))]])
    Dstar = np.transpose(np.linalg.inv(D))
    G = np.array([[a**2, a * b * np.cos(gamma), a * c * np.cos(beta)], [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)], [a * c * np.cos(beta), b * c * np.cos(alpha), c**2]])
    return D, Dstar, V, G


def crist_axes(abc, alphabetagamma, e, dmip, hexa_cryst, i_c):

    D, Dstar, V, G = crist_mat(abc, alphabetagamma)
    axes = np.zeros(((2 * e + 1)**3 - 1, 4))
    axesh = np.zeros(((2 * e + 1)**3 - 1, 10))
    axesh[:, 4] = color_trace(i_c)
    axesh[:, 8] = var_hexa(i_c)
    axesh[:, 9] = var_carre(i_c)
    axesh[:, 7] = i_c
    axes[:, 3] = i_c

    id = 0
    for i in range(-e, e + 1):
        for j in range(-e, e + 1):
            for k in range(-e, e + 1):
                if (i, j, k) != (0, 0, 0):
                    if var_uvw() == 0:
                        Ma = np.dot(Dstar, np.array([i, j, k], float))
                        axesh[id, 3] = 0
                    else:
                        Ma = np.dot(D, np.array([i, j, k], float))
                        axesh[id, 3] = 1

                    m = np.abs(functools.reduce(lambda x, y: GCD(x, y), [i, j, k]))
                    if (np.around(i / m) == i / m) & (np.around(j / m) == j / m) & (np.around(k / m) == k / m):
                        axes[id, 0:3] = np.array([i, j, k]) / m
                    else:
                        axes[id, 0:3] = np.array([i, j, k])
                    axesh[id, 0:3] = Ma / np.linalg.norm(Ma)
                    axesh[id, 5] = 1
                    axesh[id, 6] = 1
                    id = id + 1

    axesh = axesh[~np.all(axesh[:, 0:3] == 0, axis=1)]
    axes = axes[~np.all(axes == 0, axis=1)]

    for i in range(0, np.shape(axes)[0]):
        axesh[i, 6] = 1
        d = 1 / (np.sqrt(np.dot(axes[i, 0:3], np.dot(np.linalg.inv(G), axes[i, 0:3])))) * 1e10
        if d < dmip:
            axesh[i, 6] = 0
        if (hexa_cryst == 1 and np.abs(axes[i, 0] + axes[i, 1]) > e):
            axesh[i, 6] = 0

    return axes, axesh, D, Dstar, V, G


######################################################
#
# Reduce number of poles/directions as a function of d-spacing (plus or minus)
#
#######################################################
def dist_restrict1():
    global G, axes, axesh, hexa_cryst1, e
    abc = ui.abc_entry.text().split(",")
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    d2 = ui.d1_Slider.value() / 100 * 1e-10
    D, Dstar, V, G = crist_mat(abc, alphabetagamma)

    for i in range(0, np.shape(axes)[0]):
        if axes[i, 3] == 1:
            axesh[i, 6] = 1
            d = 1 / (np.sqrt(np.dot(axes[i, 0:3], np.dot(np.linalg.inv(G), axes[i, 0:3]))))
            if d < d2:
                axesh[i, 6] = 0
            if (hexa_cryst1 == 1 and np.abs(axes[i, 0] + axes[i, 1]) > e):
                axesh[i, 6] = 0

    ui.d_label_var.setText(str(np.around(d2 * 1e10, decimals=3)))
    trace()


def dist_restrict2():
    global G, axes, axesh, hexa_cryst2, e2
    abc = ui.abc_entry_2.text().split(",")
    alphabetagamma = ui.alphabetagamma_entry_2.text().split(",")
    d2 = ui.d2_Slider.value() / 100 * 1e-10

    D, Dstar, V, G = crist_mat(abc, alphabetagamma)

    for i in range(0, np.shape(axes)[0]):
        if axes[i, 3] == 2:
            axesh[i, 6] = 1
            d = 1 / (np.sqrt(np.dot(axes[i, 0:3], np.dot(np.linalg.inv(G), axes[i, 0:3]))))
            if d < d2:
                axesh[i, 6] = 0
            if (hexa_cryst2 == 1 and np.abs(axes[i, 0] + axes[i, 1]) > e2):
                axesh[i, 6] = 0

    ui.d_label_var_2.setText(str(np.around(d2 * 1e10, decimals=3)))
    trace()

###########################################################################
#
# Rotation of the sample. If Lock Axes is off rotation are along y,x,z directions. If not, the y and z axes
# of the sample are locked to the crystal axes when the check box is ticked.
# It mimics double-tilt holder (rotation of alpha along fixed x and rotation of beta along the beta tilt moving axis)
# or  tilt-rotation holder  (rotation of alpha along fixed # x and rotation of z along the z-rotation moving axis).
#
##########################################################################


def euler_label():
    global M, M2
    if np.abs(M[2, 2] - 1) < 0.0001:
        phir = 0
        phi1r = 0
        phi2r = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
    else:
        phir = np.arccos(M[2, 2]) * 180 / np.pi
        phi2r = np.arctan2(M[2, 0], M[2, 1]) * 180 / np.pi
        phi1r = np.arctan2(M[0, 2], -M[1, 2]) * 180 / np.pi

    if np.abs(M2[2, 2] - 1) < 0.0001:
        phir_2 = 0
        phi1r_2 = 0
        phi2r_2 = np.arctan2(M2[1, 0], M2[0, 0]) * 180 / np.pi
    else:
        phir_2 = np.arccos(M2[2, 2]) * 180 / np.pi
        phi2r_2 = np.arctan2(M2[2, 0], M2[2, 1]) * 180 / np.pi
        phi1r_2 = np.arctan2(M2[0, 2], -M2[1, 2]) * 180 / np.pi

    t = str(np.around(phi1r, decimals=1)) + str(',') + str(np.around(phir, decimals=1)) + str(',') + str(np.around(phi2r, decimals=1))
    t2 = str(np.around(phi1r_2, decimals=1)) + str(',') + str(np.around(phir_2, decimals=1)) + str(',') + str(np.around(phi2r_2, decimals=1))

    ui.angle_euler_label.setText(t)
    ui.angle_euler_label_2.setText(t2)


def lock():
    global M, var_lock, M_lock
    i_c = 1
    if ui.crystal1_radioButton.isChecked():
        i_c = 2
    if ui.lock_checkButton.isChecked():
        var_lock = 1
        if i_c == 1:
            M_lock = M
        else:
            M_lock = M2
    else:
        var_lock, M_lock = 0, 0

    return var_lock, M_lock


def rot_alpha_p():
    global angle_alpha, M, M2, a, trP, trC, s_a
    t_ang = -ang_work_space()
    i_c = crystal_check()
    tha = s_a * np.float(ui.angle_alpha_entry.text())
    t_a_y = np.dot(Rot(t_ang, 0, 0, 1), np.array([0, 1, 0]))
    if i_c == 1:
        M = np.dot(Rot(tha, t_a_y[0], t_a_y[1], t_a_y[2]), M)
    else:
        M2 = np.dot(Rot(tha, t_a_y[0], t_a_y[1], t_a_y[2]), M2)
    trace()
    euler_label()

    angle_alpha = angle_alpha + np.float(ui.angle_alpha_entry.text())
    ui.angle_alpha_label_2.setText(str(angle_alpha))


def rot_alpha_m():
    global angle_alpha, M, M2, a, trP, trC, s_a
    t_ang = -ang_work_space()
    i_c = crystal_check()
    tha = -s_a * np.float(ui.angle_alpha_entry.text())
    t_a_y = np.dot(Rot(t_ang, 0, 0, 1), np.array([0, 1, 0]))
    if i_c == 1:
        M = np.dot(Rot(tha, t_a_y[0], t_a_y[1], t_a_y[2]), M)
    else:
        M2 = np.dot(Rot(tha, t_a_y[0], t_a_y[1], t_a_y[2]), M2)
    trace()
    euler_label()
    angle_alpha = angle_alpha - np.float(ui.angle_alpha_entry.text())
    ui.angle_alpha_label_2.setText(str(angle_alpha))


def rot_beta_m():
    global angle_beta, M, M2, angle_alpha, angle_z, var_lock, M_lock, s_b
    t_ang = -ang_work_space()
    i_c = crystal_check()
    t_a_x = np.dot(Rot(t_ang, 0, 0, 1), np.array([1, 0, 0]))

    if var_lock == 0:
        AxeY = t_a_x
    else:
        A = np.dot(np.linalg.inv(M_lock), t_a_x)
        if i_c == 1:
            AxeY = np.dot(M, A)
        else:
            AxeY = np.dot(M2, A)

    thb = -s_b * np.float(ui.angle_beta_entry.text())
    if i_c == 1:
        M = np.dot(Rot(thb, AxeY[0], AxeY[1], AxeY[2]), M)
    else:
        M2 = np.dot(Rot(thb, AxeY[0], AxeY[1], AxeY[2]), M2)
    trace()
    euler_label()
    angle_beta = angle_beta - np.float(ui.angle_beta_entry.text())
    ui.angle_beta_label_2.setText(str(angle_beta))


def rot_beta_p():
    global angle_beta, M, M2, angle_alpha, angle_z, var_lock, M_lock, s_b
    t_ang = -ang_work_space()
    i_c = crystal_check()
    t_a_x = np.dot(Rot(t_ang, 0, 0, 1), np.array([1, 0, 0]))
    if var_lock == 0:
        AxeY = t_a_x
    else:
        A = np.dot(np.linalg.inv(M_lock), t_a_x)
        if i_c == 1:
            AxeY = np.dot(M, A)
        else:
            AxeY = np.dot(M2, A)

    thb = s_b * np.float(ui.angle_beta_entry.text())
    if i_c == 1:
        M = np.dot(Rot(thb, AxeY[0], AxeY[1], AxeY[2]), M)
    else:
        M2 = np.dot(Rot(thb, AxeY[0], AxeY[1], AxeY[2]), M2)
    trace()
    euler_label()
    angle_beta = angle_beta + np.float(ui.angle_beta_entry.text())
    ui.angle_beta_label_2.setText(str(angle_beta))


def rot_z_m():
    global angle_beta, M, M2, angle_alpha, angle_z, var_lock, M_lock, s_z, t_ang

    i_c = crystal_check()

    if var_lock == 0:
        AxeZ = np.array([0, 0, 1])
    else:
        A = np.dot(np.linalg.inv(M_lock), np.array([0, 0, 1]))
        if i_c == 1:
            AxeZ = np.dot(M, A)
        else:
            AxeZ = np.dot(M2, A)

    thz = -s_z * np.float(ui.angle_z_entry.text())
    if i_c == 1:
        M = np.dot(Rot(thz, AxeZ[0], AxeZ[1], AxeZ[2]), M)
    else:
        M2 = np.dot(Rot(thz, AxeZ[0], AxeZ[1], AxeZ[2]), M2)
    trace()
    euler_label()
    angle_z = angle_z - np.float(ui.angle_z_entry.text())
    ui.angle_z_label_2.setText(str(angle_z))


def rot_z_p():
    global angle_beta, M, M2, angle_alpha, angle_z, var_lock, M_lock, s_z, t_ang
    i_c = crystal_check()

    if var_lock == 0:
        AxeZ = np.array([0, 0, 1])
    else:
        A = np.dot(np.linalg.inv(M_lock), np.array([0, 0, 1]))
        if i_c == 1:
            AxeZ = np.dot(M, A)
        else:
            AxeZ = np.dot(M2, A)

    thz = s_z * np.float(ui.angle_z_entry.text())
    if i_c == 1:
        M = np.dot(Rot(thz, AxeZ[0], AxeZ[1], AxeZ[2]), M)
    else:
        M2 = np.dot(Rot(thz, AxeZ[0], AxeZ[1], AxeZ[2]), M2)
    trace()
    euler_label()

    angle_z = angle_z + np.float(ui.angle_z_entry.text())
    ui.angle_z_label_2.setText(str(angle_z))


####################################################################
#
# Rotate around a given pole
#
####################################################################


def rotgm():
    global g, M, M2, Dstar, a, D
    i_c = crystal_check()
    thg = -np.float(ui.rot_g_entry.text())
    diff = ui.pole_entry.text().split(",")
    diff1 = np.float(diff[0])
    diff2 = np.float(diff[1])
    diff3 = np.float(diff[2])
    A = np.array([diff1, diff2, diff3])
    if var_uvw() == 1:
        if var_hexa(i_c) == 1:
            A = np.array([(2 * diff1 + diff2), (2 * diff2 + diff1), diff3])
        Ad = np.dot(D, A)
    else:
        Ad = np.dot(Dstar, A)
    if i_c == 1:
        Ap = np.dot(M, Ad) / np.linalg.norm(np.dot(M, Ad))
        M = np.dot(Rot(thg, Ap[0], Ap[1], Ap[2]), M)
    else:
        Ap = np.dot(M2, Ad) / np.linalg.norm(np.dot(M2, Ad))
        M2 = np.dot(Rot(thg, Ap[0], Ap[1], Ap[2]), M2)

    trace()
    euler_label()
    g = g - np.float(ui.rot_g_entry.text())
    ui.rg_label.setText(str(g))
    return g, M, M2


def rotgp():
    global g, M, M2, Dstar, D
    i_c = crystal_check()
    thg = np.float(ui.rot_g_entry.text())
    diff = ui.pole_entry.text().split(",")
    diff1 = np.float(diff[0])
    diff2 = np.float(diff[1])
    diff3 = np.float(diff[2])
    A = np.array([diff1, diff2, diff3])
    if var_uvw() == 1:
        if var_hexa(i_c) == 1:
            A = np.array([(2 * diff1 + diff2), (2 * diff2 + diff1), diff3])
        Ad = np.dot(D, A)
    else:
        Ad = np.dot(Dstar, A)
    if i_c == 1:
        Ap = np.dot(M, Ad) / np.linalg.norm(np.dot(M, Ad))
        M = np.dot(Rot(thg, Ap[0], Ap[1], Ap[2]), M)
    else:
        Ap = np.dot(M2, Ad) / np.linalg.norm(np.dot(M2, Ad))
        M2 = np.dot(Rot(thg, Ap[0], Ap[1], Ap[2]), M2)
    trace()
    euler_label()
    g = g + np.float(ui.rot_g_entry.text())
    ui.rg_label.setText(str(g))
    return g, M


####################################################################
#
# Add a given pole and equivalent ones
#
####################################################################

def pole(pole1, pole2, pole3):
    global M, M2, axes, axesh, T, V, D, Dstar
    i_c = crystal_check()
    if var_hexa(i_c) == 1:
        if var_uvw() == 1:
            pole1a = 2 * pole1 + pole2
            pole2a = 2 * pole2 + pole1
            pole1 = pole1a
            pole2 = pole2a

    Gs = np.array([pole1, pole2, pole3], float)

    if var_uvw() == 0:
        Gsh = np.dot(Dstar, Gs) / np.linalg.norm(np.dot(Dstar, Gs))
    else:
        Gsh = np.dot(D, Gs) / np.linalg.norm(np.dot(D, Gs))
    if i_c == 1:
        S = np.dot(M, Gsh)
    else:
        S = np.dot(M2, Gsh)
    if S[2] < 0:
        S = -S
        Gsh = -Gsh
        pole1 = -pole1
        pole2 = -pole2
        pole3 = -pole3

    T = np.vstack((T, np.array([S[0], S[1], S[2]])))
    axes = np.vstack((axes, np.array([pole1, pole2, pole3, i_c])))
    axes = np.vstack((axes, np.array([-pole1, -pole2, -pole3, i_c])))
    if var_uvw() == 0:
        axesh = np.vstack((axesh, np.array([Gsh[0], Gsh[1], Gsh[2], 0, color_trace(i_c), 0, 1, i_c, var_hexa(i_c), var_carre(i_c)])))
        axesh = np.vstack((axesh, np.array([-Gsh[0], -Gsh[1], -Gsh[2], 0, color_trace(i_c), 0, 1, i_c, var_hexa(i_c), var_carre(i_c)])))
    else:
        axesh = np.vstack((axesh, np.array([Gsh[0], Gsh[1], Gsh[2], 1, color_trace(i_c), 0, 1, i_c, var_hexa(i_c), var_carre(i_c)])))
        axesh = np.vstack((axesh, np.array([-Gsh[0], -Gsh[1], -Gsh[2], 1, color_trace(i_c), 0, 1, i_c, var_hexa(i_c), var_carre(i_c)])))

    return axes, axesh, T


def undo_pole(pole1, pole2, pole3):
    global M, M2, axes, axesh, T, V, D, Dstar

    i_c = crystal_check()
    if var_hexa(i_c) == 1:
        if var_uvw() == 1:
            pole1a = 2 * pole1 + pole2
            pole2a = 2 * pole2 + pole1
            pole1 = pole1a
            pole2 = pole2a

    Gs = np.array([pole1, pole2, pole3], float)

    if var_uvw() == 0:
        Gsh = np.dot(Dstar, Gs) / np.linalg.norm(np.dot(Dstar, Gs))
    else:
        Gsh = np.dot(D, Gs) / np.linalg.norm(np.dot(D, Gs))
    if i_c == 1:
        S = np.dot(M, Gsh)
    else:
        S = np.dot(M2, Gsh)
    if S[2] < 0:
        S = -S
        Gsh = -Gsh
        pole1 = -pole1
        pole2 = -pole2
        pole3 = -pole3

    ind = np.where((axes[:, 0] == pole1) & (axes[:, 1] == pole2) & (axes[:, 2] == pole3) & (axes[:, 3] == i_c) | (axes[:, 0] == -pole1) & (axes[:, 1] == -pole2) & (axes[:, 2] == -pole3) & (axes[:, 3] == i_c))
    axes = np.delete(axes, ind, 0)
    T = np.delete(T, ind, 0)
    axesh = np.delete(axesh, ind, 0)

    return axes, axesh, T


def d(pole1, pole2, pole3):
    global G

    ds = (np.sqrt(np.dot(np.array([pole1, pole2, pole3]), np.dot(np.linalg.inv(G), np.array([pole1, pole2, pole3])))))
    return ds


def addpole_sym():
    global M, M2, axes, axesh, T, V, D, Dstar, G
    i_c = crystal_check()
    pole_entry = ui.pole_entry.text().split(",")
    pole1 = np.float(pole_entry[0])
    pole2 = np.float(pole_entry[1])
    pole3 = np.float(pole_entry[2])
    v = d(pole1, pole2, pole3)
    pole(pole1, pole2, pole3)

    if var_hexa(i_c) == 1:
        pole(pole1, pole2, pole3)
        pole(pole1, pole2, -pole3)
        pole(pole2, pole1, pole3)
        pole(pole2, pole1, -pole3)
        pole(-pole1 - pole2, pole2, pole3)
        pole(-pole1 - pole2, pole2, -pole3)
        pole(pole1, -pole1 - pole2, pole3)
        pole(pole1, -pole1 - pole2, -pole3)
        pole(pole2, -pole1 - pole2, pole3)
        pole(pole2, -pole1 - pole2, -pole3)
        pole(-pole1 - pole2, pole1, pole3)
        pole(-pole1 - pole2, pole1, -pole3)

    else:
        if np.abs(d(pole1, pole2, -pole3) - v) < 0.001:
            pole(pole1, pole2, -pole3)
        if np.abs(d(pole1, -pole2, pole3) - v) < 0.001:
            pole(pole1, -pole2, pole3)
        if np.abs(d(-pole1, pole2, pole3) - v) < 0.001:
            pole(-pole1, pole2, pole3)
        if np.abs(d(pole2, pole1, pole3) - v) < 0.001:
            pole(pole2, pole1, pole3)
        if np.abs(d(pole2, pole1, -pole3) - v) < 0.001:
            pole(pole2, pole1, -pole3)
        if np.abs(d(pole2, -pole1, pole3) - v) < 0.001:
            pole(pole2, -pole1, pole3)
        if np.abs(d(-pole2, pole1, pole3) - v) < 0.001:
            pole(-pole2, pole1, pole3)
        if np.abs(d(pole2, pole3, pole1) - v) < 0.001:
            pole(pole2, pole3, pole1)
        if np.abs(d(pole2, pole3, -pole1) - v) < 0.001:
            pole(pole2, pole3, -pole1)
        if np.abs(d(pole2, -pole3, pole1) - v) < 0.001:
            pole(pole2, -pole3, pole1)
        if np.abs(d(-pole2, pole3, pole1) - v) < 0.001:
            pole(-pole2, pole3, pole1)
        if np.abs(d(pole1, pole3, pole2) - v) < 0.001:
            pole(pole1, pole3, pole2)
        if np.abs(d(pole1, pole3, -pole2) - v) < 0.001:
            pole(pole1, pole3, -pole2)
        if np.abs(d(pole1, -pole3, pole2) - v) < 0.001:
            pole(pole1, -pole3, pole2)
        if np.abs(d(-pole1, pole3, pole2) - v) < 0.001:
            pole(-pole1, pole3, pole2)
        if np.abs(d(pole3, pole1, pole2) - v) < 0.001:
            pole(pole3, pole1, pole2)
        if np.abs(d(pole3, pole1, -pole2) - v) < 0.001:
            pole(pole3, pole1, -pole2)
        if np.abs(d(pole3, -pole1, pole2) - v) < 0.001:
            pole(pole3, -pole1, pole2)
        if np.abs(d(-pole3, pole1, pole2) - v) < 0.001:
            pole(-pole3, pole1, pole2)
        if np.abs(d(pole3, pole2, pole1) - v) < 0.001:
            pole(pole3, pole2, pole1)
        if np.abs(d(pole3, pole2, -pole1) - v) < 0.001:
            pole(pole3, pole2, -pole1)
        if np.abs(d(pole3, -pole2, pole1) - v) < 0.001:
            pole(pole3, -pole2, pole1)
        if np.abs(d(-pole3, pole2, pole1) - v) < 0.001:
            pole(-pole3, pole2, pole1)
    trace()


def undo_sym():
    global M, M2, axes, axesh, T, V, D, Dstar, G
    i_c = crystal_check()
    pole_entry = ui.pole_entry.text().split(",")
    pole1 = np.float(pole_entry[0])
    pole2 = np.float(pole_entry[1])
    pole3 = np.float(pole_entry[2])
    v = d(pole1, pole2, pole3)
    undo_pole(pole1, pole2, pole3)
    if var_hexa(i_c) == 1:
        undo_pole(pole1, pole2, pole3)
        undo_pole(pole1, pole2, -pole3)
        undo_pole(pole2, pole1, pole3)
        undo_pole(pole2, pole1, -pole3)
        undo_pole(-pole1 - pole2, pole2, pole3)
        undo_pole(-pole1 - pole2, pole2, -pole3)
        undo_pole(pole1, -pole1 - pole2, pole3)
        undo_pole(pole1, -pole1 - pole2, -pole3)
        undo_pole(pole2, -pole1 - pole2, pole3)
        undo_pole(pole2, -pole1 - pole2, -pole3)
        undo_pole(-pole1 - pole2, pole1, pole3)
        undo_pole(-pole1 - pole2, pole1, -pole3)

    else:
        if np.abs(d(pole1, pole2, -pole3) - v) < 0.001:
            undo_pole(pole1, pole2, -pole3)
        if np.abs(d(pole1, -pole2, pole3) - v) < 0.001:
            undo_pole(pole1, -pole2, pole3)
        if np.abs(d(-pole1, pole2, pole3) - v) < 0.001:
            undo_pole(-pole1, pole2, pole3)
        if np.abs(d(pole2, pole1, pole3) - v) < 0.001:
            undo_pole(pole2, pole1, pole3)
        if np.abs(d(pole2, pole1, -pole3) - v) < 0.001:
            undo_pole(pole2, pole1, -pole3)
        if np.abs(d(pole2, -pole1, pole3) - v) < 0.001:
            undo_pole(pole2, -pole1, pole3)
        if np.abs(d(-pole2, pole1, pole3) - v) < 0.001:
            undo_pole(-pole2, pole1, pole3)
        if np.abs(d(pole2, pole3, pole1) - v) < 0.001:
            undo_pole(pole2, pole3, pole1)
        if np.abs(d(pole2, pole3, -pole1) - v) < 0.001:
            undo_pole(pole2, pole3, -pole1)
        if np.abs(d(pole2, -pole3, pole1) - v) < 0.001:
            undo_pole(pole2, -pole3, pole1)
        if np.abs(d(-pole2, pole3, pole1) - v) < 0.001:
            undo_pole(-pole2, pole3, pole1)
        if np.abs(d(pole1, pole3, pole2) - v) < 0.001:
            undo_pole(pole1, pole3, pole2)
        if np.abs(d(pole1, pole3, -pole2) - v) < 0.001:
            undo_pole(pole1, pole3, -pole2)
        if np.abs(d(pole1, -pole3, pole2) - v) < 0.001:
            undo_pole(pole1, -pole3, pole2)
        if np.abs(d(-pole1, pole3, pole2) - v) < 0.001:
            undo_pole(-pole1, pole3, pole2)
        if np.abs(d(pole3, pole1, pole2) - v) < 0.001:
            undo_pole(pole3, pole1, pole2)
        if np.abs(d(pole3, pole1, -pole2) - v) < 0.001:
            undo_pole(pole3, pole1, -pole2)
        if np.abs(d(pole3, -pole1, pole2) - v) < 0.001:
            undo_pole(pole3, -pole1, pole2)
        if np.abs(d(-pole3, pole1, pole2) - v) < 0.001:
            undo_pole(-pole3, pole1, pole2)
        if np.abs(d(pole3, pole2, pole1) - v) < 0.001:
            undo_pole(pole3, pole2, pole1)
        if np.abs(d(pole3, pole2, -pole1) - v) < 0.001:
            undo_pole(pole3, pole2, -pole1)
        if np.abs(d(pole3, -pole2, pole1) - v) < 0.001:
            undo_pole(pole3, -pole2, pole1)
        if np.abs(d(-pole3, pole2, pole1) - v) < 0.001:
            undo_pole(-pole3, pole2, pole1)
    trace()


def addpole():

    pole_entry = ui.pole_entry.text().split(",")
    pole1 = np.float(pole_entry[0])
    pole2 = np.float(pole_entry[1])
    pole3 = np.float(pole_entry[2])
    pole(pole1, pole2, pole3)
    trace()


def undo_addpole():
    global M, M2, axes, axesh, T, V, D, Dstar, trP
    pole_entry = ui.pole_entry.text().split(",")
    pole1 = np.float(pole_entry[0])
    pole2 = np.float(pole_entry[1])
    pole3 = np.float(pole_entry[2])
    undo_pole(pole1, pole2, pole3)
    trace()

####################################################################
#
# Plot a given plane and equivalent ones. Plot a cone
#
####################################################################


def trace_plan(pole1, pole2, pole3):
    global M, axes, axesh, T, V, D, Dstar, trP, trC
    i_c = crystal_check()
    pole_i = 0
    pole_c = color_trace(i_c)

    if var_hexa(i_c) == 1:
        if var_uvw() == 1:
            pole1 = 2 * pole1 + pole2
            pole2 = 2 * pole2 + pole1
            pole_i = 1
    trP = np.vstack((trP, np.array([pole1, pole2, pole3, pole_i, pole_c, i_c])))
    b = np.ascontiguousarray(trP).view(np.dtype((np.void, trP.dtype.itemsize * trP.shape[1])))
    trP = np.unique(b).view(trP.dtype).reshape(-1, trP.shape[1])


def trace_addplan():
    global M, M2, axes, axesh, T, V, D, Dstar, trP

    pole_entry = ui.pole_entry.text().split(",")
    pole1 = np.float(pole_entry[0])
    pole2 = np.float(pole_entry[1])
    pole3 = np.float(pole_entry[2])

    trace_plan(pole1, pole2, pole3)
    trace()


def undo_trace_addplan():
    global M, M2, axes, axesh, T, V, D, Dstar, trP

    pole_entry = ui.pole_entry.text().split(",")
    pole1 = np.float(pole_entry[0])
    pole2 = np.float(pole_entry[1])
    pole3 = np.float(pole_entry[2])

    undo_trace_plan(pole1, pole2, pole3)
    trace()


def undo_trace_plan(pole1, pole2, pole3):
    global M, M2, axes, axesh, T, V, D, Dstar, trP, tr_schmid
    i_c = crystal_check()
    ind = np.where((trP[:, 0] == pole1) & (trP[:, 1] == pole2) & (trP[:, 2] == pole3) | (trP[:, 0] == -pole1) & (trP[:, 1] == -pole2) & (trP[:, 2] == -pole3) & (trP[:, 3] == i_c))

    trP = np.delete(trP, ind, 0)
    b = np.ascontiguousarray(trP).view(np.dtype((np.void, trP.dtype.itemsize * trP.shape[1])))
    trP = np.unique(b).view(trP.dtype).reshape(-1, trP.shape[1])


def trace_plan_sym():
    global M, M2, axes, axesh, T, V, D, Dstar, G
    i_c = crystal_check()
    pole_entry = ui.pole_entry.text().split(",")
    pole1 = np.float(pole_entry[0])
    pole2 = np.float(pole_entry[1])
    pole3 = np.float(pole_entry[2])
    v = d(pole1, pole2, pole3)
    trace_plan(pole1, pole2, pole3)

    if var_hexa(i_c) == 1:
        trace_plan(pole1, pole2, pole3)
        trace_plan(pole1, pole2, -pole3)
        trace_plan(pole2, pole1, pole3)
        trace_plan(pole2, pole1, -pole3)
        trace_plan(-pole1 - pole2, pole2, pole3)
        trace_plan(-pole1 - pole2, pole2, -pole3)
        trace_plan(pole1, -pole1 - pole2, pole3)
        trace_plan(pole1, -pole1 - pole2, -pole3)
        trace_plan(pole2, -pole1 - pole2, pole3)
        trace_plan(pole2, -pole1 - pole2, -pole3)
        trace_plan(-pole1 - pole2, pole1, pole3)
        trace_plan(-pole1 - pole2, pole1, -pole3)

    else:
        if np.abs(d(pole1, pole2, -pole3) - v) < 0.001:
            trace_plan(pole1, pole2, -pole3)
        if np.abs(d(pole1, -pole2, pole3) - v) < 0.001:
            trace_plan(pole1, -pole2, pole3)
        if np.abs(d(-pole1, pole2, pole3) - v) < 0.001:
            trace_plan(-pole1, pole2, pole3)
        if np.abs(d(pole2, pole1, pole3) - v) < 0.001:
            trace_plan(pole2, pole1, pole3)
        if np.abs(d(pole2, pole1, -pole3) - v) < 0.001:
            trace_plan(pole2, pole1, -pole3)
        if np.abs(d(pole2, -pole1, pole3) - v) < 0.001:
            trace_plan(pole2, -pole1, pole3)
        if np.abs(d(-pole2, pole1, pole3) - v) < 0.001:
            trace_plan(-pole2, pole1, pole3)
        if np.abs(d(pole2, pole3, pole1) - v) < 0.001:
            trace_plan(pole2, pole3, pole1)
        if np.abs(d(pole2, pole3, -pole1) - v) < 0.001:
            trace_plan(pole2, pole3, -pole1)
        if np.abs(d(pole2, -pole3, pole1) - v) < 0.001:
            trace_plan(pole2, -pole3, pole1)
        if np.abs(d(-pole2, pole3, pole1) - v) < 0.001:
            trace_plan(-pole2, pole3, pole1)
        if np.abs(d(pole1, pole3, pole2) - v) < 0.001:
            trace_plan(pole1, pole3, pole2)
        if np.abs(d(pole1, pole3, -pole2) - v) < 0.001:
            trace_plan(pole1, pole3, -pole2)
        if np.abs(d(pole1, -pole3, pole2) - v) < 0.001:
            trace_plan(pole1, -pole3, pole2)
        if np.abs(d(-pole1, pole3, pole2) - v) < 0.001:
            trace_plan(-pole1, pole3, pole2)
        if np.abs(d(pole3, pole1, pole2) - v) < 0.001:
            trace_plan(pole3, pole1, pole2)
        if np.abs(d(pole3, pole1, -pole2) - v) < 0.001:
            trace_plan(pole3, pole1, -pole2)
        if np.abs(d(pole3, -pole1, pole2) - v) < 0.001:
            trace_plan(pole3, -pole1, pole2)
        if np.abs(d(-pole3, pole1, pole2) - v) < 0.001:
            trace_plan(-pole3, pole1, pole2)
        if np.abs(d(pole3, pole2, pole1) - v) < 0.001:
            trace_plan(pole3, pole2, pole1)
        if np.abs(d(pole3, pole2, -pole1) - v) < 0.001:
            trace_plan(pole3, pole2, -pole1)
        if np.abs(d(pole3, -pole2, pole1) - v) < 0.001:
            trace_plan(pole3, -pole2, pole1)
        if np.abs(d(-pole3, pole2, pole1) - v) < 0.001:
            trace_plan(-pole3, pole2, pole1)
    trace()


def undo_trace_plan_sym():
    global M, M2, axes, axesh, T, V, D, Dstar, G
    i_c = crystal_check()
    pole_entry = ui.pole_entry.text().split(",")
    pole1 = np.float(pole_entry[0])
    pole2 = np.float(pole_entry[1])
    pole3 = np.float(pole_entry[2])
    v = d(pole1, pole2, pole3)

    undo_trace_plan(pole1, pole2, pole3)
    if var_hexa(i_c) == 1:
        undo_trace_plan(pole1, pole2, pole3)
        undo_trace_plan(pole1, pole2, -pole3)
        undo_trace_plan(pole2, pole1, pole3)
        undo_trace_plan(pole2, pole1, -pole3)
        undo_trace_plan(-pole1 - pole2, pole2, pole3)
        undo_trace_plan(-pole1 - pole2, pole2, -pole3)
        undo_trace_plan(pole1, -pole1 - pole2, pole3)
        undo_trace_plan(pole1, -pole1 - pole2, -pole3)
        undo_trace_plan(pole2, -pole1 - pole2, pole3)
        undo_trace_plan(pole2, -pole1 - pole2, -pole3)
        undo_trace_plan(-pole1 - pole2, pole1, pole3)
        undo_trace_plan(-pole1 - pole2, pole1, -pole3)

    else:
        if np.abs(d(pole1, pole2, -pole3) - v) < 0.001:
            undo_trace_plan(pole1, pole2, -pole3)
        if np.abs(d(pole1, -pole2, pole3) - v) < 0.001:
            undo_trace_plan(pole1, -pole2, pole3)
        if np.abs(d(-pole1, pole2, pole3) - v) < 0.001:
            undo_trace_plan(-pole1, pole2, pole3)
        if np.abs(d(pole2, pole1, pole3) - v) < 0.001:
            undo_trace_plan(pole2, pole1, pole3)
        if np.abs(d(pole2, pole1, -pole3) - v) < 0.001:
            undo_trace_plan(pole2, pole1, -pole3)
        if np.abs(d(pole2, -pole1, pole3) - v) < 0.001:
            undo_trace_plan(pole2, -pole1, pole3)
        if np.abs(d(-pole2, pole1, pole3) - v) < 0.001:
            undo_trace_plan(-pole2, pole1, pole3)
        if np.abs(d(pole2, pole3, pole1) - v) < 0.001:
            undo_trace_plan(pole2, pole3, pole1)
        if np.abs(d(pole2, pole3, -pole1) - v) < 0.001:
            undo_trace_plan(pole2, pole3, -pole1)
        if np.abs(d(pole2, -pole3, pole1) - v) < 0.001:
            undo_trace_plan(pole2, -pole3, pole1)
        if np.abs(d(-pole2, pole3, pole1) - v) < 0.001:
            undo_trace_plan(-pole2, pole3, pole1)
        if np.abs(d(pole1, pole3, pole2) - v) < 0.001:
            undo_trace_plan(pole1, pole3, pole2)
        if np.abs(d(pole1, pole3, -pole2) - v) < 0.001:
            undo_trace_plan(pole1, pole3, -pole2)
        if np.abs(d(pole1, -pole3, pole2) - v) < 0.001:
            undo_trace_plan(pole1, -pole3, pole2)
        if np.abs(d(-pole1, pole3, pole2) - v) < 0.001:
            undo_trace_plan(-pole1, pole3, pole2)
        if np.abs(d(pole3, pole1, pole2) - v) < 0.001:
            undo_trace_plan(pole3, pole1, pole2)
        if np.abs(d(pole3, pole1, -pole2) - v) < 0.001:
            undo_trace_plan(pole3, pole1, -pole2)
        if np.abs(d(pole3, -pole1, pole2) - v) < 0.001:
            undo_trace_plan(pole3, -pole1, pole2)
        if np.abs(d(-pole3, pole1, pole2) - v) < 0.001:
            undo_trace_plan(-pole3, pole1, pole2)
        if np.abs(d(pole3, pole2, pole1) - v) < 0.001:
            undo_trace_plan(pole3, pole2, pole1)
        if np.abs(d(pole3, pole2, -pole1) - v) < 0.001:
            undo_trace_plan(pole3, pole2, -pole1)
        if np.abs(d(pole3, -pole2, pole1) - v) < 0.001:
            undo_trace_plan(pole3, -pole2, pole1)
        if np.abs(d(-pole3, pole2, pole1) - v) < 0.001:
            undo_trace_plan(-pole3, pole2, pole1)
    trace()


def trace_plan2(B):
    global M, M2, axes, axesh, T, V, D, Dstar, a, trP

    for h in range(0, B.shape[0]):
        pole1 = B[h, 0]
        pole2 = B[h, 1]
        pole3 = B[h, 2]
        Gs = np.array([pole1, pole2, pole3], float)

        if B[h, 3] == 0:
            Gsh = np.dot(Dstar, Gs) / np.linalg.norm(np.dot(Dstar, Gs))
        else:
            Gsh = np.dot(D, Gs) / np.linalg.norm(np.dot(D, Gs))
        if B[h, 4] == 1:
            S = np.dot(M, Gsh)
        else:
            S = np.dot(M2, Gsh)
        if S[2] < 0:
            S = -S
            Gsh = -Gsh
            pole1 = -pole1
            pole2 = -pole2
            pole3 = -pole3
        r = np.sqrt(S[0]**2 + S[1]**2 + S[2]**2)
        A = np.zeros((2, 100))
        Q = np.zeros((1, 2))
        t = np.arctan2(S[1], S[0]) * 180 / np.pi
        w = 0
        ph = np.arccos(S[2] / r) * 180 / np.pi

        for g in np.linspace(-np.pi, np.pi, 100):
            Aa = np.dot(Rot(t, 0, 0, 1), np.dot(Rot(ph, 0, 1, 0), np.array([np.sin(g), np.cos(g), 0])))
            if np.sign(Aa[2]) < 0 and np.abs(Aa[2]) < 1e-8:
                Aa = -Aa
            A[:, w] = proj(Aa[0], Aa[1], Aa[2]) * 300
            Q = np.vstack((Q, A[:, w]))
            w = w + 1

        Q = np.delete(Q, 0, 0)
        Q = Q[~np.isnan(Q).any(axis=1)]
        if B[h, 4] == 1:
            a.plot(Q[:, 0] + 300, Q[:, 1] + 300, 'b')
        if B[h, 4] == 2:
            a.plot(Q[:, 0] + 300, Q[:, 1] + 300, 'g')
        if B[h, 4] == 3:
            a.plot(Q[:, 0] + 300, Q[:, 1] + 300, 'r')


####################################################################
#
# Click a pole
#
####################################################################

def click_a_pole(event):

    global M, M2, Dstar, D, minx, maxx, miny, maxy, a, Stc
    i_c = crystal_check()
    if event.button == 3:
        x = event.xdata
        y = event.ydata

        x = (x - 300) / 300
        y = (y - 300) / 300
        X = 2 * x / (1 + x**2 + y**2)
        Y = 2 * y / (1 + x**2 + y**2)
        Z = (-1 + x**2 + y**2) / (1 + x**2 + y**2)
        if Z < 0:
            X = -X
            Y = -Y
        if i_c == 1:
            A = np.dot(np.linalg.inv(M), np.array([X, Y, Z]))
        else:
            A = np.dot(np.linalg.inv(M2), np.array([X, Y, Z]))
        if var_uvw() == 0:
            A = np.dot(np.linalg.inv(Dstar), A) * 1e10 * 100
        else:
            A = np.dot(np.linalg.inv(D), A) * 1e-10 * 100
            if var_hexa(i_c) == 1:
                Aa = (2 * A[0] - A[1]) / 3
                Ab = (2 * A[1] - A[0]) / 3
                A[0] = Aa
                A[1] = Ab

        pole(A[0], A[1], A[2])
        Stc = np.vstack((Stc, np.array([A[0], A[1], A[2]])))
        trace()


def undo_click_a_pole():
    global Stc
    undo_pole(Stc[-1, 0], Stc[-1, 1], Stc[-1, 2])
    Stc = Stc[:-1, :]
    trace()

####################################################################
#
# Inclinaison-beta indicator when the mouse is on the stereo
#
####################################################################


def coordinates(event):
    t_ang = ang_work_space() * np.pi / 180
    if event.xdata and event.ydata:
        x = event.xdata
        y = event.ydata
        x = (x - 300) / 300
        y = (y - 300) / 300
        X0 = 2 * x / (1 + x**2 + y**2)
        Y0 = 2 * y / (1 + x**2 + y**2)
        Z0 = (-1 + x**2 + y**2) / (1 + x**2 + y**2)
        Rxyz = np.dot(Rot(t_ang * 180 / np.pi, 0, 0, 1), [X0, Y0, Z0])
        X = Rxyz[0]
        Y = Rxyz[1]
        Z = Rxyz[2]
        lat = np.arctan2(np.sqrt(X**2 + Z**2), Y) * 180 / np.pi
        if X < 0:
            lat = -lat
        longi = -np.arctan2(Z, X) * 180 / np.pi
        if ui.alpha_signBox.isChecked():
            longi = -longi
        if np.abs(longi) > 90:
            if longi > 0:
                longi = longi - 180
            else:
                longi = longi + 180

        c = str(np.around(longi, decimals=1)) + str(',') + str(np.around(lat, decimals=1))
        ui.coord_label.setText(str(c))


####################################################################
#
# Reset view after zoom/update axes/angles
#
####################################################################


def reset_view():
    global a

    a.axis([minx, maxx, miny, maxy])
    mpl.rcParams['font.size'] = ui.text_size_entry.text()
    trace()


def tilt_axes():
    global s_a, s_b, s_z
    s_a, s_b, s_z = 1, 1, 1
    if ui.alpha_signBox.isChecked():
        s_a = -1
    if ui.beta_signBox.isChecked():
        s_b = -1
    if ui.theta_signBox.isChecked():
        s_b = -1
    return s_a, s_b, s_z

####################################################################
#
# Enable or disable Wulff net
#
####################################################################


def wulff():
    global a
    if ui.wulff_button.isChecked():
        fn = os.path.join(os.path.dirname(__file__), 'stereo.png')
        img = Image.open(fn)
        img = img.rotate(float(ang_work_space()), fillcolor='white')
        img = np.array(img)
    else:
        img = 255 * np.ones([600, 600, 3], dtype=np.uint8)
        circle = plt.Circle((300, 300), 300, color='black', fill=False)
        a.add_artist(circle)
        a.plot(300, 300, '+', markersize=10, mew=3, color='black')

    a.imshow(img, interpolation="bicubic")
    a.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.99, bottom=0.01, right=0.9, left=0.1)
    a.figure.canvas.draw()


def text_label(A, B):
    Aa = A[0]
    Ab = A[1]
    Ac = A[2]

    if B[3] == 1 & int(B[8]) == 1:
        Aa = (2 * A[0] - A[1])
        Ab = (2 * A[1] - A[0])
        Ac = 3 * A[2]
        m = functools.reduce(lambda x, y: GCD(x, y), [Aa, Ab, Ac])
        if np.abs(m) > 1e-3:
            Aa = Aa / m
            Ab = Ab / m
            Ac = Ac / m

    if np.sign(Aa) < 0:
        s0 = r'$\overline{' + str(np.abs(int(Aa))) + '}$'
    else:
        s0 = str(np.abs(int(Aa)))
    if np.sign(Ab) < 0:
        s1 = r'$\overline{' + str(np.abs(int(Ab))) + '}$'
    else:
        s1 = str(np.abs(int(Ab)))
    if np.sign(Ac) < 0:
        s2 = r'$\overline{' + str(np.abs(int(Ac))) + '}$'
    else:
        s2 = str(np.abs(int(Ac)))
    s = s0 + ',' + s1 + ',' + s2
    if int(B[8]) == 1:
        if np.sign(-Aa - Ab) < 0:
            s3 = r'$\overline{' + str(int(np.abs(-Aa - Ab))) + '}$'
        else:
            s3 = str(int(np.abs(-Aa - Ab)))
        s = s0 + ',' + s1 + ',' + s3 + ',' + s2
    if B[3] == 1:
        s = '[' + s + ']'
    return s


def crystal_check():
    global D, Dstar, V, G
    if ui.crystal2_radioButton.isChecked():
        i_c = 2
        abc = ui.abc_entry_2.text().split(",")
        alphabetagamma = ui.alphabetagamma_entry_2.text().split(",")
        D, Dstar, V, G = crist_mat(abc, alphabetagamma)
    else:
        i_c = 1
        abc = ui.abc_entry.text().split(",")
        alphabetagamma = ui.alphabetagamma_entry.text().split(",")
        D, Dstar, V, G = crist_mat(abc, alphabetagamma)
    return i_c


########################################################
# Main
#
#######################################################

########################################################
# Refresh action on stereo
#
#######################################################

def trace():
    global T, x, y, z, axes, axesh, M, M2, trP, a, trC, s_a, s_b, s_z, Qp, D1, D0, S
    minx, maxx = a.get_xlim()
    miny, maxy = a.get_ylim()
    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    P = np.zeros((axes.shape[0], 2))
    T = np.zeros((axes.shape[0], 3))
    trace_plan2(trP)
    C = []
    O = []
    tilt_axes()
    for i in range(0, axes.shape[0]):
        if axesh[i, 6] == 1:
            axeshr = np.array([axesh[i, 0], axesh[i, 1], axesh[i, 2]])
            if axesh[i, 7] == 1:
                T[i, :] = np.dot(M, axeshr)
            else:
                T[i, :] = np.dot(M2, axeshr)
            P[i, :] = proj(T[i, 0], T[i, 1], T[i, 2]) * 300
            s = text_label(axes[i, :], axesh[i, :])
            a.annotate(s, (P[i, 0] + 300, P[i, 1] + 300))
        if axesh[i, 4] == 1:
            C.append('b')
        if axesh[i, 4] == 2:
            C.append('g')
        if axesh[i, 4] == 3:
            C.append('r')
        if axesh[i, 9] == 1:
            O.append('none')
        if axesh[i, 9] == 0:
            O.append(C[-1])

    s0 = axesh[:, 6]
    a.scatter(P[:, 0] + 300, P[:, 1] + 300, edgecolors=C, s=s0 * np.float(ui.size_var.text()), facecolors=O, linewidth=1.5)
    trace_misorientation(Qp)
    a.axis([minx, maxx, miny, maxy])
    wulff()


# "
#
# Plot from Euler angle
#
# "

def princ2():
    global T, angle_alpha, angle_beta, angle_z, M, M2, Dstar, D, g, M0, trP, a, axeshr, minx, maxx, miny, maxy, trC, Stc, dmip, tr_schmid, s_a, s_b, s_c, axes, axesh, D, Dstar, V, G, Qp

    trP = np.zeros((1, 6))
    Stc = np.zeros((1, 3))
    Qp = np.zeros((1, 2))
    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    phi1phiphi2 = ui.phi1phiphi2_entry.text().split(",")
    phi1 = np.float(phi1phiphi2[0])
    phi = np.float(phi1phiphi2[1])
    phi2 = np.float(phi1phiphi2[2])

    phi1phiphi2_2 = ui.phi1phiphi2_2_entry.text().split(",")
    phi1_2 = np.float(phi1phiphi2_2[0])
    phi_2 = np.float(phi1phiphi2_2[1])
    phi2_2 = np.float(phi1phiphi2_2[2])

    crist()
    tilt_axes()

    P = np.zeros((axes.shape[0], 2))
    T = np.zeros((axes.shape[0], 3))
    C = []
    O = []

    for i in range(0, axes.shape[0]):
        axeshr = np.array([axesh[i, 0], axesh[i, 1], axesh[i, 2]])
        if axesh[i, 5] != -1 and axesh[i, 6] == 1:
            if axesh[i, 7] == 1:
                T[i, :] = np.dot(rotation(phi1, phi, phi2), axeshr)
            else:
                T[i, :] = np.dot(rotation(phi1_2, phi_2, phi2_2), axeshr)

            P[i, :] = proj(T[i, 0], T[i, 1], T[i, 2]) * 300
            s = text_label(axes[i, :], axesh[i, :])
            a.annotate(s, (P[i, 0] + 300, P[i, 1] + 300))
        if axesh[i, 4] == 1:
            C.append('b')
        if axesh[i, 4] == 2:
            C.append('g')
        if axesh[i, 4] == 3:
            C.append('r')
        if axesh[i, 9] == 1:
            O.append('none')
        if axesh[i, 9] == 0:
            O.append(C[-1])

    s0 = axesh[:, 6]
    a.scatter(P[:, 0] + 300, P[:, 1] + 300, edgecolors=C, s=s0 * np.float(ui.size_var.text()), facecolors=O, linewidth=1.5)

    minx, maxx = -2, 602
    miny, maxy = -2, 602
    a.axis([minx, maxx, miny, maxy])
    wulff()

    angle_alpha = 0
    angle_beta = 0
    angle_z = 0
    g = 0
    ui.angle_alpha_label_2.setText('0.0')
    ui.angle_beta_label_2.setText('0.0')
    ui.angle_z_label_2.setText('0.0')
    ui.angle_beta_label_2.setText('0.0')
    ui.angle_z_label_2.setText('0.0')
    M = rotation(phi1, phi, phi2)
    M2 = rotation(phi1_2, phi_2, phi2_2)
    t = str(np.around(phi1, decimals=1)) + str(',') + str(np.around(phi, decimals=1)) + str(',') + str(np.around(phi2, decimals=1))
    t2 = str(np.around(phi1_2, decimals=1)) + str(',') + str(np.around(phi_2, decimals=1)) + str(',') + str(np.around(phi2_2, decimals=1))
    ui.angle_euler_label.setText(t)
    ui.angle_euler_label_2.setText(t2)

    return T, angle_alpha, angle_beta, angle_z, g, M, M2

#####################################
#
# Misorientation
#
#####################################


def check_gb():
    abc2 = ui.abc_entry_2.text().split(",")
    alphabetagamma2 = ui.alphabetagamma_entry_2.text().split(",")
    abc = ui.abc_entry.text().split(",")
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    if abc == abc2 and alphabetagamma == alphabetagamma2:
        gb = 1
    else:
        gb = 2
    return gb


def Rota(t, u, v, w, g):
    Ae = np.dot(g, np.array([u, v, w]))
    Re = Rot(t, Ae[0], Ae[1], Ae[2])
    return Re


def cryststruct():
    global cs
    gb = check_gb()
    if gb == 1:
        abc = ui.abc_entry.text().split(",")
        alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    else:
        return
    a = np.float(abc[0]) * 1e-10
    b = np.float(abc[1]) * 1e-10
    c = np.float(abc[2]) * 1e-10
    alp = np.float(alphabetagamma[0])
    bet = np.float(alphabetagamma[1])
    gam = np.float(alphabetagamma[2])

    if gam == 90 and alp == 90 and bet == 90 and a == b and b == c:
        cs = 1

    if gam == 120 and alp == 90 and bet == 90:
        cs = 2

    if gam == 90 and alp == 90 and bet == 90 and a == b and b != c:
        cs = 3

    if alp != 90 and a == b and b == c:
        cs = 4

    if gam == 90 and alp == 90 and bet == 90 and a != b and b != c:
        cs = 5

    if gam != 90 and alp == 90 and bet == 90 and a != b and b != c:
        cs = 6

    if gam != alp != bet and a != b != c:
        cs = 7
    return cs


def Sy(g):
    global cs
    if cs == 1:
        S1 = Rota(90, 1, 0, 0, g)
        S2 = Rota(180, 1, 0, 0, g)
        S3 = Rota(270, 1, 0, 0, g)
        S4 = Rota(90, 0, 1, 0, g)
        S5 = Rota(180, 0, 1, 0, g)
        S6 = Rota(270, 0, 1, 0, g)
        S7 = Rota(90, 0, 0, 1, g)
        S8 = Rota(180, 0, 0, 1, g)
        S9 = Rota(270, 0, 0, 1, g)
        S10 = Rota(180, 1, 1, 0, g)
        S11 = Rota(180, 1, 0, 1, g)
        S12 = Rota(180, 0, 1, 1, g)
        S13 = Rota(180, -1, 1, 0, g)
        S14 = Rota(180, -1, 0, 1, g)
        S15 = Rota(180, 0, -1, 1, g)
        S16 = Rota(120, 1, 1, 1, g)
        S17 = Rota(240, 1, 1, 1, g)
        S18 = Rota(120, -1, 1, 1, g)
        S19 = Rota(240, -1, 1, 1, g)
        S20 = Rota(120, 1, -1, 1, g)
        S21 = Rota(240, 1, -1, 1, g)
        S22 = Rota(120, 1, 1, -1, g)
        S23 = Rota(240, 1, 1, -1, g)
        S24 = np.eye(3, 3)
        S = np.vstack((S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24))

    if cs == 2:
        S1 = Rota(60, 0, 0, 1, g)
        S2 = Rota(120, 0, 0, 1, g)
        S3 = Rota(180, 0, 0, 1, g)
        S4 = Rota(240, 0, 0, 1, g)
        S5 = Rota(300, 0, 0, 1, g)
        S6 = np.eye(3, 3)
        S7 = Rota(180, 1, 0, 0, g)
        S8 = Rota(180, 0, 1, 0, g)
        S9 = Rota(180, 1 / 2, np.sqrt(3) / 2, 0, g)
        S10 = Rota(180, -1 / 2, np.sqrt(3) / 2, 0, g)
        S11 = Rota(180, np.sqrt(3) / 2, 1 / 2, 0, g)
        S12 = Rota(180, -np.sqrt(3) / 2, 1 / 2, 0, g)
        S = np.vstack((S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12))

    if cs == 3:
        S1 = Rota(90, 0, 0, 1, g)
        S2 = Rota(180, 0, 0, 1, g)
        S3 = Rota(270, 0, 0, 1, g)
        S4 = Rota(180, 0, 1, 0, g)
        S5 = Rota(180, 1, 0, 0, g)
        S6 = Rota(180, 1, 1, 0, g)
        S7 = Rota(180, 1, -1, 0, g)
        S8 = np.eye(3, 3)
        S = np.vstack((S1, S2, S3, S4, S5, S6, S7, S8))

    if cs == 4:
        n1 = np.dot(Dstar, [1, 1, 1])
        n2 = np.dot(Dstar, [0, -1, 1])
        n3 = np.dot(Dstar, [-1, 0, 1])
        n4 = np.dot(Dstar, [1, -1, 0])
        S1 = Rota(120, n1[0], n1[1], n1[2], g)
        S2 = Rota(240, n1[0], n1[1], n1[2], g)
        S3 = Rota(180, n2[0], n2[1], n2[2], g)
        S4 = Rota(180, n3[0], n3[1], n3[2], g)
        S5 = Rota(180, n4[0], n4[1], n4[2], g)
        S6 = np.eye(3, 3)
        S = np.vstack((S1, S2, S3, S4, S5, S6))

    if cs == 5:
        S1 = Rota(180, 0, 0, 1, g)
        S2 = Rota(180, 1, 0, 0, g)
        S3 = Rota(180, 0, 1, 0, g)
        S4 = np.eye(3, 3)
        S = np.vstack((S1, S2, S3, S4))

    if cs == 6:
        S1 = Rota(180, 0, 0, 1, g)
        S2 = np.eye(3, 3)
        S = np.vstack((S1, S2))

    if cs == 7:
        S = np.eye(3, 3)

    return S


def null(A, rcond=None):

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q


def desorientation():
    global cs, Qp, M, M2, D1, D0, S, Dstar, D, D2

    cryststruct()
    gA = M
    gB = M2
    k = 0
    S = Sy(gA)

    D0 = np.zeros((int(np.shape(S)[0] / 3) + 1, 5))
    D1 = np.zeros((int(np.shape(S)[0] / 3) + 1, 3))
    D2 = np.zeros((int(np.shape(S)[0] / 3) + 1, 3))
    Qp = np.zeros((int(np.shape(S)[0] / 3) + 1, 2))

    for i in range(0, np.shape(S)[0], 3):
        In = np.dot(np.array([[S[i, 0], S[i + 1, 0], S[i + 2, 0]], [S[i, 1], S[i + 1, 1], S[i + 2, 1]], [S[i, 2], S[i + 1, 2], S[i + 2, 2]]]), gA)
        A = np.dot(In, np.linalg.inv(gB)) - np.eye(3)
        V = null(A, 0.001).T

        if 0.5 * (np.trace(A + np.eye(3)) - 1) > 1:
            D0[k, 3] = 0
        elif 0.5 * (np.trace(A + np.eye(3)) - 1) < -1:
            D0[k, 3] = 180
        else:
            D0[k, 3] = np.arccos(0.5 * (np.trace(A + np.eye(3)) - 1)) * 180 / np.pi

        if np.abs(D0[k, 3]) < 1e-5:
            D0[k, 0] = 0
            D0[k, 1] = 0
            D0[k, 2] = 0
        else:
            D0[k, 0] = V[0, 0] / np.linalg.norm(V)
            D0[k, 1] = V[0, 1] / np.linalg.norm(V)
            D0[k, 2] = V[0, 2] / np.linalg.norm(V)

        Ds1 = np.dot(np.linalg.inv(gA), np.array([D0[k, 0], D0[k, 1], D0[k, 2]]))
        Ds2 = np.dot(np.linalg.inv(gB), np.array([D0[k, 0], D0[k, 1], D0[k, 2]]))

        if var_uvw() == 1:
            Ds1 = np.dot(np.linalg.inv(D) * 1e-10, Ds1)
            Ds2 = np.dot(np.linalg.inv(D) * 1e-10, Ds2)
        else:
            Ds1 = np.dot(np.linalg.inv(Dstar) * 1e10, Ds1)
            Ds2 = np.dot(np.linalg.inv(Dstar) * 1e10, Ds2)

        D1[k, 0:3] = 100 * Ds1 / np.linalg.norm(Ds1)
        D2[k, 0:3] = 100 * Ds2 / np.linalg.norm(Ds2)

        if D0[k, 2] < 0:
            D0[k, 0] = -D0[k, 0]
            D0[k, 1] = -D0[k, 1]
            D0[k, 2] = -D0[k, 2]
            D1[k, 0] = -D1[k, 0]
            D1[k, 1] = -D1[k, 1]
            D1[k, 2] = -D1[k, 2]

        D0[k, 4] = k
        Qp[k, :] = proj(D0[k, 0], D0[k, 1], D0[k, 2]) * 300

        k = k + 1

    trace()


def trace_misorientation(B):
    global Qp, D1, D0, D2

    i_c = crystal_check()
    ui.misorientation_list.clear()
    sepg = '('
    sepd = ')'
    if var_uvw() == 1:
        sepg = '['
        sepd = ']'
    if Qp.shape[0] > 1:
        ui.misorientation_list.setRowCount(int(np.shape(S)[0] / 3))
        ui.misorientation_list.setColumnCount(3)
        a.plot(B[:-1, 0] + 300, B[:-1, 1] + 300, 's', color='black')
        for l in range(0, int(np.shape(S)[0] / 3)):
            if var_hexa(i_c) == 1 and var_uvw() == 1:
                if ui.axis_checkBox.isChecked():
                    saxe = sepg + str(int((2 * D1[l, 0] - D1[l, 1]) / 3)) + ',' + str(int((2 * D1[l, 1] - D1[l, 0]) / 3)) + ',' + str(int(D1[l, 2])) + sepd
                    a.annotate(saxe, (B[l, 0] + 300, B[l, 1] + 300), size=ui.text_size_entry.text())

                axe1 = QtWidgets.QTableWidgetItem(sepg + str(int((2 * D1[l, 0] - D1[l, 1]) / 3)) + ',' + str(int((2 * D1[l, 1] - D1[l, 0]) / 3)) + ',' + str(int(D1[l, 2])) + sepd)
                axe2 = QtWidgets.QTableWidgetItem(sepg + str(int((2 * D2[l, 0] - D2[l, 1]) / 3)) + ',' + str(int((2 * D2[l, 1] - D2[l, 0]) / 3)) + ',' + str(int(D2[l, 2])) + sepd)
                angle = QtWidgets.QTableWidgetItem(str(np.around(D0[l, 3], decimals=2)))

            else:
                if ui.axis_checkBox.isChecked():
                    saxe = sepg + str(int(D1[l, 0])) + ',' + str(int(D1[l, 1])) + ',' + str(int(D1[l, 2])) + sepd

                    a.annotate(saxe, (B[l, 0] + 300, B[l, 1] + 300), size=ui.text_size_entry.text())

                axe1 = QtWidgets.QTableWidgetItem(sepg + str(int(D1[l, 0])) + ',' + str(int(D1[l, 1])) + ',' + str(int(D1[l, 2])) + sepd)
                axe2 = QtWidgets.QTableWidgetItem(sepg + str(int(D2[l, 0])) + ',' + str(int(D2[l, 1])) + ',' + str(int(D2[l, 2])) + sepd)
                angle = QtWidgets.QTableWidgetItem(str(np.around(D0[l, 3], decimals=2)))

            if ui.numbers_checkBox.isChecked():
                snum = str(int(D0[l, 4]) + 1)
                a.annotate(snum, (B[l, 0] + 300, B[l, 1] + 300), size=ui.text_size_entry.text())

            ui.misorientation_list.setHorizontalHeaderLabels(['Axe 1', 'Axe 2', 'Angle'])
            ui.misorientation_list.setItem(l, 0, axe1)
            ui.misorientation_list.setItem(l, 1, axe2)
            ui.misorientation_list.setItem(l, 2, angle)


def desorientation_clear():
    global Qp
    Qp = np.zeros((0, 2))
    trace()

#######################################################################
#######################################################################
#
# GUI Menu/Dialog
#
#######################################################################


######################################################
#
# Menu
#
##########################################################

###########################################################
#
# Structure
#
##############################################################

def structure(item):
    global x0, var_hexa, d_label_var, e_entry
    item = x0[item - 1]
    ui.abc_entry.setText(str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]))
    ui.alphabetagamma_entry.setText(str(item[4]) + ',' + str(item[5]) + ',' + str(item[6]))
    if eval(item[4]) == 90 and eval(item[5]) == 90 and eval(item[6]) == 120:
        ui.hexa_button.setChecked(True)
        ui.e_entry.setText('2')
    else:
        ui.e_entry.setText('1')
        ui.hexa_button.setChecked(False)


def structure2(item):
    global x0, var_hexa, d_label_var_2, e_entry_2
    item = x0[item - 1]
    ui.abc_entry_2.setText(str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]))
    ui.alphabetagamma_entry_2.setText(str(item[4]) + ',' + str(item[5]) + ',' + str(item[6]))
    if eval(item[4]) == 90 and eval(item[5]) == 90 and eval(item[6]) == 120:
        ui.hexa_button_2.setChecked(True)
        ui.e_entry_2.setText('2')
    else:
        ui.e_entry_2.setText('1')
        ui.hexa_button_2.setChecked(False)

#######################################
#
# Save stereo as png
#
######################################


def image_save():
    filename = QtWidgets.QFileDialog.getSaveFileName(Index, "Save file", "", ".png")
    f = str(filename[0]) + ".png"
    canvas.print_figure(f)


####################
#
# define work space (real or reciprocal) to take tilt/y axis angles into account
#
######################

def ang_work_space():
    if ui.real_space_checkBox.isChecked():
        t_ang = np.float(ui.image_angle_entry.text())
    else:
        t_ang = np.float(ui.tilt_angle_entry.text())
    return t_ang

##################################################
#
# Add matplotlib toolbar to zoom and pan
#
###################################################


class NavigationToolbar(NavigationToolbar):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Pan', 'Zoom')]

    def set_message(self, msg):
        pass


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

#############################################################
#
# Launch
#
# "


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

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    sys.excepthook = except_hook
    QtWidgets.qApp.setApplicationName("Misorientation")
    Index = QtWidgets.QMainWindow()
    ui = misorientationUI.Ui_Misorientation()
    ui.setupUi(Index)
    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui.mplvl.addWidget(canvas)
    toolbar = NavigationToolbar(canvas, canvas)
    toolbar.setMinimumWidth(601)

# Read structure file

    file_struct = open(os.path.join(os.path.dirname(__file__), 'structure.txt'), "r")
    x0 = []
    for line in file_struct:
        x0.append(list(map(str, line.split())))
    i = 0
    file_struct.close()
    ui.structure_box.addItem(' ')
    ui.structure2_box.addItem(' ')
    for item in x0:
        ui.structure_box.addItem(item[0])
        ui.structure2_box.addItem(item[0])
        i = i + 1

    ui.structure_box.currentIndexChanged.connect(structure)
    ui.structure2_box.currentIndexChanged.connect(structure2)

# Read space_group file


# Ctrl+z shortcut to remove clicked pole

    shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+z"), Index)
    shortcut.activated.connect(undo_click_a_pole)

# Connect dialog boxes and buttons

    ui.actionSave_figure.triggered.connect(image_save)

    ui.button_trace2.clicked.connect(princ2)
    ui.angle_alpha_buttonp.clicked.connect(rot_alpha_p)
    ui.angle_alpha_buttonm.clicked.connect(rot_alpha_m)
    ui.angle_beta_buttonp.clicked.connect(rot_beta_p)
    ui.angle_beta_buttonm.clicked.connect(rot_beta_m)
    ui.angle_z_buttonp.clicked.connect(rot_z_p)
    ui.angle_z_buttonm.clicked.connect(rot_z_m)
    ui.lock_checkButton.stateChanged.connect(lock)
    ui.rot_gm_button.clicked.connect(rotgm)
    ui.rot_gp_button.clicked.connect(rotgp)
    ui.addpole_button.clicked.connect(addpole)
    ui.undo_addpole_button.clicked.connect(undo_addpole)
    ui.sym_button.clicked.connect(addpole_sym)
    ui.undo_sym_button.clicked.connect(undo_sym)
    ui.trace_plan_button.clicked.connect(trace_addplan)
    ui.undo_trace_plan_button.clicked.connect(undo_trace_addplan)
    ui.trace_plan_sym_button.clicked.connect(trace_plan_sym)
    ui.undo_trace_plan_sym_button.clicked.connect(undo_trace_plan_sym)
    ui.reset_view_button.clicked.connect(reset_view)
    figure.canvas.mpl_connect('motion_notify_event', coordinates)
    figure.canvas.mpl_connect('button_press_event', click_a_pole)
    ui.misorientation_button.clicked.connect(desorientation)
    ui.clear_misorientation_button.clicked.connect(desorientation_clear)
    ui.d1_Slider.sliderReleased.connect(dist_restrict1)
    ui.d2_Slider.sliderReleased.connect(dist_restrict2)

# Initialize variables

    dmip = 0
    dmip2 = 0
    var_lock = 0
    ui.lock_checkButton.setChecked(False)
    ui.crystal1_radioButton.setChecked(True)
    ui.color_trace_bleu.setChecked(True)
    ui.color_trace_rouge_2.setChecked(True)
    ui.wulff_button.setChecked(True)
    ui.d_label_var.setText('0')
    ui.d_label_var_2.setText('0')
    ui.text_size_entry.setText('12')
    mpl.rcParams['font.size'] = ui.text_size_entry.text()
    ui.abc_entry.setText('1,1,1')
    ui.alphabetagamma_entry.setText('90,90,90')
    ui.abc_entry_2.setText('1,1,1')
    ui.alphabetagamma_entry_2.setText('90,90,90')
    ui.phi1phiphi2_entry.setText('0,0,0')
    ui.phi1phiphi2_2_entry.setText('0,0,0')

    ui.e_entry.setText('1')
    ui.angle_euler_label.setText(' ')
    ui.size_var.setText('40')
    ui.e_entry_2.setText('1')
    ui.angle_alpha_entry.setText('5')
    ui.angle_beta_entry.setText('5')
    ui.angle_z_entry.setText('5')
    ui.angle_beta_entry.setText('5')
    ui.angle_z_entry.setText('5')
    ui.rot_g_entry.setText('5')
    ui.tilt_angle_entry.setText('0')
    ui.image_angle_entry.setText('0')
    a = figure.add_subplot(111)
    tilt_axes()
    wulff()
    Index.show()
    sys.exit(app.exec_())
