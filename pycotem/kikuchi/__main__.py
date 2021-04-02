##################
#
# Kikuchi is a tool to orient crystals from kikuchi lines in TEM
#
##################


from __future__ import division
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import os
from PIL import Image
from PIL import ImageEnhance
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
from . import kikuchiUI
from . import refineUI
from . import tiltUI

######################
#
# click event control. Prevent to define a point when zooming
#
######################


def onpress(event):
    global press, move
    press = True


def onmove(event):
    global press, move
    if press:
        move = True


def onrelease(event):
    global press, move

    if press and not move:
        click(event)
    press = False
    move = False

######################################
#
# Store clicked positions on the image
#
######################################


def click(event):
    global gclick, s, minx, maxx, miny, maxy

    a = figure.add_subplot(111)
    minx, maxx = a.get_xlim()
    miny, maxy = a.get_ylim()
    x = event.xdata
    y = event.ydata
    a.annotate(str(s), (x, y))
    a.plot(x, y, 'b+')
    a.axis('off')
    a.figure.canvas.draw()
    s = s + 1
    gclick = np.vstack((gclick, np.array([x, y])))

    return gclick, s


########################
#
# Erase clicked points , and center
#
#####################

def reset_points():
    global image_diff, gclick, minx, maxx, miny, maxy, s

    brightness()
    a = figure.add_subplot(111)
    a.axis([minx, maxx, miny, maxy])
    a.axis('off')
    a.figure.canvas.draw()
    s = 1
    gclick = np.zeros((1, 2))
    ui.ListBox_theo.clear()
    return s, gclick


def reset():
    global image_diff, gclick, s, minx, maxx, miny, maxy

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    img = Image.open(str(image_diff[0]))
    img = np.array(img)
    figure.suptitle(str(image_diff[0]))
    a.imshow(img, origin="upper")
    minx = 0
    maxx = width
    miny = height
    maxy = 0
    a.axis([minx, maxx, miny, maxy])
    a.axis('off')
    a.figure.canvas.draw()
    gclick = np.zeros((1, 2))
    ui.ListBox_theo.clear()
    brightness()
    s = 1
    return s, gclick


def remove_center():
    ui.centerLine.clear()

################################################################
#
# Determine interplanar distances from the crystal parameters
# and the space group
# Return a list of h,k,l indices close to the measured distance d  in Dist
# Return also the angles between hkl directions Ang
#
###################################################################


def listb():
    global Dist, Ang, G, Dstar
    abc = ui.abc_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alp = np.float(alphabetagamma[0])
    bet = np.float(alphabetagamma[1])
    gam = np.float(alphabetagamma[2])
    e = np.int(ui.indice_entry.text())
    alp = alp * np.pi / 180
    bet = bet * np.pi / 180
    gam = gam * np.pi / 180
    Dist = np.zeros(((2 * e + 1)**3 - 1, 5))
    G = np.array([[a**2, a * b * np.cos(gam), a * c * np.cos(bet)], [a * b * np.cos(gam), b**2, b * c * np.cos(alp)], [a * c * np.cos(bet), b * c * np.cos(alp), c**2]])
    V = a * b * c * np.sqrt(1 - (np.cos(alp)**2) - (np.cos(bet))**2 - (np.cos(gam))**2 + 2 * np.cos(alp) * np.cos(bet) * np.cos(gam))
    D = np.array([[a, b * np.cos(gam), c * np.cos(bet)], [0, b * np.sin(gam), c * (np.cos(alp) - np.cos(bet) * np.cos(gam)) / np.sin(gam)], [0, 0, V / (a * b * np.sin(gam))]])
    Dstar = np.transpose(np.linalg.inv(D))
    w = 0
    for i in range(-e, e + 1):
        for j in range(-e, e + 1):
            for k in range(-e, e + 1):
                if (i, j, k) != (0, 0, 0):
                    di = 1 / (np.sqrt(np.dot(np.array([i, j, k]), np.dot(np.linalg.inv(G), np.array([i, j, k])))))
                    I = extinction(ui.space_group_Box.currentText(), i, j, k)
                    if I != 0:
                        Dist[w, :] = np.array([np.around(di, decimals=3), i, j, k, I])
                        w = w + 1

    Dist = Dist[Dist[:, 0].argsort()]
    Dist = Dist[~np.all(Dist == 0, axis=1)]
    Dist = Dist[::-1]

    lenD = np.shape(Dist)[0]
    Ang = np.zeros((lenD, lenD))

    for i in range(lenD):
        T = np.zeros(lenD)
        for j in range(lenD):
            c1c = np.dot(Dstar, Dist[i, 1:4])
            c2c = np.dot(Dstar, Dist[j, 1:4])
            T[j] = np.around(np.arccos(np.dot(c1c, c2c) / (np.linalg.norm(c1c) * np.linalg.norm(c2c))) * 180 / np.pi, decimals=3)

        Ang[i, :] = T

    return Dist, Ang

##################################
# Return interplanar distance
##################################


def distance(i, j, k):
    abc = ui.abc_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alp = np.float(alphabetagamma[0])
    bet = np.float(alphabetagamma[1])
    gam = np.float(alphabetagamma[2])
    alp = alp * np.pi / 180
    bet = bet * np.pi / 180
    gam = gam * np.pi / 180
    G = np.array([[a**2, a * b * np.cos(gam), a * c * np.cos(bet)], [a * b * np.cos(gam), b**2, b * c * np.cos(alp)], [a * c * np.cos(bet), b * c * np.cos(alp), c**2]])
    di = 1 / (np.sqrt(np.dot(np.array([i, j, k]), np.dot(np.linalg.inv(G), np.array([i, j, k])))))

    return di


#############################################
#
# Select the band and store it in B2 array. From the points determine m, b coefficient for the line equation y=mx+d0. d is the calibrated distance
#
#########################################################################

def addband():
    global gclick, d, counter, x_calib, height, B2

    Pt1 = gclick[1]
    Pt1[1] = -Pt1[1] + height
    Pt2 = gclick[2]
    Pt2[1] = -Pt2[1] + height
    Pt3 = gclick[3]
    Pt3[1] = -Pt3[1] + height
    m = (Pt2[1] - Pt1[1]) / (Pt2[0] - Pt1[0])
    b = (Pt1[1] * Pt2[0] - Pt2[1] * Pt1[0]) / (Pt2[0] - Pt1[0])
    d0 = (m * Pt3[0] - Pt3[1] + b) / np.sqrt(1 + m**2)
    B2 = np.vstack((B2, [m, b, d0]))
    ui.ListBox_d_2.addItem(str(np.shape(B2)[0] - 2) + ', dp (px):' + str(np.around(np.abs(d0), decimals=3)))
    reset_points()

    return B2

#############################################
#
# Remove a band from the list
#
############################################


def removeband():
    global B2

    g = ui.ListBox_d_2.currentRow()
    if g > -1:
        B2 = np.delete(B2, (ui.ListBox_d_2.currentRow() + 1), axis=0)
        ui.ListBox_d_2.takeItem(ui.ListBox_d_2.currentRow())

    return B2

#############################################
#
# Add the transmitted beam position
#
############################################


def addcenter():
    global Ct
    Ct = gclick[1]
    ui.centerLine.setText(str(np.around(Ct[0], decimals=1)) + ',' + str(np.around(Ct[1], decimals=1)))
    reset_points()
    return Ct

#############################################
#
# Open an image.
#
############################################


def open_image():
    global width, height, s, gclick, mean_ih, image_diff, B2, press, move, minx, maxx, miny, maxy
    press = False
    move = False
    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    image_diff = QtWidgets.QFileDialog.getOpenFileName(Index, "Open image file", "", "*.png *.jpg *.bmp *.tiff *.tif *.jpeg")
    img = Image.open(str(image_diff[0]))
    ih = img.convert('L').histogram()
    mean_ih = sum(ih[g] * g / sum(ih) for g in range(len(ih)))
    a.imshow(img, origin='upper')
    figure.suptitle(str(image_diff[0]))
    width, height = img.size
    a.axis([0, width, height, 0])
    a.axis('off')
    a.figure.canvas.draw()
    ui.ListBox_d_2.clear()
    s = 1
    gclick = np.zeros((1, 2))
    B2 = np.zeros((1, 3))
    ui.ListBox_theo.clear()
    ui.ListBox_d_2.clear()
    ui.brightness_slider.setValue(100)
    minx = 0
    maxx = width
    miny = height
    maxy = 0

    return s, gclick


##############################################
#
# Rotation matrix for Euler angle and rotation along axis
#
##############################################


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


def Rot(th, a, b, c):
    th = th * np.pi / 180
    aa = a / np.linalg.norm([a, b, c])
    bb = b / np.linalg.norm([a, b, c])
    cc = c / np.linalg.norm([a, b, c])
    c1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    c2 = np.array([[aa**2, aa * bb, aa * cc], [bb * aa, bb**2, bb * cc], [cc * aa,
                                                                          cc * bb, cc**2]], float)
    c3 = np.array([[0, -cc, bb], [cc, 0, -aa], [-bb, aa, 0]], float)
    R = np.cos(th) * c1 + (1 - np.cos(th)) * c2 + np.sin(th) * c3

    return R

##############################################
#
# Determine  the orientation: get the beam direction from two intersecting bands. A third band is necessary to discriminate the correct indices of the bands
# Inputs are stored in the B2=[m,p,d,h,k,l] matrix from the bands
#
##############################################

##############################################
# Get the third coordinate a g vector from the camera length L
##############################################


def three_coor(vect, L):

    c = np.linalg.norm(vect)**2 / L
    vector = [vect[0], vect[1], -c]
    return vector

##############################################################
#
# Test functions for 2 to 6 bands. Search for conditions where the theoretical angles stored in Tab of length P0 are in agreement with the measured ones stored in tab. The min misorientation is 0.5 degrees. It can be increased up to 3 degrees by 0.5 degree step until match. Return a list of  g vectors
#
#################################################################


def testangle2(tab):
    global P0, Tab, eps
    liste_possibles = []
    while liste_possibles == []:
        for ii in P0:

            T1 = np.where(np.abs(Tab[ii, :] - tab[0, 1]) <= eps)

            for jj in T1[0]:
                T = [list(list2[ii].astype(np.int)),
                     list(list2[jj].astype(np.int))]
                liste_possibles.append(T)

        if eps < 3:
            eps += 0.5
            ui.ListBox_theo.addItem('Still running...')
        else:
            ui.ListBox_theo.addItem('No match')
            break

    return liste_possibles


def testangle3(tab):
    global P0, Tab, eps
    liste_possibles = []
    while liste_possibles == []:
        for ii in P0:

            T1 = np.where(np.abs(Tab[ii, :] - tab[0, 1]) <= eps)
            T2 = np.where(np.abs(Tab[ii, :] - tab[0, 2]) <= eps)
            for jj in T1[0]:
                for kk in T2[0]:
                    if np.abs(Tab[kk, jj] - tab[1, 2]) <= eps:
                        T = [list(list2[ii].astype(np.int)),
                             list(list2[jj].astype(np.int)),
                             list(list2[kk].astype(np.int))]
                        liste_possibles.append(T)

        if eps < 3:
            eps += 0.5
            ui.ListBox_theo.addItem('Still running...')
        else:
            ui.ListBox_theo.addItem('No match')
            break

    return liste_possibles


def testangle4(tab):
    global P0, Tab, eps
    liste_possibles = []
    while liste_possibles == []:
        for ii in P0:
            T1 = np.where(np.abs(Tab[ii, :] - tab[0, 1]) <= eps)
            T2 = np.where(np.abs(Tab[ii, :] - tab[0, 2]) <= eps)
            T3 = np.where(np.abs(Tab[ii, :] - tab[0, 3]) <= eps)
            for jj in T1[0]:
                for kk in T2[0]:
                    for pp in T3[0]:
                        if np.abs(Tab[kk, jj] - tab[1, 2]) <= eps and np.abs(Tab[pp, jj] - tab[1, 3]) <= eps \
                                and np.abs(Tab[kk, pp] - tab[2, 3]) <= eps:
                            T = [list(list2[ii].astype(np.int)),
                                 list(list2[jj].astype(np.int)),
                                 list(list2[kk].astype(np.int)),
                                 list(list2[pp].astype(np.int))]
                            liste_possibles.append(T)

        if eps < 3:
            eps += 0.5
            ui.ListBox_theo.addItem('Still running...')
        else:
            ui.ListBox_theo.addItem('No match')
            break

    return liste_possibles


def testangle5(tab):
    global P0, Tab, eps
    liste_possibles = []
    while liste_possibles == []:
        for ii in P0:
            T1 = np.where(np.abs(Tab[ii, :] - tab[0, 1]) <= eps)
            T2 = np.where(np.abs(Tab[ii, :] - tab[0, 2]) <= eps)
            T3 = np.where(np.abs(Tab[ii, :] - tab[0, 3]) <= eps)
            T4 = np.where(np.abs(Tab[ii, :] - tab[0, 4]) <= eps)
            for jj in T1[0]:
                for kk in T2[0]:
                    for pp in T3[0]:
                        for hh in T4[0]:
                            if np.abs(Tab[kk, jj] - tab[1, 2]) <= eps and np.abs(Tab[pp, jj] - tab[1, 3]) <= eps \
                                    and np.abs(Tab[kk, pp] - tab[2, 3]) <= eps and np.abs(Tab[hh, jj] - tab[1, 4]) <= eps \
                                    and np.abs(Tab[hh, kk] - tab[2, 4]) <= eps and np.abs(Tab[hh, pp] - tab[3, 4]) <= eps:
                                T = [list(list2[ii].astype(np.int)),
                                     list(list2[jj].astype(np.int)),
                                     list(list2[kk].astype(np.int)),
                                     list(list2[pp].astype(np.int)),
                                     list(list2[hh].astype(np.int))]
                                liste_possibles.append(T)

        if eps < 3:
            ui.ListBox_theo.addItem('Still running...')
            eps += 0.5
        else:
            ui.ListBox_theo.addItem('No match')
            break

    return liste_possibles


def testangle6(tab):
    global P0, Tab, eps
    liste_possibles = []
    while liste_possibles == []:
        for ii in P0:
            T1 = np.where(np.abs(Tab[ii, :] - tab[0, 1]) <= eps)
            T2 = np.where(np.abs(Tab[ii, :] - tab[0, 2]) <= eps)
            T3 = np.where(np.abs(Tab[ii, :] - tab[0, 3]) <= eps)
            T4 = np.where(np.abs(Tab[ii, :] - tab[0, 4]) <= eps)
            T5 = np.where(np.abs(Tab[ii, :] - tab[0, 5]) <= eps)
            for jj in T1[0]:
                for kk in T2[0]:
                    for pp in T3[0]:
                        for hh in T4[0]:
                            for ll in T5[0]:
                                if np.abs(Tab[kk, jj] - tab[1, 2]) <= eps and np.abs(Tab[pp, jj] - tab[1, 3]) <= eps \
                                        and np.abs(Tab[kk, pp] - tab[2, 3]) <= eps and np.abs(Tab[hh, jj] - tab[1, 4]) <= eps \
                                        and np.abs(Tab[hh, kk] - tab[2, 4]) <= eps and np.abs(Tab[hh, pp] - tab[3, 4]) <= eps \
                                        and np.abs(Tab[ll, jj] - tab[1, 5]) <= eps and np.abs(Tab[kk, ll] - tab[2, 5]) <= eps \
                                        and np.abs(Tab[ll, pp] - tab[3, 5]) <= eps and np.abs(Tab[hh, ll] - tab[4, 5]) <= eps:
                                    T = [list(list2[ii].astype(np.int)),
                                         list(list2[jj].astype(np.int)),
                                         list(list2[kk].astype(np.int)),
                                         list(list2[pp].astype(np.int)),
                                         list(list2[hh].astype(np.int)),
                                         list(list2[ll].astype(np.int))]
                                    liste_possibles.append(T)

        if eps < 3:
            ui.ListBox_theo.addItem('Still running...')
            eps += 0.5
        else:
            ui.ListBox_theo.addItem('No match')
            break

    return liste_possibles

##############################################
#
# Test the uniqueness of a set of solution. Return a vector d with the interplanar distance of the row vector of A
#
##############################################


def Uniqueness(A):
    l = np.shape(A)[0]
    abc = ui.abc_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alp = np.float(alphabetagamma[0])
    bet = np.float(alphabetagamma[1])
    gam = np.float(alphabetagamma[2])
    alp = alp * np.pi / 180
    bet = bet * np.pi / 180
    gam = gam * np.pi / 180
    G = np.array([[a**2, a * b * np.cos(gam), a * c * np.cos(bet)], [a * b * np.cos(gam), b**2, b * c * np.cos(alp)], [a * c * np.cos(bet), b * c * np.cos(alp), c**2]])
    d = np.zeros((1, l))

    for t in range(0, l):
        d[:, t] = 1 / (np.sqrt(np.dot(A[t, :], np.dot(np.linalg.inv(G), A[t, :]))))

    return d


def tilt_axes():
    global s_a, s_b, s_z
    s_a, s_b, s_z = -1, -1, -1
    if ui_Tilt.alpha_signBox.isChecked():
        s_a = 1
    if ui_Tilt.beta_signBox.isChecked():
        s_b = 1
    if ui_Tilt.theta_signBox.isChecked():
        s_z = 1
    return s_a, s_b, s_z


def rot_tilt_angle():
    tilt_axes()
    tilt_a = np.float(ui_Tilt.tilt_a_entry.text())
    tilt_b = np.float(ui_Tilt.tilt_b_entry.text())
    tilt_z = np.float(ui_Tilt.tilt_z_entry.text())
    t_ang = np.float(ui_Tilt.t_ang_entry.text())
    R = np.dot(Rot(s_z * tilt_z, 0, 0, 1), np.dot(Rot(s_b * tilt_b, 1, 0, 0), np.dot(Rot(s_a * tilt_a, 0, 1, 0), Rot(t_ang, 0, 0, 1))))
    return R

##############################################################################
#
#  Show results function
# 1. Get the electron wavelength lambd from acceleration voltage
# 2. Camera length L in Px  L=r(px)*d/lamba, r measured distance in pixel, d the reticular spacing in Angstrom, lambda wavelength, r.d being the calibration constant in the calibration.txt file
# 3. For all the bands in B2, calculate the normal vectors to the bands v0 and their normalized 3D value v and the value of p (in the equation y=mx+p)
# 4. Calculate the angle between the normal vectors and store in Tab
# 5. Test the angles with the testangle function and store the possibilities in the K array and keep only the unique solutions
# 6. Compute orientation from Mackenzie approach
# 7. Display the solutions in the results box

#################################################################################


def show_result():
    global Ct, B2, Dist, P0, eps, Tab, list2, tab, K, v0, pOB, L, v, euler, lambd, calib, euler_tilt
    ui.ListBox_theo.clear()
    if ui_Refine.L_entry.text() == '':
        ui_Refine.L_entry.setText(str(x_calib[ui.Calib_box.currentIndex()][4]))
    if ui_Refine.V_entry.text() == '':
        ui_Refine.V_entry.setText(str(x_calib[ui.Calib_box.currentIndex()][1]))
    calib = np.float(ui_Refine.L_entry.text())
    Vt = np.float(ui_Refine.V_entry.text()) * 1e3

    h = 6.62607004e-34
    m0 = 9.10938356e-31
    e = 1.60217662e-19
    c = 299792458
    lambd = h / np.sqrt(2 * m0 * e * Vt) * 1 / np.sqrt(1 + e * Vt / (2 * m0 * c**2)) * 1e10

    L = calib / lambd
    list2, Tab = listb()
    eps = np.float(ui.tol_entry.text())
    P0 = range(len(list2))
    Num = np.shape(B2)[0]
    v = np.zeros((Num - 1, 3))
    v0 = np.zeros((Num - 1, 2))
    dp = np.zeros((Num - 1, 1))
    for z in range(0, Num - 1):
        v0[z, :] = np.array([(Ct[0] + B2[z + 1, 0] * (height - Ct[1]) - B2[z + 1, 1] * B2[z + 1, 0]) / (1 + B2[z + 1, 0]**2) - Ct[0], height - (B2[z + 1, 0] * (Ct[0] + B2[z + 1, 0] * (height - Ct[1]) - B2[z + 1, 1] * B2[z + 1, 0]) / (1 + B2[z + 1, 0]**2) + B2[z + 1, 1]) - Ct[1]])
        g = (1 + 0.5 * np.sign(v0[z, 1]) * B2[z + 1, 2] / np.linalg.norm(v0[z, :]))
        v0[z, :] = -v0[z, :] * g
        v[z, :] = three_coor(v0[z, :], L)
        dp[z] = calib / 2 / np.linalg.norm(v[z, :] - three_coor(v0[z, :] / g, L))
        v[z, :] = v[z, :] / np.linalg.norm(v[z, :])

    v[:, 1] = -v[:, 1]
    tab = np.zeros((np.shape(v)[0], np.shape(v)[0]))
    for l in range(0, np.shape(v)[0]):
        for f in range(0, np.shape(v)[0]):
            tab[l, f] = np.arccos(np.dot(v[l, :], v[f, :])) * 180 / np.pi

    n = np.shape(tab)[0]
    if eps < 3:
        if n == 2:
            K = testangle2(tab)
        elif n == 3:
            K = testangle3(tab)
        elif n == 4:
            K = testangle4(tab)
        elif n == 5:
            K = testangle5(tab)
        elif n == 6:
            K = testangle6(tab)
        else:
            ui.ListBox_theo.addItem('Number of bands should be more than 1 \n and less than 7')
            return
    else:
        ui.ListBox_theo.addItem('tol. should be below 3 degrees')
        return

    K = np.asarray(K)
    U = np.zeros((np.shape(K)[0], Num - 1))

    for t in range(0, np.shape(K)[0]):
        U[t, :] = Uniqueness(K[t, :, 1:4])

    K = K[np.unique(U, return_index=True, axis=0)[1], :, :]
    euler = np.zeros((K.shape[0], 5))
    for sol in range(0, K.shape[0]):
        g_c = K[sol, :, 1:4]
        g_c = np.dot(Dstar, g_c.T)
        n_c = np.linalg.norm(g_c.T, axis=1)
        g_c = (g_c / n_c).T
        g_sample = v
        if np.abs(np.linalg.det(np.dot(g_c.T, g_sample))) < 1e-7:
            g_cc = np.cross(g_c[0, :], g_c[1, :])
            g_sc = np.cross(g_sample[0, :], g_sample[1, :])
            g_sc = g_sc / np.linalg.norm(g_sc)
            g_cc = g_cc / np.linalg.norm(g_cc)
            g_c = np.vstack((g_c, g_cc))
            g_sample = np.vstack((g_sample, g_sc))
        if np.linalg.det(np.dot(g_c.T, g_sample)) < 0:
            g_c = -g_c

        U, S, V = np.linalg.svd(np.dot(g_c.T, g_sample))
        M = np.dot(V.T, U.T)

        phi = np.arccos(M[2, 2]) * 180 / np.pi
        phi_2 = np.arctan2(M[2, 0], M[2, 1]) * 180 / np.pi
        phi_1 = np.arctan2(M[0, 2], -M[1, 2]) * 180 / np.pi

        t = 0
        d_dev = 0
        for r in range(0, Num - 1):
            ang_dev = np.clip(np.dot(np.dot(M, g_c[r, :]), g_sample[r, :]), -1, 1)
            d_dev = d_dev + n_c[r] * 100 * np.abs(1 / n_c[r] - dp[r])
            t = t + np.abs(np.arccos(ang_dev))
        t = t / g_sample.shape[0] * 180 / np.pi
        d_dev = d_dev / g_sample.shape[0]
        euler[sol, :] = np.array([np.around(phi_1, decimals=3), np.around(phi, decimals=3), np.around(phi_2, decimals=3), np.around(t, decimals=3), np.around(d_dev, decimals=3)])

    so = np.lexsort((euler[:, 4], euler[:, 3]))
    K = K[so, :, :]
    euler = euler[so]
    ui.ListBox_theo.clear()
    for h in range(0, euler.shape[0]):
        M_tilt = np.dot(rot_tilt_angle(), rotation(euler[h, 0], euler[h, 1], euler[h, 2]))
        phi_tilt = np.around(np.arccos(M_tilt[2, 2]) * 180 / np.pi, decimals=3)
        phi_2_tilt = np.around(np.arctan2(M_tilt[2, 0], M_tilt[2, 1]) * 180 / np.pi, decimals=3)
        phi_1_tilt = np.around(np.arctan2(M_tilt[0, 2], -M_tilt[1, 2]) * 180 / np.pi, decimals=3)
        ss = 'g:'
        for hh in range(0, Num - 1):
            ss = ss + ' ' + str(K[h, hh, 1:4])
        ui.ListBox_theo.addItem(ss)
        ui.ListBox_theo.addItem(str(phi_1_tilt) + ',' + str(phi_tilt) + ',' + str(phi_2_tilt))
        ui.ListBox_theo.addItem(str(euler[h, 3]) + ',' + str(euler[h, 4]))

    return K, euler

#######################################
#
# Display results . Compute the expected bands from the orientation matrix. Refine the results by rotation along x,y,z axes, or by adjusting rd and V
#
#######################################


def plot_bands():
    global B2, K, v0, Ct, pOB, width, height, image_diff, L, Dstar

    Num = np.shape(B2)[0]
    brightness()
    a = figure.add_subplot(111)

    for z in range(0, Num - 1):

        v0 = np.array([(Ct[0] + B2[z + 1, 0] * (height - Ct[1]) - B2[z + 1, 1] * B2[z + 1, 0]) / (1 + B2[z + 1, 0]**2) - Ct[0], height - (B2[z + 1, 0] * (Ct[0] + B2[z + 1, 0] * (height - Ct[1]) - B2[z + 1, 1] * B2[z + 1, 0]) / (1 + B2[z + 1, 0]**2) + B2[z + 1, 1]) - Ct[1]])
        ds = np.sqrt(B2[z + 1, 0]**2 + 1) * B2[z + 1, 2]
        v0 = v0 * (1 + 0.5 * np.sign(v0[1]) * B2[z + 1, 2] / np.linalg.norm(v0))
        pOB = B2[z + 1, 1] - 0.5 * ds

        a.plot([-2 * Ct[0], 2 * Ct[0]], [height - (pOB + B2[z + 1, 0] * (-2) * Ct[0]), height - (pOB + B2[z + 1, 0] * 2 * Ct[0])], 'r-')
        a.plot([-2 * Ct[0], 2 * Ct[0]], [height - (pOB + ds / 2 + B2[z + 1, 0] * (-2) * Ct[0]), height - (pOB + ds / 2 + B2[z + 1, 0] * 2 * Ct[0])], 'b-')
        a.plot([-2 * Ct[0], 2 * Ct[0]], [height - (pOB - ds / 2 + B2[z + 1, 0] * (-2) * Ct[0]), height - (pOB - ds / 2 + B2[z + 1, 0] * 2 * Ct[0])], 'b-')

        a.plot([Ct[0], Ct[0] + v0[0]], [Ct[1], Ct[1] + v0[1]], 'g-')
        a.annotate(str(z), (Ct[0] + v0[0], Ct[1] + v0[1]))

    a.axis([0, width, height, 0])
    a.axis('off')
    a.figure.canvas.draw()


def plot_orientation_init():
    global M, sol, euler, angx, angy, angz, calib, lambd
    line = ui.ListBox_theo.currentRow()
    sol = int(line / 3)
    M = rotation(euler[sol, 0], euler[sol, 1], euler[sol, 2])
    angx, angy, angz = 0, 0, 0
    ui_Refine.tx_label.setText(str(np.around(angx, decimals=2)))
    ui_Refine.ty_label.setText(str(np.around(angy, decimals=2)))
    ui_Refine.tz_label.setText(str(np.around(angz, decimals=2)))
    M_tilt = np.dot(rot_tilt_angle(), M)
    phi_tilt = np.around(np.arccos(M_tilt[2, 2]) * 180 / np.pi, decimals=3)
    phi_2_tilt = np.around(np.arctan2(M_tilt[2, 0], M_tilt[2, 1]) * 180 / np.pi, decimals=3)
    phi_1_tilt = np.around(np.arctan2(M_tilt[0, 2], -M_tilt[1, 2]) * 180 / np.pi, decimals=3)
    ui_Refine.euler_label.setText(str(phi_1_tilt) + ',' + str(phi_tilt) + ',' + str(phi_2_tilt))
    plot_orientation()


def plot_orientation():
    global B2, K, v0, Ct, pOB, width, height, image_diff, L, Dstar, sol, M, calib, lambd

    brightness()
    a = figure.add_subplot(111)
    g_c = K[sol, :, 1:4]
    g_c = np.dot(Dstar, g_c.T)
    n_c = np.linalg.norm(g_c.T, axis=1)
    g_c = (g_c / n_c).T
    L = calib / lambd
    for z in range(0, g_c.shape[0]):
        g_xyz = np.dot(M, g_c[z, :])
        vm = L * np.array([0, 0, 1]) - L * g_xyz[2] * g_xyz
        br = np.arcsin(lambd / 2 / distance(K[sol, z, 1], K[sol, z, 2], K[sol, z, 3]))
        vp1 = vm + np.linalg.norm(vm) * np.tan(br) * g_xyz
        vp2 = vm - np.linalg.norm(vm) * np.tan(br) * g_xyz

        a.plot([Ct[0], Ct[0] - vp1[0]], [Ct[1], Ct[1] + vp1[1]], 'g-')
        a.annotate(str(z), (Ct[0] - vp1[0], Ct[1] + vp1[1]))
        a.plot([-vp1[0] + Ct[0] - 1000 * vp1[1], -vp1[0] + Ct[0] + 1000 * vp1[1]], [vp1[1] + Ct[1] - 1000 * vp1[0], vp1[1] + Ct[1] + 1000 * vp1[0]], 'r-')
        a.plot([-vp2[0] + Ct[0] - 1000 * vp2[1], -vp2[0] + Ct[0] + 1000 * vp2[1]], [vp2[1] + Ct[1] - 1000 * vp2[0], vp2[1] + Ct[1] + 1000 * vp2[0]], 'r-')
        a.axis([0, width, height, 0])
        a.axis('off')
        a.figure.canvas.draw()


def Rxm():
    global M, angx
    tx = -np.float(ui_Refine.Rx_entry.text())
    M = np.dot(Rot(tx, 1, 0, 0), M)
    angx = angx + tx
    update_orientation()
    return


def Rxp():
    global M, angx
    tx = np.float(ui_Refine.Rx_entry.text())
    M = np.dot(Rot(tx, 1, 0, 0), M)
    angx = angx + tx
    update_orientation()
    return


def Rym():
    global angy, M
    ty = -np.float(ui_Refine.Ry_entry.text())
    M = np.dot(Rot(ty, 0, 1, 0), M)
    angy = angy + ty
    update_orientation()
    return


def Ryp():
    global angy, M
    ty = np.float(ui_Refine.Ry_entry.text())
    M = np.dot(Rot(ty, 0, 1, 0), M)
    angy = angy + ty
    update_orientation()
    return


def Rzm():
    global angz, M
    tz = np.float(ui_Refine.Rz_entry.text())
    M = np.dot(Rot(tz, 0, 0, 1), M)
    angz = angz + tz
    update_orientation()
    return


def Rzp():
    global angz, M
    tz = -np.float(ui_Refine.Rz_entry.text())
    M = np.dot(Rot(tz, 0, 0, 1), M)
    angz = angz + tz
    update_orientation()
    return


def update_L():
    global calib, M, angx, angy, angz, Vt
    if ui_Refine.L_entry.text() == '':
        ui_Refine.L_entry.setText(str(x_calib[ui.Calib_box.currentIndex()][4]))
    if ui_Refine.V_entry.text() == '':
        ui_Refine.V_entry.setText(str(x_calib[ui.Calib_box.currentIndex()][1]))
    calib = np.float(ui_Refine.L_entry.text())
    Vt = np.float(ui_Refine.V_entry.text()) * 1e3
    show_result()
    plot_orientation_init()


def update_orientation():
    global calib, Vt, M, angx, angy, angz

    calib = np.float(ui_Refine.L_entry.text())
    Vt = np.float(ui_Refine.V_entry.text())
    ui_Refine.tx_label.setText(str(np.around(angx, decimals=2)))
    ui_Refine.ty_label.setText(str(np.around(angy, decimals=2)))
    ui_Refine.tz_label.setText(str(np.around(angz, decimals=2)))
    M_tilt = np.dot(rot_tilt_angle(), M)
    phi = np.arccos(M_tilt[2, 2]) * 180 / np.pi
    phi_2 = np.arctan2(M_tilt[2, 0], M_tilt[2, 1]) * 180 / np.pi
    phi_1 = np.arctan2(M_tilt[0, 2], -M_tilt[1, 2]) * 180 / np.pi
    ui_Refine.euler_label.setText(str(np.around(phi_1, decimals=3)) + ',' + str(np.around(phi, decimals=3)) + ',' + str(np.around(phi_2, decimals=3)))
    plot_orientation()
    return

###################################################
#
# Get extinction from the space group
#
#####################################################


def extinction(space_group, h, k, l):
    global x_space, G
    F = 0
    s = 0
    q = 2 * np.pi * 1e-10 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l])))))

    for i in range(0, len(x_space)):
        if space_group == x_space[i][0]:
            s = i

    while (s < (len(x_space) - 1) and (len(x_space[s + 1]) == 4)):

        f = str(x_space[s + 1][0])
        for z in range(0, len(x_scatt)):
            if f == x_scatt[z][0]:
                f = eval(x_scatt[z][1]) * np.exp(-eval(x_scatt[z][2]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][3]) * np.exp(-eval(x_scatt[z][4]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][5]) * np.exp(-eval(x_scatt[z][6]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][7]) * np.exp(-eval(x_scatt[z][8]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][9])
        F = F + f * np.exp(2j * np.pi * (eval(x_space[s + 1][1]) * h + eval(x_space[s + 1][2]) * k + eval(x_space[s + 1][3]) * l))
        s = s + 1
    I = np.around(float(np.real(F * np.conj(F))), decimals=2)
    return I


######################################################################################
#
# Change the brightness of the image
# Use the mean intensity mean_ih of the gray scaled original image to scale the slider
#
#####################################################################################


def brightness():

    img = Image.open(str(image_diff[0]))
    ui.brightness_slider.setMinimum(10)
    ui.brightness_slider.setMaximum(25500 / mean_ih)

    bv = ui.brightness_slider.value()
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(bv / 100)

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    figure.suptitle(str(image_diff[0]))
    a.imshow(img, origin='upper')
    a.axis([0, width, height, 0])
    a.axis('off')
    a.figure.canvas.draw()


#####################################################################################
#
# Import crystal structure from the structures.txt file: a,b,c,alpha,beta,gamma,space group
#
#####################################################################################

def structure(item):
    global x0, var_hexa, d_label_var, e_entry

    ui.abc_entry.setText(str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]))
    ui.alphabetagamma_entry.setText(str(item[4]) + ',' + str(item[5]) + ',' + str(item[6]))

    ii = ui.space_group_Box.findText(str(item[7]))
    ui.space_group_Box.setCurrentIndex(ii)

##################################################
#
# Add matplotlib toolbar to zoom and pan
#
###################################################


class NavigationToolbar(NavigationToolbar):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom')]

    def set_message(self, msg):
        pass


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

###############################################################
#
# Main, import Gui from KikuchiUI and refineUI
#
################################################################


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
    QtWidgets.qApp.setApplicationName("Kikuchi")
    Index = QtWidgets.QMainWindow()
    ui = kikuchiUI.Ui_Kikuchi()
    ui.setupUi(Index)
    figure = plt.figure(facecolor='white', figsize=[2, 2], dpi=100)
    canvas = FigureCanvas(figure)
    ui.mplvl.addWidget(canvas)
    toolbar = NavigationToolbar(canvas, canvas)
    toolbar.setMinimumWidth(601)

    # read the txt files

    f_space = open(os.path.join(os.path.dirname(__file__), 'space_group.txt'), "r")

    x_space = []

    for line in f_space:
        x_space.append(list(map(str, line.split())))

    list_space_group = []
    for i in range(0, len(x_space)):
        if len(x_space[i]) == 1:
            list_space_group.append(x_space[i][0])

    f_space.close()

    file_struct = open(os.path.join(os.path.dirname(__file__), 'structure.txt'), "r")

    x0 = []

    for line in file_struct:
        x0.append(list(map(str, line.split())))

    i = 0
    file_struct.close()

    for item in x0:
        entry = ui.menuStructure.addAction(item[0])
        entry.triggered.connect(lambda checked, item=item: structure(item))
        ui.space_group_Box.addItem(x0[i][7])
        i = i + 1

################################################
#
# Import calibrations from the txt file: microscope name..., E(kV), r.d constant in a calibrated diffraction pattern (d should be in A, r in pixel)
#
##############################################

    f_calib = open(os.path.join(os.path.dirname(__file__), 'calibrations.txt'), "r")

    x_calib = []

    for line in f_calib:
        x_calib.append(list(map(str, line.split())))

    f_calib.close()
    counter = len(x_calib)

    for i in range(counter):
        ui.Calib_box.addItem(x_calib[i][0] + ' ' + x_calib[i][1] + 'keV ' + x_calib[i][2] + 'cm' + ', Binning:' + x_calib[i][3])

    f_scatt = open(os.path.join(os.path.dirname(__file__), 'scattering.txt'), "r")

    x_scatt = []

    for line in f_scatt:
        x_scatt.append(list(map(str, line.split())))

    f_scatt.close()

    ui.actionSave_figure.triggered.connect(open_image)
    figure.canvas.mpl_connect('button_press_event', onpress)
    figure.canvas.mpl_connect('button_release_event', onrelease)
    figure.canvas.mpl_connect('motion_notify_event', onmove)
    press = False
    move = False

    Tilt = QtWidgets.QDialog()
    ui_Tilt = tiltUI.Ui_Tilt()
    ui_Tilt.setupUi(Tilt)
    ui.actionTilt.triggered.connect(Tilt.show)
    ui_Tilt.tilt_a_entry.setText('0')
    ui_Tilt.tilt_b_entry.setText('0')
    ui_Tilt.tilt_z_entry.setText('0')
    ui_Tilt.t_ang_entry.setText('0')

    Refine = QtWidgets.QDialog()
    ui_Refine = refineUI.Ui_Refine()
    ui_Refine.setupUi(Refine)
    ui.actionRefine_orientation.triggered.connect(Refine.show)
    ui_Refine.Rxm_button.clicked.connect(Rxm)
    ui_Refine.Rxp_button.clicked.connect(Rxp)
    ui_Refine.Rym_button.clicked.connect(Rym)
    ui_Refine.Ryp_button.clicked.connect(Ryp)
    ui_Refine.Rzm_button.clicked.connect(Rzm)
    ui_Refine.Rzp_button.clicked.connect(Rzp)
    ui_Refine.update_button.clicked.connect(update_L)
    ui_Refine.Rx_entry.setText('0.1')
    ui_Refine.Ry_entry.setText('0.1')
    ui_Refine.Rz_entry.setText('0.1')

    ui.Button_reset.clicked.connect(reset_points)
    ui.reset_view_button.clicked.connect(reset)
    ui.Button_addband.clicked.connect(addband)
    ui.Button_remove_band.clicked.connect(removeband)
    ui.Button_add_center.clicked.connect(addcenter)
    ui.brightness_slider.setMinimum(1)
    ui.brightness_slider.setMaximum(50)
    ui.brightness_slider.valueChanged.connect(brightness)
    ui.plot_bands_button.clicked.connect(plot_bands)
    ui.Button_orientation.clicked.connect(plot_orientation_init)
    ui.Button_run.clicked.connect(show_result)
    ui.indice_entry.setText('3')
    ui.tol_entry.setText('1')
    s = 1
    gclick = np.zeros((1, 2))
    B2 = np.zeros((1, 3))

    Index.show()
    sys.exit(app.exec_())
