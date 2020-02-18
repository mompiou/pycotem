#############################################################################################################################################################
#
# KikuPy: determine the crytal orientation given a set of three Kikuchi bands. From F. Mompiou (CEMES-CNRS) with the help of Juan Du (Tsinghua University) and Guillaume Perret (Univ. de Toulouse)
# Requirements
# python 2.7 with matplotlib, numpy, pyqt4
# Procedure:
# 1. Open an image.
# 2. Change brightness to see the Kikuchi bands.
# 3. Select a calibration of the pattern from the list (can be changed in the calibrations.txt file
# 4. Select crystal parameters either from the fields or from an import structure from the menu list. Choose "max indices" of the possible Kikuchi bands (3 by default)
# 5. Determine the g vectors associated to the bands. Click on three consecutive Pts -> Pts1 & 2 are on the same line Pt3 is in the parallel line
# 6. Add bands
# 7. Repeat the procedure to get three bands
# 8. Add the center of the pattern (transmitted beam)
# 9. Click Run to get possibles solutions
# 10. Pick one from the results list and click show results to draw overlays and determine the corresponding Euler angles
#
# Tested with different files. Matches may not be found at first. Repeat the determination of the bands by making a "reset points". Try to pick the finest bands. In some cases the indices of the bands may be exact after  a circular permutation.
#############################################################################################################################################################


from __future__ import division
import numpy as np
from PyQt4 import QtGui, QtCore
import sys
import os
from PIL import Image
from PIL import ImageEnhance
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
import kikuchiUI

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
    r = str(np.around(x, decimals=2)) + ',' + str(np.around(y, decimals=2))
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
    img = Image.open(str(image_diff))
    img = np.array(img)
    figure.suptitle(str(image_diff))
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
    # ui.ListBox_d_2.clear()
    brightness()
    # ui.brightness_slider.setValue(100)
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
    d = eval(x_calib[ui.Calib_box.currentIndex()][2]) / d0
    B2 = np.vstack((B2, [m, b, d0]))
    ui.ListBox_d_2.addItem(str(np.shape(B2)[0] - 2) + ', d:' + str(np.around(np.abs(d), decimals=3)))
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
    image_diff = QtGui.QFileDialog.getOpenFileName(Index, "Open image file", "", "*.png *.jpg *.bmp *.tiff *.tif *.jpeg")
    img = Image.open(str(image_diff))
    ih = img.convert('L').histogram()
    mean_ih = sum(ih[g] * g / sum(ih) for g in range(len(ih)))
    a.imshow(img, origin='upper')
    figure.suptitle(str(image_diff))
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


#####################################################
#
# Rotation matrix for Euler angle and rotation along axis
#
#####################################################


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

###########################################################################################################################################################
#
# Determine  the orientation: get the beam direction from two intersecting bands. A third band is necessary to discriminate the correct indices of the bands
# Inputs are stored in the B2=[m,p,d,h,k,l] matrix from the bands
#
##########################################################################################################################################################

####################################################
# Get the third coordinate a g vector from the camera length L
#####################################################


def three_coor(vect, L):

    c = np.linalg.norm(vect)**2 / L
    vector = [vect[0], vect[1], -c]
    return vector

##############################################################
#
# Test functions for 3 to 6 bands. Search for conditions where the theoretical angles stored in Tab of length P0 are in agreement with the measured ones stored in tab. The min misorientation is 0.5 degrees. It can be increased up to 3 degrees by 0.5 degree step until match. Return a list of  g vectors
#
#################################################################


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
        else:
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
        else:
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
            eps += 0.5
        else:
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
            eps += 0.5
        else:
            break

    return liste_possibles

##################################################################
#
# Test the uniqueness of a set of solution. Return a vector d with the interplanar distance of the row vector of A
#
######################################################################


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

##############################################################################
#
#  Show results function
# 1. Get the electron wavelength lambd from acceleration voltage
# 2. Camera length L in Px  L=r(px)*d/lamba, r measured distance in pixel, d the reticular spacing in Angstrom, lambda wavelength, r.d being the calibration constant in the calibration.txt file
# 3. For all the bands in B2, calculate the normal vectors to the bands v0 and their normalized 3D value v and the value of p (in the equation y=mx+p)
# 4. Calculate the angle between the normal vectors and store in Tab
# 5. Test the angles with the testangle function and store the possibilities in the K array and keep only the unique solutions
# 6. Display the solutions in the results box

#################################################################################


def show_result():
    global Ct, B2, Dist, P0, eps, Tab, list2, tab, K, v0, pOB, L, v, vg
    # pdb.set_trace() #breakpoint for debugging
    h = 6.62607004e-34
    m0 = 9.10938356e-31
    e = 1.60217662e-19
    c = 299792458
    V = eval(x_calib[ui.Calib_box.currentIndex()][1]) * 1e3
    lambd = h / np.sqrt(2 * m0 * e * V) * 1 / np.sqrt(1 + e * V / (2 * m0 * c**2))

    L = eval(x_calib[ui.Calib_box.currentIndex()][2]) * 1e-10 / (lambd)
    list2, Tab = listb()
    eps = 0.5
    P0 = range(len(list2))
    Num = np.shape(B2)[0]
    v = np.zeros((Num - 1, 3))
    v0 = np.zeros((Num - 1, 2))
    pOB = np.zeros((Num - 1, 1))
    for z in range(0, Num - 1):
        v0[z, :] = np.array([(Ct[0] + B2[z + 1, 0] * (height - Ct[1]) - B2[z + 1, 1] * B2[z + 1, 0]) / (1 + B2[z + 1, 0]**2) - Ct[0], height - (B2[z + 1, 0] * (Ct[0] + B2[z + 1, 0] * (height - Ct[1]) - B2[z + 1, 1] * B2[z + 1, 0]) / (1 + B2[z + 1, 0]**2) + B2[z + 1, 1]) - Ct[1]])
        pOB[z] = B2[z + 1, 1] - 0.5 * B2[z + 1, 2] * np.sqrt(1 + B2[z + 1, 0]**2)
        v[z, :] = three_coor(v0[z, :], L)
        v[z, :] = v[z, :] / np.linalg.norm(v[z, :])

    tab = np.zeros((np.shape(v)[0], np.shape(v)[0]))
    for l in range(0, np.shape(v)[0]):
        for f in range(0, np.shape(v)[0]):
            tab[l, f] = np.arccos(np.dot(v[l, :], v[f, :])) * 180 / np.pi

    n = np.shape(tab)[0]
    if n == 3:
        K = testangle3(tab)
    elif n == 4:
        K = testangle4(tab)
    elif n == 5:
        K = testangle5(tab)
    elif n == 6:
        K = testangle6(tab)
    else:
        print('Numbers of bands should be between 3 and 6')

    K = np.asarray(K)
    U = np.zeros((np.shape(K)[0], Num - 1))

    for t in range(0, np.shape(K)[0]):
        U[t, :] = Uniqueness(K[t, :, 1:4])

    K = K[np.unique(U, return_index=True, axis=0)[1], :, :]

    for g in range(0, np.shape(K)[0]):
        ui.ListBox_theo.addItem(str(K[g, :, 1:4]))

    return K

# "
#
# Display results and get the orientation
#
# 1. Pick the solution in the box, calculate the theoretical width of the bands
# 2.  Plot the bands, center line and edges.
# 3. Measure the distance between the transmitted beam Ct with the first two bands OB1 and OB2 and the intersection of the two bands OA
# 4. Get the  distance of the center (xa,ya) to the line of equation y=mx+p : |m*xa-ya+p|/sqrt(1+m**2)
# 5. Get the angles made by the Kikuchi bands with the transmitted beam phi1,phi2,phi3
# 6. Pick the 3 first bands g1,g2,g3 and get g4=g1xg2
# 7. Form the matrix M composed with the row vector of g1,g2,g4 with the appropriate sign (there are 8 possible positions of the central beam with respect to the crossing A
# 8. Get the experimental angle gamma1 formed by the first band with respect to the Ox axis
# 9. Compute the q vector (normalized beam direction) from the equation q/|q|=M^-1(|g1|cos(phi1),|g2|cos(phi2),|g4|cos(phi3))
# 10. Determine Euler angles (phi_1, phi, phi_2) from q and the position of g1 and gamma1 for all the possibilities
# 11.  Determine the position of g1,g2 and g3 and compare them to the corresponding 3 v0. If the three vectors are ok then the sum of the norm of the difference vector g-v0 should be minimum (stored in E). Then the relative position of the bands with respect to the center is correct.
# 12.  Display the correct Euler angles in the orientation box.
#####################################################################


def orientation():
    global B2, K, v0, Ct, pOB, width, height, image_diff, L, v, vg, sol, Res, Dstar

    Num = np.shape(B2)[0]
    ds = []
    sol = ui.ListBox_theo.currentRow()

    for i in range(0, Num - 1):

        ds = np.append(ds, np.sqrt(B2[i + 1, 0]**2 + 1) * eval(x_calib[ui.Calib_box.currentIndex()][2]) / distance(K[sol, i, 1], K[sol, i, 2], K[sol, i, 3]))

    brightness()

    a = figure.add_subplot(111)

    for z in range(0, np.shape(v0)[0]):
        a.plot([-2 * Ct[0], 2 * Ct[0]], [height - (pOB[z] + B2[z + 1, 0] * (-2) * Ct[0]), height - (pOB[z] + B2[z + 1, 0] * 2 * Ct[0])], 'r-')
        a.plot([-2 * Ct[0], 2 * Ct[0]], [height - (pOB[z] + ds[z] / 2 + B2[z + 1, 0] * (-2) * Ct[0]), height - (pOB[z] + ds[z] / 2 + B2[z + 1, 0] * 2 * Ct[0])], 'b-')
        a.plot([-2 * Ct[0], 2 * Ct[0]], [height - (pOB[z] - ds[z] / 2 + B2[z + 1, 0] * (-2) * Ct[0]), height - (pOB[z] - ds[z] / 2 + B2[z + 1, 0] * 2 * Ct[0])], 'b-')

        a.plot([Ct[0], Ct[0] + v0[z, 0]], [Ct[1], Ct[1] + v0[z, 1]], 'g-')
        a.annotate(str(z), (Ct[0] + v0[z, 0], Ct[1] + v0[z, 1]))

    a.axis([0, width, height, 0])
    a.axis('off')
    a.figure.canvas.draw()

    OB1 = np.abs(B2[1, 0] * Ct[0] - (height - Ct[1]) + pOB[0]) / np.sqrt(1 + B2[1, 0]**2)  # first band
    OB2 = np.abs(B2[2, 0] * Ct[0] - (height - Ct[1]) + pOB[1]) / np.sqrt(1 + B2[2, 0]**2)		# second band

    # A is the intersection of the two bands
    OA = np.sqrt(((pOB[1] - pOB[0]) / (B2[1, 0] - B2[2, 0]) - Ct[0])**2 + (height - (B2[1, 0] * pOB[1] - B2[2, 0] * pOB[0]) / (B2[1, 0] - B2[2, 0]) - Ct[1])**2)

    phi1 = np.arctan(L / OB1)[0]
    phi2 = np.arctan(L / OB2)[0]
    phi3 = np.arctan(OA / L)[0]

    #################################################################

    g1 = K[sol, 0, 1:4]
    g2 = K[sol, 1, 1:4]
    g3 = K[sol, 2, 1:4]

    g1 = np.dot(Dstar, g1)
    g2 = np.dot(Dstar, g2)
    g3 = np.dot(Dstar, g3)
    g4 = np.cross(g1, g2)

    gamma1 = np.arctan2(-v0[0, 1], -v0[0, 0]) * 180 / np.pi
    M = np.zeros((8, 3, 3))
    M[0, :, :] = np.vstack((g1, g2, g4))
    M[1, :, :] = np.vstack((-g1, g2, g4))
    M[2, :, :] = np.vstack((-g1, -g2, g4))
    M[3, :, :] = np.vstack((g1, -g2, g4))
    M[4, :, :] = np.vstack((g1, g2, -g4))
    M[5, :, :] = np.vstack((-g1, g2, -g4))
    M[6, :, :] = np.vstack((-g1, -g2, -g4))
    M[7, :, :] = np.vstack((g1, -g2, -g4))

    E = np.zeros((8, 4))
    q = np.zeros((8, 3))

    for i in range(0, 8):

        q[i, :] = np.dot(np.linalg.inv(M[i, :, :]), np.array([np.linalg.norm(M[i, 0, :]) * np.cos(phi1), np.linalg.norm(M[i, 1, :]) * np.cos(phi2), np.linalg.norm(M[i, 2, :]) * np.cos(phi3)]))
        phi = np.arccos(q[i, 2]) * 180 / np.pi
        phi_2 = np.arctan2(q[i, 0], q[i, 1]) * 180 / np.pi
        gr = np.dot(rotation(0, phi, phi_2), M[i, 0, :])
        theta1 = np.arctan2(gr[1], gr[0]) * 180 / np.pi
        phi_1 = gamma1 - theta1
        g1r = np.dot(rotation(phi_1, phi, phi_2), M[i, 0, :])
        g1r = g1r[0:2] / np.linalg.norm(g1r[0:2])
        g2r = np.dot(rotation(phi_1, phi, phi_2), M[i, 1, :])
        g2r = g2r[0:2] / np.linalg.norm(g2r[0:2])
        g3r = np.dot(rotation(phi_1, phi, phi_2), g3)
        g3r = np.sign(g3r[2]) * g3r[0:2] / np.linalg.norm(g3r[0:2])
        r2 = np.dot(Rot(180, 0, 1, 0), rotation(phi_1, phi, phi_2))  # need an extra rotation because of the inversion of the y axis in the image
        phi = np.arccos(r2[2, 2]) * 180 / np.pi
        phi_2 = np.arctan2(r2[2, 0], r2[2, 1]) * 180 / np.pi
        phi_1 = np.arctan2(r2[0, 2], -r2[1, 2]) * 180 / np.pi

        E[i, :] = np.array([phi_1, phi, phi_2, np.linalg.norm(g2r + v0[1, :] / np.linalg.norm(v0[1, :])) + np.linalg.norm(g3r + v0[2, :] / np.linalg.norm(v0[2, :]))])

    Res = E[np.where(np.amin(E[:, 3]) == E[:, 3])[0], :][0][0:3]

    ui.orientation_result.setText(str(np.around(Res[0], decimals=1)) + ',' + str(np.around(Res[1], decimals=1)) + ',' + str(np.around(Res[2], decimals=1)))


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

    img = Image.open(str(image_diff))
    ui.brightness_slider.setMinimum(10)
    ui.brightness_slider.setMaximum(25500 / mean_ih)

    bv = ui.brightness_slider.value()
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(bv / 100)

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    figure.suptitle(str(image_diff))
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


###############################################################
#
# Main, import Gui from KikuPyUI
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

    app = QtGui.QApplication(sys.argv)
    Index = QtGui.QMainWindow()
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
        x_space.append(map(str, line.split()))

    list_space_group = []
    for i in range(0, len(x_space)):
        if len(x_space[i]) == 1:
            list_space_group.append(x_space[i][0])

    f_space.close()

    file_struct = open(os.path.join(os.path.dirname(__file__), 'structure.txt'), "r")

    x0 = []

    for line in file_struct:
        x0.append(map(str, line.split()))

    i = 0
    file_struct.close()

    for item in x0:
        entry = ui.menuStructure.addAction(item[0])
        Index.connect(entry, QtCore.SIGNAL('triggered()'), lambda item=item: structure(item))
        ui.space_group_Box.addItem(x0[i][7])
        i = i + 1

###########################################################################################################################################
#
# Import calibrations from the txt file: microscope name..., E(kV), r.d constant in a calibrated diffraction pattern (d should be in A^-1, r in pixel)
#
###############################################################################################################################################
    f_calib = open(os.path.join(os.path.dirname(__file__), 'calibrations.txt'), "r")

    x_calib = []

    for line in f_calib:
        x_calib.append(map(str, line.split()))

    f_calib.close()
    counter = len(x_calib)

    for i in range(counter):
        ui.Calib_box.addItem(x_calib[i][0] + ' ' + x_calib[i][1] + 'keV ' + x_calib[i][2])

    f_scatt = open(os.path.join(os.path.dirname(__file__), 'scattering.txt'), "r")

    x_scatt = []

    for line in f_scatt:
        x_scatt.append(map(str, line.split()))

    f_scatt.close()

    Index.connect(ui.actionSave_figure, QtCore.SIGNAL('triggered()'), open_image)
    figure.canvas.mpl_connect('button_press_event', onpress)
    figure.canvas.mpl_connect('button_release_event', onrelease)
    figure.canvas.mpl_connect('motion_notify_event', onmove)
    press = False
    move = False

    ui.Button_reset.clicked.connect(reset_points)
    ui.reset_view_button.clicked.connect(reset)
    ui.Button_addband.clicked.connect(addband)
    ui.Button_remove_band.clicked.connect(removeband)
    ui.Button_remove_center.clicked.connect(remove_center)
    ui.Button_add_center.clicked.connect(addcenter)
    ui.brightness_slider.setMinimum(1)
    ui.brightness_slider.setMaximum(50)
    ui.brightness_slider.valueChanged.connect(brightness)
    ui.Button_orientation.clicked.connect(orientation)
    ui.Button_run.clicked.connect(show_result)
    ui.indice_entry.setText('3')
    s = 1
    gclick = np.zeros((1, 2))
    B2 = np.zeros((1, 3))

    Index.show()
    sys.exit(app.exec_())
