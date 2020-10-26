from __future__ import division
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import os
from itertools import combinations
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import pyplot as plt
from . import diffractionUI

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


def click(event):
    global gclick, s
    x = event.xdata
    y = event.ydata
    s = s + 1
    gclick = np.vstack((gclick, np.array([x, y])))
    plot_click()
    distance()
    return gclick, s


def plot_click():
    global gclick, minx, maxx, miny, maxy
    a = figure.add_subplot(111)
    minx, maxx = a.get_xlim()
    miny, maxy = a.get_ylim()

    for i in range(1, gclick.shape[0]):
        a.annotate(str(i), (gclick[i, 0], gclick[i, 1]))
    a.plot(gclick[1:, 0], gclick[1:, 1], 'b+')
    a.axis('off')
    a.figure.canvas.draw()

#########################
#
# Reset points
#
############################


def reset_points():
    global image_diff, gclick, s, minx, maxx, miny, maxy

    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    img = Image.open(str(image_diff[0]))
    img = np.array(img)
    figure.suptitle(str(image_diff[0]))
    a.imshow(img, origin="upper")
    a.axis([minx, maxx, miny, maxy])
    a.axis('off')
    a.figure.canvas.draw()
    s = 1
    gclick = np.zeros((1, 2))
    ui.ListBox_d_2.clear()
    ui.ListBox_theo.clear()
    return s, gclick


######################
#
# reset view
#
#########################

def reset():
    global width, height, gclick, s, image_diff

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
    plot_click()
    ui.ListBox_d_2.clear()
    ui.ListBox_theo.clear()
    s = 1


#########################
#
# Get experimental interplanar distance
#
############################


def distance():
    global gclick, d, counter, x_calib
    for i in range(1, gclick.shape[0]):
        x = gclick[1, 0]
        y = gclick[1, 1]
        x1 = gclick[i, 0]
        y1 = gclick[i, 1]
        n = np.int(ui.n_entry.text())
        d = np.sqrt((x1 - x)**2 + (y1 - y)**2) / n

        i = 0
        if d == 0:
            bet = 0
        else:
            if x1 - x > 0:
                bet = 180 - np.arccos((y1 - y) / (n * d)) * 180 / np.pi
            else:
                bet = np.arccos((y1 - y) / (n * d)) * 180 / np.pi

        d = eval(x_calib[ui.Calib_box.currentIndex()][4]) / d

    if np.isinf(d) == 0:
        ui.ListBox_d_2.addItem(str(np.around(d, decimals=2)) + ',' + str(np.around(bet, decimals=2)))
    return d

##################
#
# Two beams to three beams
#
###################


def two_to_three():
    h = 6.62607004e-34
    m0 = 9.10938356e-31
    e = 1.60217662e-19
    c = 299792458
    Vt = eval(x_calib[ui.Calib_box.currentIndex()][1]) * 1e3
    lambd = h / np.sqrt(2 * m0 * e * Vt) * 1 / np.sqrt(1 + e * Vt / (2 * m0 * c**2)) * 1e10
    d = np.float(ui.ListBox_d_2.currentItem().text().split(',')[0])
    epsilon = np.around(lambd / 2 / d * 180 / np.pi, decimals=3)

    return epsilon

#########################
#
# Get theoretical distance
#
############################


def angle_check(Dis):
    global G, Dstar, D
    if ui.tilt_a_entry.text() == []:
        return
    else:
        s_a, s_b, s_z = tilt_axes()
        ta = np.float(ui.tilt_a_entry.text())
        tb = np.float(ui.tilt_b_entry.text())
        tz = np.float(ui.tilt_z_entry.text())
        inc = ui.ListBox_d_2.currentItem().text().split(',')

        t_ang = np.float(ui.tilt_axis_angle_entry.text())
        R = np.dot(Rot(s_z * tz, 0, 0, 1), np.dot(Rot(s_b * tb, 1, 0, 0), Rot(s_a * ta, 0, 1, 0)))
        ny = -np.float(inc[1])
        epsilon = 0
        if ui.two_to_three_checkBox.isChecked():
            epsilon = two_to_three() * np.pi / 180
        t = np.dot(Rot(ny, 0, 0, 1), np.array([0, np.cos(epsilon), np.sin(epsilon)]))
        g_d = np.dot(R, np.dot(Rot(t_ang, 0, 0, 1), t))

        if ui.diff_spot_Listbox.count() == 0:
            Dis = np.hstack((Dis, np.zeros((Dis.shape[0], 1))))
            return Dis
        else:
            s1 = [ui.diff_spot_Listbox.item(i).text() for i in range(ui.diff_spot_Listbox.count())]
            tilt_a = []
            tilt_b = []
            tilt_z = []
            inclination = []
            g_hkl = []
            epsilon = []

            for i in range(0, len(s1)):
                l = list(map(float, s1[i].split(',')))
                tilt_a.append(l[0])
                tilt_b.append(l[1])
                tilt_z.append(l[2])
                inclination.append(l[3])
                g_hkl.append(l[4:7])
                epsilon.append(l[8])
            inclination = np.array(inclination)
            tilt_a = np.array(tilt_a)
            tilt_b = np.array(tilt_b)
            tilt_z = np.array(tilt_z)
            g_hkl = np.array(g_hkl)
            epsilon = np.array(epsilon) * np.pi / 180
            t_ang = np.float(ui.tilt_axis_angle_entry.text())
            for i in range(0, np.shape(tilt_a)[0]):
                Dis = np.hstack((Dis, np.zeros((Dis.shape[0], 1))))
                R = np.dot(Rot(s_z * tilt_z[i], 0, 0, 1), np.dot(Rot(s_b * tilt_b[i], 1, 0, 0), Rot(s_a * tilt_a[i], 0, 1, 0)))
                ny = -inclination[i]
                t = np.dot(Rot(ny, 0, 0, 1), np.array([0, np.cos(epsilon[i]), np.sin(epsilon[i])]))
                for k in range(0, Dis.shape[0]):
                    a = np.arccos(np.dot(np.dot(R, np.dot(Rot(t_ang, 0, 0, 1), t)), g_d))
                    Dis[k, -1] = np.around(np.abs(angle(Dis[k, 1:4], g_hkl[i, :]) - a) * 180 / np.pi, decimals=1)
    Dis = Dis[np.argsort(Dis[:, -1])]
    return Dis


def cryst():
    global D, Dstar, G
    abc = ui.abc_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    alpha = np.float(alphabetagamma[0])
    beta = np.float(alphabetagamma[1])
    gamma = np.float(alphabetagamma[2])
    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180
    V = a * b * c * np.sqrt(1 - (np.cos(alpha)**2) - (np.cos(beta))**2 - (np.cos(gamma))**2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
    D = np.array([[a, b * np.cos(gamma), c * np.cos(beta)], [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)], [0, 0, V / (a * b * np.sin(gamma))]])
    G = np.array([[a**2, a * b * np.cos(gamma), a * c * np.cos(beta)], [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)], [a * c * np.cos(beta), b * c * np.cos(alpha), c**2]])
    Dstar = np.transpose(np.linalg.inv(D))
    return


def distance_theo():
    global d, G, Dstar, D
    cryst()
    e = np.int(ui.indice_entry.text())
    Dist = np.zeros(((2 * e + 1)**3 - 1, 5))

    ui.ListBox_theo.clear()
    w = 0
    for i in range(e + 1, -e, -1):
        for j in range(e + 1, -e, -1):
            for k in range(e + 1, -e, -1):
                if (i, j, k) != (0, 0, 0):
                    di = 1 / (np.sqrt(np.dot(np.array([i, j, k]), np.dot(np.linalg.inv(G), np.array([i, j, k])))))

                    if di < (d / (1 - d * np.float(ui.precision_entry.text()) / eval(x_calib[ui.Calib_box.currentIndex()][4]))) and di > (d / (1 + d * np.float(ui.precision_entry.text()) / eval(x_calib[ui.Calib_box.currentIndex()][4]))):
                        I = extinction(ui.SpaceGroup_box.currentText(), i, j, k)
                        Dist[w, :] = np.array([np.around(di, decimals=3), int(i), int(j), int(k), I])
                        w = w + 1
    Dist = angle_check(Dist[:w, :])
    for k in range(0, Dist.shape[0]):
        ui.ListBox_theo.addItem(str(Dist[k, 0]) + '  |  ' + str(int(Dist[k, 1])) + ',' + str(int(Dist[k, 2])) + ',' + str(int(Dist[k, 3])) + '  |  ' + str(Dist[k, 4]) + '  |  ' + ', '.join(map(str, Dist[k, 5:])))
    return


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
                    I = extinction(ui.SpaceGroup_box.currentText(), i, j, k)
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

########################
#
# Add/Remove spot
#
#########################


def add_spot():

    s2 = ui.ListBox_d_2.currentItem().text().split(',')
    s3 = ui.tilt_a_entry.text()
    s4 = ui.tilt_b_entry.text()
    s5 = ui.tilt_z_entry.text()
    if ui.two_to_three_checkBox.isChecked():
        s6 = str(two_to_three())
    else:
        s6 = '0'
    if ui.do_not_guess_checkBox.isChecked():
        s = s3 + ',' + s4 + ',' + s5 + ',' + s2[1] + ',' + '0' + ',' + '0' + ',' + '0' + ',' + s2[0] + ',' + s6
    else:
        s1 = ui.ListBox_theo.currentItem().text().split('|')
        s11 = s1[1].split(',')
        s11 = [x.strip(' ') for x in s11]
        s = s3 + ',' + s4 + ',' + s5 + ',' + s2[1] + ',' + s11[0] + ',' + s11[1] + ',' + s11[2] + ',' + s2[0] + ',' + s6
    ui.diff_spot_Listbox.addItem(s)
    ss = ui.diff_spot_Listbox.count()
    item = ui.diff_spot_Listbox.item(ss - 1)
    item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)


def remove_spot():
    ui.diff_spot_Listbox.takeItem(ui.diff_spot_Listbox.currentRow())

######################################################
#
# Determine orientation from a set of at least 3 diff
#
####################################################


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


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


def dhkl(g1, g2, g3):
    global G
    return 1 / (np.sqrt(np.dot(np.array([g1, g2, g3]), np.dot(np.linalg.inv(G), np.array([g1, g2, g3])))))


def angle(p1, p2):
    global Dstar, D
    p1c = np.dot(Dstar, p1)
    p2c = np.dot(Dstar, p2)
    the = np.clip(np.dot(p1c, p2c) / (np.linalg.norm(p1c) * np.linalg.norm(p2c)), -1, 1)
    the = np.arccos(the)
    return the


def cryststruct():
    global cs
    abc = ui.abc_entry.text().split(",")
    alphabetagamma = ui.alphabetagamma_entry.text().split(",")
    a = np.float(abc[0])
    b = np.float(abc[1])
    c = np.float(abc[2])
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

    if gam != 90 and alp != 90 and bet != 90 and a != b and b != c:
        cs = 7
    return cs


def Sy():
    global Dstar, D
    if cs == 1:
        S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, -1, 1], [-1, 1, 0], [-1, 0, 1]])

    if cs == 2:
        S = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1 / 2, np.sqrt(3) / 2, 0], [-1 / 2, np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, 1 / 2, 0], [np.sqrt(3) / 2, 1 / 2, 0], [-np.sqrt(3) / 2, 1 / 2, 0]])

    if cs == 3:
        S = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0]])

    if cs == 4:
        S = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1 / 2, np.sqrt(3) / 2, 0], [-1 / 2, np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, 1 / 2, 0], [np.sqrt(3) / 2, 1 / 2, 0], [-np.sqrt(3) / 2, 1 / 2, 0]])

    if cs == 5:
        S = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    if cs == 6:
        S = np.array([[0, 1, 0]])

    if cs == 7:
        S = np.array([[0, 0, 0]])

    S = np.dot(np.linalg.inv(Dstar), np.dot(D, S.T)).T

    return S

##################
#
# Determine orientation
#
#################


def check_ambiguity(g2):
    global cs
    S = Sy()
    aa = 1
    for i in range(0, S.shape[0]):
        az = np.arccos(np.around(np.dot(g2, S[i, :]) / np.linalg.norm(g2) / np.linalg.norm(S[i, :]), decimals=8)) * 180 / np.pi
        if np.abs(az - 180) < 1e-3 or np.abs(az) < 1e-3:
            aa = 0
            return aa
    return aa


def euler_determine(g_c, g_sample, d):
    global Dstar, cs
    cryststruct()
    g_c = np.dot(Dstar, g_c.T)
    n_c = np.linalg.norm(g_c.T, axis=1)
    g_c = (g_c / n_c).T
    aa = 0

    if g_c.shape[0] == 2:
        g_cross = np.cross(g_c[0, :], g_c[1, :])
        g_c = np.vstack((g_c, g_cross / np.linalg.norm(g_cross)))
        g_sample_cross = np.cross(g_sample[0, :], g_sample[1, :])
        g_sample = np.vstack((g_sample, g_sample_cross / np.linalg.norm(g_sample_cross)))
        aa = check_ambiguity(g_cross)

    if np.linalg.matrix_rank(g_c) < 3:
        g_cross = np.cross(g_c[0, :], g_c[1, :])
        g_c = np.vstack((g_c, g_cross / np.linalg.norm(g_cross)))
        g_sample_cross = np.cross(g_sample[0, :], g_sample[1, :])
        g_sample = np.vstack((g_sample, g_sample_cross / np.linalg.norm(g_sample_cross)))
        aa = check_ambiguity(g_cross)

    else:
        if g_c.shape[0] == 3:
            c = list(combinations(range(g_c.shape[0]), 2))
            for i in range(len(c)):
                a = list(set(list(range(g_c.shape[0]))) - set(list(c[i])))
                ne = np.linalg.norm(np.dot(g_c[c[i], :], g_c[a, :].T))
                if ne < 1e-8:
                    aa = check_ambiguity(g_c[a, :])
                    if aa == 1:
                        g_sample_cross = g_sample[a, :][0]

        else:
            c = list(combinations(range(g_c.shape[0]), 3))
            for i in range(len(c)):
                de = np.linalg.det(g_c[c[i], :])
                a = list(set(list(range(g_c.shape[0]))) - set(list(c[i])))
                if np.abs(de) < 1e-8:
                    ne = np.linalg.norm(np.dot(g_c[c[i], :], g_c[a, :].T))
                    if ne < 1e-8:
                        aa = check_ambiguity(g_c[a, :])
                        if aa == 1:
                            g_sample_cross = g_sample[a, :][0]

    if np.linalg.det(np.dot(g_c.T, g_sample)) < 0:
        g_c = -g_c

    U, S, V = np.linalg.svd(np.dot(g_c.T, g_sample))
    M = np.dot(V.T, U.T)

    phi = np.arccos(M[2, 2]) * 180 / np.pi
    phi_2 = np.arctan2(M[2, 0], M[2, 1]) * 180 / np.pi
    phi_1 = np.arctan2(M[0, 2], -M[1, 2]) * 180 / np.pi
    t, t2 = 0, 0
    for r in range(0, d.shape[0]):
        ang_dev = np.clip(np.dot(np.dot(M, g_c[r, :]), g_sample[r, :]), -1, 1)
        t = t + np.abs(np.arccos(ang_dev))
        t2 = t2 + n_c[r] * 100 * np.abs(1 / n_c[r] - d[r])
    t = t / g_sample.shape[0] * 180 / np.pi
    t2 = t2 / g_sample.shape[0]

    if aa == 1:
        M = np.dot(Rot(180, g_sample_cross[0], g_sample_cross[1], g_sample_cross[2]), M)
        phip = np.arccos(M[2, 2]) * 180 / np.pi
        phi_2p = np.arctan2(M[2, 0], M[2, 1]) * 180 / np.pi
        phi_1p = np.arctan2(M[0, 2], -M[1, 2]) * 180 / np.pi

    else:
        phip, phi_1p, phi_2p = 0, 0, 0

    R = np.array([phi_1, phi, phi_2, phi_1p, phip, phi_2p, t, t2])
    return R


########################################
#
# Get orientation from the selected spots.
#
#########################################


def tilt_axes():
    global s_a, s_b, s_z
    s_a, s_b, s_z = -1, -1, -1
    if ui.alpha_signBox.isChecked():
        s_a = 1
    if ui.beta_signBox.isChecked():
        s_b = 1
    if ui.theta_signBox.isChecked():
        s_z = 1
    return s_a, s_b, s_z
###########################################


def guess():
    if ui.do_not_guess_checkBox.isChecked():
        ui.label_2.setEnabled(False)
        ui.ListBox_theo.setEnabled(False)
        ui.distance_button.setEnabled(False)
        ui.precision_entry.setEnabled(False)
        ui.precision_label.setEnabled(False)
    else:
        ui.label_2.setEnabled(True)
        ui.ListBox_theo.setEnabled(True)
        ui.distance_button.setEnabled(True)
        ui.precision_entry.setEnabled(True)
        ui.precision_label.setEnabled(True)


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
            ui.euler_listbox.addItem('Still running...')
        else:
            ui.euler_listbox.addItem('No match')
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
            ui.euler_listbox.addItem('Still running...')
        else:
            ui.euler_listbox.addItem('No match')
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
            ui.euler_listbox.addItem('Still running...')
        else:
            ui.euler_listbox.addItem('No match')
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
            ui.euler_listbox.addItem('Still running...')
            eps += 0.5
        else:
            ui.euler_listbox.addItem('No match')
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
            ui.euler_listbox.addItem('Still running...')
            eps += 0.5
        else:
            ui.euler_listbox.addItem('No match')
            break

    return liste_possibles


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


def do_not_guess(g_s):
    global eps, P0, Tab, list2

    list2, Tab = listb()
    eps = 1
    P0 = range(len(list2))
    N = g_s.shape[0]
    tab = np.zeros((N, N))
    for l in range(0, N):
        for f in range(0, N):
            tab[l, f] = np.arccos(np.dot(g_s[l, :], g_s[f, :])) * 180 / np.pi

    n = np.shape(tab)[0]
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
        ui.ListBox_theo.addItem('Number of bands should be less than 7')
        return

    K = np.asarray(K)
    U = np.zeros((np.shape(K)[0], N))

    for t in range(0, np.shape(K)[0]):
        U[t, :] = Uniqueness(K[t, :, 1:4])

    K = K[np.unique(U, return_index=True, axis=0)[1], :, :]

    return K


def get_data():

    s_a, s_b, s_z = tilt_axes()
    ui.euler_listbox.clear()
    s = [str(x.text()) for x in ui.diff_spot_Listbox.selectedItems()]
    tilt_a = []
    tilt_b = []
    tilt_z = []
    inclination = []
    g_c = []
    d_g = []
    epsilon = []

    for i in range(0, len(s)):
        l = list(map(float, s[i].split(',')))
        tilt_a.append(l[0])
        tilt_b.append(l[1])
        tilt_z.append(l[2])
        inclination.append(l[3])
        d_g.append(l[7])
        g_c.append(l[4:7])
        epsilon.append(l[8])

    inclination = np.array(inclination)
    d_g = np.array(d_g)
    tilt_a = np.array(tilt_a)
    tilt_b = np.array(tilt_b)
    tilt_z = np.array(tilt_z)
    g_c = np.array(g_c)
    epsilon = np.array(epsilon) * np.pi / 180

    t_ang = np.float(ui.tilt_axis_angle_entry.text())
    g_sample = np.zeros((np.shape(tilt_a)[0], 3))

    for i in range(0, np.shape(tilt_a)[0]):
        R = np.dot(Rot(s_z * tilt_z[i], 0, 0, 1), np.dot(Rot(s_b * tilt_b[i], 1, 0, 0), Rot(s_a * tilt_a[i], 0, 1, 0)))
        ny = -inclination[i]
        t = np.dot(Rot(ny, 0, 0, 1), np.array([0, np.cos(epsilon[i]), np.sin(epsilon[i])]))
        g_sample[i, :] = np.dot(R, np.dot(Rot(t_ang, 0, 0, 1), t))

    if ui.do_not_guess_checkBox.isChecked():
        g_c = do_not_guess(g_sample)

    return g_c, g_sample, d_g


def display_result(A):
    if (A[3:6] == 0).all():
        ui.euler_listbox.addItem('Phi1,Phi,Phi2')
        ui.euler_listbox.addItem(','.join(map("{:.3f}".format, A[0:3])))
    else:
        ui.euler_listbox.addItem('Phi1,Phi,Phi2 - ambiguous result')
        ui.euler_listbox.addItem(','.join(map("{:.3f}".format, A[0:3])))
        ui.euler_listbox.addItem(','.join(map("{:.3f}".format, A[3:6])))
    ui.euler_listbox.addItem('Mean angular deviation, Mean d-spacing dev.')
    ui.euler_listbox.addItem(','.join(map("{:.3f}".format, A[6:8])))


def get_orientation():
    global G, Dstar
    g_c, g_sample, d_g = get_data()
    if ui.do_not_guess_checkBox.isChecked():
        euler = np.zeros((g_c.shape[0], 8))
        for sol in range(0, g_c.shape[0]):
            euler[sol, :] = euler_determine(g_c[sol, :, 1:4], g_sample, d_g)
        so = np.lexsort((np.around(euler[:, 7], decimals=3), np.around(euler[:, 6], decimals=3)))
        g0 = g_c[so, :, :]
        euler = euler[so]
        ui.euler_listbox.clear()
        for h in range(0, euler.shape[0]):
            ss = 'g:'
            for hh in range(0, g_sample.shape[0]):
                ss = ss + ' ' + str(g0[h, hh, 1:4])
            ui.euler_listbox.addItem(ss)

            display_result(euler[h, :])

    else:
        R = euler_determine(g_c, g_sample, d_g)
        display_result(R)

#########################
#
# Open image
#
############################


def open_image():
    global width, height, image_diff, s, gclick, minx, maxx, miny, maxy, press, move
    press = False
    move = False
    a = figure.add_subplot(111)
    a.figure.clear()
    a = figure.add_subplot(111)
    image_diff = QtWidgets.QFileDialog.getOpenFileName(Index, "Open image file", "", "*.png *.jpg *.bmp *.tiff *.tif *.jpeg")
    img = Image.open(str(image_diff[0]))
    img = np.array(img)
    a.imshow(img, origin='upper')
    figure.suptitle(str(image_diff[0]))
    height, width = img.shape[0], img.shape[1]
    a.axis([0, width, height, 0])
    a.axis('off')
    a.figure.canvas.draw()
    ui.ListBox_d_2.clear()
    s = 1
    gclick = np.zeros((1, 2))
    ui.ListBox_d_2.clear()
    minx = 0
    maxx = width
    miny = height
    maxy = 0
    return s, gclick


#####################################################
#
# Get diffraction spectrum
#
###################################################

def pair(number):
    if number % 2 == 0:
        return 1


def impair(number):
    if number % 2 == 0:
        return 0


class Spect(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Spect, self).__init__(parent)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.toolbar = NavigationToolbar(self.canvas, self)

        gridLayout = QtWidgets.QGridLayout()
        self.lineEdit = QtWidgets.QLineEdit()
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        gridLayout.addWidget(self.lineEdit, 1, 2, 1, 1)
        self.label = QtWidgets.QLabel('Max indices')
        self.label.setObjectName(_fromUtf8("Max indices"))
        gridLayout.addWidget(self.label, 1, 0, 1, 2)

        self.buttonBox = QtWidgets.QDialogButtonBox()
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        gridLayout.addWidget(self.buttonBox, 2, 0, 1, 3)
        gridLayout.addWidget(self.canvas, 0, 0, 1, 3)
        gridLayout.addWidget(self.toolbar, 3, 0, 1, 3)

        self.buttonBox.rejected.connect(self.close)
        self.buttonBox.accepted.connect(self.spectre)

        self.setLayout(gridLayout)

    def spectre(self):
        global x_space
        a2 = self.figure.add_subplot(111)
        a2.clear()
        abc = ui.abc_entry.text().split(",")
        a = np.float(abc[0])
        b = np.float(abc[1])
        c = np.float(abc[2])
        alphabetagamma = ui.alphabetagamma_entry.text().split(",")
        alpha = np.float(alphabetagamma[0])
        beta = np.float(alphabetagamma[1])
        gamma = np.float(alphabetagamma[2])
        e = np.int(self.lineEdit.text())
        alp = alpha * np.pi / 180
        bet = beta * np.pi / 180
        gam = gamma * np.pi / 180
        G = np.array([[a**2, a * b * np.cos(gam), a * c * np.cos(bet)], [a * b * np.cos(gam), b**2, b * c * np.cos(alp)], [a * c * np.cos(bet), b * c * np.cos(alp), c**2]])

        di = np.zeros((1, 4))
        cont = np.zeros((1, 1))
        space_group = ui.SpaceGroup_box.currentText()

        for tt in range(0, len(x_space)):
            if space_group == x_space[tt][0]:
                s = tt
        rr = s + 1

        while len(x_space[rr]) == 4:
            rr = rr + 1

        for h in range(-e, e + 1):
            for k in range(-e, e + 1):
                for l in range(-e, e + 1):
                    if (h, k, l) != (0, 0, 0):
                        gg = 0

                        for c in range(0, cont.shape[0]):
                            if 1 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l]))))) > cont[c] - 0.00000001 and 1 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l]))))) < cont[c] + 0.00000001:
                                gg = 1

                        if gg == 0:
                            di = np.vstack((di, (1 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l]))))), h, k, l)))
                            cont = np.vstack((cont, 1 / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l])))))))

        for dc in range(1, di.shape[0]):
            F = 0
            q = 2 * np.pi / di[dc, 0]
            for ii in range(s + 1, rr):
                f = str(x_space[ii][0])

                for z in range(0, len(x_scatt)):

                    if f == x_scatt[z][0]:
                        f = eval(x_scatt[z][1]) * np.exp(-eval(x_scatt[z][2]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][3]) * np.exp(-eval(x_scatt[z][4]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][5]) * np.exp(-eval(x_scatt[z][6]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][7]) * np.exp(-eval(x_scatt[z][8]) * (q / 4 / np.pi)**2) + eval(x_scatt[z][9])

                F = F + f * np.exp(2j * np.pi * (eval(x_space[ii][1]) * di[dc, 1] + eval(x_space[ii][2]) * di[dc, 2] + eval(x_space[ii][3]) * di[dc, 3]))

            ff = float(np.real(F * np.conj(F)))
            bar_width = 1 / np.shape(di)[0]
            if ff > 0.0000001:
                ann = str(int(di[dc, 1])) + str(int(di[dc, 2])) + str(int(di[dc, 3]))
                a2.text(di[dc, 0], ff, ann, rotation='vertical')
                a2.bar(di[dc, 0], ff, width=bar_width, align='center', color='red')

        plt.xlabel('Interplanar distance (nm)')
        plt.ylabel('Intensity (a.u.)')
        self.canvas.draw()

#########################
#
# Compute extinction conditions
#
############################


def extinction(space_group, h, k, l):
    global x_space, G
    F = 0
    s = 0
    q = 2 * np.pi / (np.sqrt(np.dot(np.array([h, k, l]), np.dot(np.linalg.inv(G), np.array([h, k, l])))))

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


def import_data():

    ui.ListBox_theo.clear()
    ui.diff_spot_Listbox.clear()
    ui.euler_listbox.clear()

    if ui.abc_entry.text() is None:
        return

    cryst()

    data_file = QtWidgets.QFileDialog.getOpenFileName(Index, "Open a data file", "", "*.txt")
    data = open(data_file[0], 'r')
    x0 = []

    for line in data:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        x0.append(list(map(str, line.split())))

    data.close()
    for item in x0:
        ui.diff_spot_Listbox.addItem(item[0])
        ss = ui.diff_spot_Listbox.count()
        item_diff = ui.diff_spot_Listbox.item(ss - 1)
        item_diff.setFlags(item_diff.flags() | QtCore.Qt.ItemIsEditable)


def export_data():
    s = [str(x.text()) for x in ui.diff_spot_Listbox.selectedItems()]
    res = [str(ui.euler_listbox.item(i).text()) for i in range(ui.euler_listbox.count())]
    name = QtWidgets.QFileDialog.getSaveFileName(Index, 'Save File')
    fout = open(name[0], 'w')
    fout.write('# Diffraction data file \n')
    for item in s:
        fout.write("%s\n" % item)

    fout.write('\n')
    fout.write('# Results \n')
    for item in res:
        fout.write("# %s\n" % item)

    fout.close()


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

####################
#
# Launch
#
# ##################


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
    QtWidgets.qApp.setApplicationName("Diffraction")
    Index = QtWidgets.QMainWindow()
    ui = diffractionUI.Ui_Diffraction()
    ui.setupUi(Index)
    figure = plt.figure()
    canvas = FigureCanvas(figure)
    ui.mplvl.addWidget(canvas)
    toolbar = NavigationToolbar(canvas, canvas)
    toolbar.setMinimumWidth(601)
    # Index.setWindowOpacity(0.8)
######################################################
#
# Import calibrations from txt files: microscope, E(kV),camera lebghth (cm), binning,px/A
#
######################################################

f_calib = open(os.path.join(os.path.dirname(__file__), 'calibrations.txt'), "r")

x_calib = []

for line in f_calib:
    x_calib.append(list(map(str, line.split())))

f_calib.close()
counter = len(x_calib)

for i in range(counter):
    ui.Calib_box.addItem(x_calib[i][0] + ' ' + x_calib[i][1] + 'keV ' + x_calib[i][2] + 'cm' + ', Binning:' + x_calib[i][3])

#########################
#
# Get space group data from inpt file space_group.txt
#
############################

f_space = open(os.path.join(os.path.dirname(__file__), 'space_group.txt'), "r")

x_space = []

for line in f_space:
    x_space.append(list(map(str, line.split())))

list_space_group = []
for i in range(0, len(x_space)):
    if len(x_space[i]) == 1:
        list_space_group.append(x_space[i][0])

f_space.close()


#############################
# import crystal structures from un txt file Name,a,b,c,alpha,beta,gamma,space group
#
############################

def structure(item):
    global x0, e_entry
    ui.abc_entry.setText(str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]))
    ui.alphabetagamma_entry.setText(str(item[4]) + ',' + str(item[5]) + ',' + str(item[6]))
    ii = ui.SpaceGroup_box.findText(item[7])
    ui.SpaceGroup_box.setCurrentIndex(ii)


file_struct = open(os.path.join(os.path.dirname(__file__), 'structure.txt'), "r")

x0 = []

for line in file_struct:
    x0.append(list(map(str, line.split())))

i = 0
file_struct.close()

for item in x0:
    entry = ui.menuStructure.addAction(item[0])
    entry.triggered.connect(lambda checked, item=item: structure(item))
    ui.SpaceGroup_box.addItem(x0[i][7])
    i = i + 1

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

ui.actionImport.triggered.connect(import_data)
ui.actionExport.triggered.connect(export_data)

ui.Button_reset.clicked.connect(reset_points)
ui.Button_reset_all.clicked.connect(reset)
ui.distance_button.clicked.connect(distance_theo)
dialSpect = Spect()
dialSpect.setWindowTitle("Spectrum")
ui.actionCalculate_spectrum.triggered.connect(dialSpect.exec_)
ui.add_spot_button.clicked.connect(add_spot)
ui.remove_spot_button.clicked.connect(remove_spot)
ui.orientation_button.clicked.connect(get_orientation)
ui.do_not_guess_checkBox.toggled.connect(guess)

ui.diff_spot_Listbox.setSelectionMode(QtWidgets.QListWidget.ExtendedSelection)
ui.n_entry.setText('1')
ui.indice_entry.setText('3')
ui.tilt_axis_angle_entry.setText('0')
ui.tilt_a_entry.setText('0')
ui.tilt_b_entry.setText('0')
ui.tilt_z_entry.setText('0')
ui.precision_entry.setText('10')
s = 1
gclick = np.zeros((1, 2))
guess()
Index.show()
app.exec_()
