#!/usr/bin/python3

import cv2
import numpy as np
import math
import cmath
from matplotlib import pyplot as plt


def DFT_manual(fx):
        """Vypocitaj jednorozmernu diskretnu fourierovu transformaciu pola fx"""
        M = len(fx)
        Fu = [0] * M
        for u in range(0, M):
                sum = 0
                for x in range(0, M):
                        sum += fx[x] * cmath.exp(-2j * cmath.pi * u * x / M)
                Fu[u] = sum
        return Fu

def IDFT_manual(Fu):
        """Vypocitaj inverznu jednorozmernu diskretnu fourierovu transformaciu pola Fu"""
        M = len(Fu)
        fx = [0] * M
        for x in range(0, M):
                sum = 0
                for u in range(0, M):
                        sum += Fu[u] * cmath.exp(2j * cmath.pi * u * x / M)
                fx[x] = sum / M
        return fx

img = cv2.imread('img/dataset/tilia-cordata/5.jpg')
height, width = img.shape[:2]

# odfiltruj zelenu
mask = cv2.inRange(img, np.array([0, 0, 0], dtype = "uint8"), np.array([130, 220, 220], dtype = "uint8"))
zelena = cv2.bitwise_and(img, img, mask = mask)

# najdi hrany
#hrany = cv2.Canny(zelena, 100, 200)
seda = cv2.cvtColor(zelena, cv2.COLOR_BGR2GRAY)
ret,prah = cv2.threshold(seda, 10, 255, 0)

# pospajaj
kernel = np.ones((5,5), np.uint8)
zatvor = cv2.morphologyEx(prah, cv2.MORPH_CLOSE, kernel)

# najdi obrys
contours, hierarchy = cv2.findContours(zatvor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_areas = sorted(contours, key=cv2.contourArea)
obrysCelyPovodnyKontura = largest_areas[-1]
obrysCelyPovodny = np.zeros(zatvor.shape, np.uint8)
cv2.drawContours(obrysCelyPovodny, [obrysCelyPovodnyKontura], 0, (255,0,0), -1)

# orez najdeny obrys
x,y,w,h = cv2.boundingRect(obrysCelyPovodnyKontura)
orezany = obrysCelyPovodny[y:y+h, x:x+w]

# zmen na standardnu velkost
nova_sirka = 900
r = nova_sirka / float(width)
nova_vyska = int(height*r)
preskalovany = cv2.resize(orezany, (nova_sirka, nova_vyska))
obrysCely = preskalovany
width = nova_sirka
height = nova_vyska

# najdi celu konturu obrysu
contours, hierarchy = cv2.findContours(obrysCely.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_areas = sorted(contours, key=cv2.contourArea)
obrysCelyKontura = largest_areas[-1]

# hladaj stonku
zrusenieStonky = cv2.erode(obrysCely, kernel, iterations=5)
opatovneNafuknutie = cv2.dilate(zrusenieStonky, kernel, iterations=5)
hladanieStonky = obrysCely - opatovneNafuknutie
contours, hierarchy = cv2.findContours(hladanieStonky.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_areas = sorted(contours, key=cv2.contourArea)
stonkaKontura = largest_areas[-1]
#stonka = np.zeros((height, width, 3), np.uint8)
stonka = np.zeros(hladanieStonky.shape, np.uint8)
cv2.drawContours(stonka, [stonkaKontura], 0, (255,0,0), -1)

# sprav obrys listu bez stonky
obrysListTmp = cv2.bitwise_and(obrysCely, cv2.bitwise_not(stonka))
contours, hierarchy = cv2.findContours(obrysListTmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_areas = sorted(contours, key=cv2.contourArea)
obrysListKontura = largest_areas[-1]

# najdi zaciatok stonky
points = np.transpose(np.nonzero(stonka))		# = cv2.findNonZero(stonka)
# zotried podla vzdialenosti od stredu obrazku
sorted(points, key = lambda p: np.sqrt((p[1] - width/2)**2 + (p[0] - height/2)**2))
sx = points[0, 1]
sy = points[0, 0]
# najdi tazisko listu
M = cv2.moments(obrysListKontura)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
# vypocitaj uhol
obrysUhol = np.zeros((height, width, 3), np.uint8)
cv2.drawContours(obrysUhol, [obrysCelyKontura], 0, (0,255,0), -1)
cv2.line(obrysUhol, (int(sx), int(sy)), (int(cx), int(cy)), (255, 0, 0), 2)
uhol = np.arctan2(sy-cy, sx-cx)
#uhol = -(np.pi - uhol)
uhol -= (np.pi/2)
#print(np.rad2deg(uhol))

'''
# sprav kostru (topological skeleton): http://opencvpython.blogspot.sk/2012/05/skeletonization-using-opencv-python.html
robenieKostry = stonka.copy()
kostra = np.zeros(robenieKostry.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
done = False
while (not done):
	eroded = cv2.erode(robenieKostry, element)
	temp = cv2.dilate(eroded, element)
	temp = cv2.subtract(robenieKostry, temp)
	kostra = cv2.bitwise_or(kostra, temp)
	robenieKostry = eroded.copy()
	zeros = np.size(robenieKostry) - cv2.countNonZero(robenieKostry)
	if zeros==np.size(robenieKostry):
		done = True
#cv2.imwrite('test.png', kostra)

# skumaj orientaciu stonky
# jedna moznost by mohla byt aproximacia bezierovou krivkou: http://jimherold.com/2012/04/20/least-squares-bezier-fit/
# skusme to ale manualne
# sprav zoznam suradnic pixelov
points = np.transpose(np.nonzero(kostra))		# = cv2.findNonZero(kostra)
# zotried podla vzdialenosti od stredu obrazku
sorted(points, key = lambda p: np.sqrt((p[1] - width/2)**2 + (p[0] - height/2)**2))
# vezmi desatinu bodov od stredu
k = len(points)/2
x = points[0:k, 1]
y = height-points[0:k, 0]
# preloz tieto data priamkou: y = a0 + a1*x
fit = np.polyfit(x, y, 1)
fit_fn = np.poly1d(fit)
print(fit_fn)
a1 = fit[0]
a0 = fit[1]
# vypocitaj uhol priamky
#TODO problem s pocitanim uhla alebo otacanim
uhol = np.rad2deg(np.arctan2(a1, 1))		# dy, dx
print(uhol)
'''

# otoc obrazok
otocScaleKoef = 1.5			# pri otacani sa velkost obrazku prenasobi tymto koeficientom
#M = cv2.getRotationMatrix2D((width/2, height/2), uhol, 1)
#otoceny = cv2.warpAffine(img, M, (width, height))
ox = width/2
oy = height/2
mx = width
my = height
for pos in obrysListKontura:
	px = pos[0,1]
	py = pos[0,0]
	nx = ox + math.cos(uhol) * (px - ox) - math.sin(uhol) * (py - oy)
	ny = oy + math.sin(uhol) * (px - ox) + math.cos(uhol) * (py - oy)
	pos[0,1] = nx
	pos[0,0] = ny
	if nx < mx:
		mx = nx
	if ny < my:
		my = ny
# posun na 0,0
for pos in obrysListKontura:
	pos[0,1] -= mx
	pos[0,0] -= my
# vykresli
otoceny = np.zeros((height*otocScaleKoef, width*otocScaleKoef, 3), np.uint8)
cv2.drawContours(otoceny, [obrysListKontura], 0, (255,0,0), -1)

# skumaj obrys listu
#x = obrysListKontura[:, :, 1]
#y = obrysListKontura[:, :, 0]
#k = map(complex, (x, y))
#print(list(k))
ks = []
for pos in obrysListKontura:
	k = complex(pos[:,1], pos[:,0])
	ks.append(k)
# sprav FT
Fu = np.fft.fft(ks)			# Fu = np.array(DFT_manual(ks))
# filtruj
l = 30
Fu[l:-l] = 0
#print(*Fu)
#python 2
for p in Fu: print p

# sprav IFT
ks = np.fft.ifft(Fu)			# ks = IDFT_manual(Fu)
# vykresli vyfiltrovany obrys
obrysFTdesc = np.zeros((height*otocScaleKoef, width*otocScaleKoef, 3), np.uint8)

px = -1
py = -1
for pos in ks:
	if px == -1 or py == -1:
		px = pos.real
		py = pos.imag
		continue
	cv2.line(obrysFTdesc, (int(py), int(px)), (int(pos.imag), int(pos.real)), (0, 0, 255), 2)
	px = pos.real
	py = pos.imag



# vykresli vysledok
plt.subplot(341)
plt.imshow(img, cmap = 'gray')
plt.title('Zdroj'), plt.xticks([]), plt.yticks([])
plt.subplot(342)
plt.imshow(zelena, cmap = 'gray')
plt.title('Zelena'), plt.xticks([]), plt.yticks([])
plt.subplot(343)
plt.imshow(zatvor, cmap = 'gray')
plt.title('Zatvor'), plt.xticks([]), plt.yticks([])
plt.subplot(344)
plt.imshow(obrysCelyPovodny, cmap = 'gray')
plt.title('Obrys cely'), plt.xticks([]), plt.yticks([])
plt.subplot(345)
plt.imshow(orezany, cmap = 'gray')
plt.title('Orezany'), plt.xticks([]), plt.yticks([])
plt.subplot(346)
plt.imshow(preskalovany, cmap = 'gray')
plt.title('Preskalovany'), plt.xticks([]), plt.yticks([])
plt.subplot(347)
plt.imshow(zrusenieStonky, cmap = 'gray')
plt.title('Zrusenie stonky'), plt.xticks([]), plt.yticks([])
plt.subplot(348)
plt.imshow(hladanieStonky, cmap = 'gray')
plt.title('Hladanie stonky'), plt.xticks([]), plt.yticks([])
plt.subplot(349)
plt.imshow(stonka, cmap = 'gray')
plt.title('Stonka'), plt.xticks([]), plt.yticks([])
plt.subplot(3,4,10)
plt.imshow(obrysUhol)
plt.title('Uhol'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,4,11)
#plt.imshow(kostra, cmap = 'gray')
#plt.title('Kostra'), plt.xticks([]), plt.yticks([])
#plt.subplot(3,4,12)
#plt.plot(x, y, 'yo', x, fit_fn(x), '--k')
#plt.title('Kostra'), plt.xticks([]), plt.yticks([])
plt.subplot(3,4,11)
plt.imshow(otoceny)
plt.title('Otoceny'), plt.xticks([]), plt.yticks([])
plt.subplot(3,4,12)
plt.imshow(obrysFTdesc, cmap = 'gray')
plt.title('Po FT'), plt.xticks([]), plt.yticks([])
plt.show()
