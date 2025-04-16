import cv2
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
import sys,os
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from labellines import labelLines
from matplotlib.font_manager import FontProperties
import pandas as pd

class Direction:
    def __init__(self,borderY = None,pic_h = None):
        self._borderY = borderY
        self._picH = pic_h
    #預處理
    def _preprocess(self,img):
        #將原始圖片轉為灰階，以進行後續處理。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #使用高斯模糊消除圖片中的雜訊，避免其干擾後續辨識。
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #將圖片進行二值化，突顯出要辨識的斜紋。
        _, threshold = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
        #使用 Morphological Transformations 清楚分開每條斜紋。 
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        return morph
    
    #保留每一個line的中心點。
    def getLines(self,morph_img):
        #找到所有輪廓。
        contours = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #找到第一級輪廓(最外圈輪廓)。
        contours = contours[0] if len(contours) == 2 else contours[1]
        #先準備一個空list，之後裝取所有在標準線以下的斜紋中心。
        pos = []
        
        for c in contours:
            #輪廓要至少有5個點才能配橢圓。
            if len(c) > 5:
                #將每個輪廓都配一個橢圓。
                ellipse = cv2.fitEllipse(c)
                #得到橢圓中心點、長短軸各自長度的一半、長軸角度。
                (cen_x,cen_y), (axis1, axis2), angle = ellipse
                #如果橢圓中心在標準線以下，就放入pos。
                if(cen_y>self._borderY):
                    pos.append([cen_x,cen_y])
        return pos #回傳在標準線以下的所有橢圓中心點。
    
    #以下這個函數用List中的第一個元素來排序，之後排序斜紋的時候會用到。
    def getIndex0Element(self,ls):
        return ls[0]
    
    #回傳第一幀圖片中的其中一條斜紋。
    def getFirstRope(self,pos):
        #將裝有所有中心點座標的2維list，轉成array。
        array_pos = np.array(pos)
        #將array裝成DataFrame，這樣可以利用X值做排序。
        df = pd.DataFrame(array_pos,columns = ["X","Y"])
        #用X欄位做排序，排列順序由小到大。
        df = df.sort_values(by = "X",ascending = True)
        #再轉回array，比較好取出座標中的x、y值。
        array_pos = np.array(df)
        
        #先算出標準線到這一幀最底下這個範圍中的中間y值。
        area_center_y = (self._picH+self._borderY)//2
        #將每條斜紋都與中間y值比較距離，組成[距離,座標]的斜紋資料。
        y_dis_pos = ([[abs(p[1]-area_center_y),p] for p in array_pos])
        #由距離值排序資料，由小到大。
        y_dis_pos.sort(key = self.getIndex0Element)
        #選離中間y值最近的一條斜紋資料，取得他的座標。
        closest_line = y_dis_pos[0][1]
        
        return closest_line#回傳離中間值y軸最近的斜紋中心座標。
    
    def getOtherFrameRope(self,pos,closest_line):
        #跟第一幀一樣的X排序作法。
        array_pos = np.array(pos)
        df = pd.DataFrame(array_pos,columns = ["X","Y"])
        df = df.sort_values(by = "X",ascending = True)
        array_pos = np.array(df)
        #將每個座標與第一幀取出來的斜紋做距離比較，得到每個斜紋與第一幀斜紋的資料。
        dis_pos = ([[((p[0]-closest_line[0])**2+(p[1]-closest_line[1])**2)**0.5,p] for p in array_pos])
        #用距離做排序，由小到大。
        dis_pos.sort(key = self.getIndex0Element)
        #取得最小距離的斜紋資料，並取出座標，此與第一幀斜紋為同一條斜紋。
        closest_OtherFrameLine = dis_pos[0][1] 
        #回傳與第一幀同一條的斜紋的座標。
        return closest_OtherFrameLine 
    
    #比較兩幀的lines:將每一條line在兩張圖片的中心y相減，判斷上下行。
    def decide(self,f,f2):
        #對圖片做預處理，做邊緣偵測(Canny)。
        m = self._preprocess(f)
        m_after = self._preprocess(f2)
        #取得第一幀的所有標準線以下斜紋座標。
        pos = self.getLines(m)
        #取得另一幀的所有標準線以下斜紋座標。
        pos_after = self.getLines(m_after)
        #取得第一幀的特定斜紋。
        certainLine = self.getFirstRope(pos)
        #取得與第一幀的特定斜紋相同的另一幀斜紋。
        sameAs_certainLine = self.getOtherFrameRope(pos_after,certainLine)
        
        #將兩條斜紋的y取出來：
        #第一幀的y。
        y0 = certainLine[1]
        #另一幀的y。
        y_after = sameAs_certainLine[1]
        #取得兩幀同一條斜紋的差值。
        delta_y = y0-y_after 
        #如果是負的，後面一幀比較下方，下行；如果是正的，後面一幀比較上方，上行。
        if delta_y<0:
            return False #判斷為下行。
        else:
            return True #判斷為上行。

class LineDetector:
    def __init__(self, borderY=None, upward = True):
        #界線的位置。
        self._borderY = borderY
        #鋼纜移動方向。
        self._upward = upward
        #用來儲存每條鋼纜上所有斜紋的平均中心點X座標。
        self._groupMeanCenterXs = []
    
    #偵測傳入的原始圖片[img]的斜紋。
    def detect(self, img):
        self._img = img
        #將原始圖片進行預處理。
        preprocessImg = self._preprocess(self._img)
        #使用Canny邊緣檢測找出斜紋的邊緣。
        canny = cv2.Canny(preprocessImg, 100, 100, apertureSize = 3)
        #找出圖片中的輪廓並對每一個輪廓都適配（Fit）一個橢圓。 
        fittedEllipses = self._findContourAndFitEllipse(canny)
        #移除屬於離群值的橢圓（面積異常小或大、角度異常小或大）。
        fittedEllipses = self._removeOutliersEllipses(fittedEllipses, lowerFactor = 2.0, upperFactor = 3.0)
        #找出橢圓長軸(以兩個端點表示)，用來表示鋼纜上的斜紋。
        fittedEllipses = self._findEllipseMajorAxes(fittedEllipses)
        #將橢圓照著長軸起點的X座標由小到大（圖片中由左到右）排序。
        fittedEllipses = sorted(fittedEllipses, key = lambda fe: fe['majorAxe']['startPoint'][0])
        #將橢圓分組，正常情況下鋼纜有幾條橢圓就有幾組。
        groupedFittedEllipses = self._groupEllipses(fittedEllipses)
        #如果分出來的組數異常，則不做後續處理，直接回傳None。
        if len(groupedFittedEllipses) != len(self._groupMeanCenterXs) and len(self._groupMeanCenterXs) > 0:
            return None
        #使用分好組的橢圓來獲得代表斜紋的直線，與該線的斜率與角度。
        self._lines, self._slopes, self._angles = self._computeSlopeAndAngle(groupedFittedEllipses)
        #如果鋼纜移動方向為向上。
        if self._upward:
            #將每組橢圓照著長軸起點旋轉後的Y座標由小到大（圖片中由上到下）排序。
            for i in range(0, len(groupedFittedEllipses)):
                sortResult = sorted(zip(groupedFittedEllipses[i], self._lines[i], self._slopes[i], self._angles[i]), key = lambda x: self._rotatePointAroundImageCenter(x[0]['majorAxe']['startPoint'], np.mean(self._angles[i]))[1])
                groupedFittedEllipses[i] = [x[0] for x in sortResult]
                self._lines[i] = [x[1] for x in sortResult]
                self._slopes[i] = [x[2] for x in sortResult]
                self._angles[i] = [x[3] for x in sortResult]
        #如果鋼纜移動方向為向下。
        else:
            #將每組橢圓照著長軸起點旋轉後的Y座標由大到小（圖片中由下到上）排序。
            for i in range(0, len(groupedFittedEllipses)):
                sortResult = sorted(zip(groupedFittedEllipses[i], self._lines[i], self._slopes[i], self._angles[i]), key = lambda x: self._rotatePointAroundImageCenter(x[0]['majorAxe']['startPoint'], np.mean(self._angles[i]))[1], reverse = True)
                groupedFittedEllipses[i] = [x[0] for x in sortResult]
                self._lines[i] = [x[1] for x in sortResult]
                self._slopes[i] = [x[2] for x in sortResult]
                self._angles[i] = [x[3] for x in sortResult]
        #將代表同一個斜紋的破碎橢圓接在一起。
        groupedFittedEllipses = self._combineSmallEllipses(groupedFittedEllipses)
        #平移所有橢圓，使同組的橢圓有相同的中心點X座標（這樣計數器在判別斜紋時會更精確）。
        groupedFittedEllipses = self._translateEllipses(groupedFittedEllipses)
        #補齊因光線、陰影等外部條件而沒有被辨識到的斜紋。只有界線之前的斜紋才需要補，因為過了界線有沒有補都不影響計數。
        groupedFittedEllipses = self._compensate(groupedFittedEllipses)
        return (self._lines, self._slopes, self._angles)
    
    #將原始圖片[img]進行預處理。
    def _preprocess(self, img):
        #將原始圖片轉為灰階，以進行後續處理。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #使用高斯模糊消除圖片中的雜訊，避免其干擾後續辨識。
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #將圖片進行二值化，突顯出要辨識的斜紋。
        _, threshold = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY)
        #使用 Morphological Transformations 清楚分開每條斜紋。 
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        return morph
    
    #找出圖片[img]中的輪廓並對每一個輪廓都適配（Fit）一個橢圓。
    def _findContourAndFitEllipse(self, img):
        #找出圖片中的輪廓。 
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        #存放適配輪廓的橢圓及其他相關資訊。
        fittedEllipse = []
        for c in contours:
            #輪廓至少要由5個以上的點組成，才能用橢圓適配。
            if len(c) > 5:
                #橢圓本身，裡面包含橢圓的中心點，長短軸長度以及傾斜角度。
                ellipse = cv2.fitEllipse(c)
                #利用橢圓的長軸與短軸來計算橢圓面積。
                a, b = ellipse[1]
                #短軸長度。
                minorLength = min(a,b)
                #長軸長度。
                majorLength = max(a,b)
                #計算橢圓的面積。
                area = (minorLength / 2) * (majorLength / 2) * np.pi
                #新增橢圓本身及其面積。
                fittedEllipse.append({'ellipse':ellipse, 'area':area})
        return fittedEllipse
    
    '''
    移除[ellipses]中屬於離群值的橢圓（面積異常小或大、角度異常小或大），[minArea]代表接受的最小橢圓面積，面積小於[minArea]的橢圓會直接被移除。
    [lowerFactor]與[upperFactor]用來設定移除標準的嚴格程度。[lowerFactor]與[upperFactor]越大，被移除的橢圓越少，[lowerFactor]與[upperFactor]越小，被移除的橢圓越多。
    '''
    def _removeOutliersEllipses(slef, ellipses, minArea = 80, lowerFactor = 1.0, upperFactor = 1.0):
        #使用 Median Absolute Deviation(MAD) 方法來檢測橢圓面積與角度的離群值，移除面積異常小或大、角度異常小或大的橢圓。
        #先將面積小於 minArea 的橢圓直接移除，避免其影響中位數（Median）的大小，使得正常面積的橢圓反而被當成離群值。
        ellipses = list(filter(lambda e: e['area'] > minArea, ellipses))
        areas = [e['area'] for e in ellipses]
        angles = [e['ellipse'][2] for e in ellipses]
        medianAreas = np.median(areas)
        medianAngles = np.median(angles)
        diffAreas = np.abs(areas - medianAreas)
        diffAngles = np.abs(angles - medianAngles)
        scalingFactorAreas = np.median(diffAreas)
        scalingFactorAngles = np.median(diffAngles)
        lowerAreas = medianAreas - lowerFactor * scalingFactorAreas
        upperAreas = medianAreas + upperFactor * scalingFactorAreas
        lowerAngles = medianAngles - lowerFactor * scalingFactorAngles
        upperAngles = medianAngles + upperFactor * scalingFactorAngles
        #移除面積異常小或大、角度異常小或大的橢圓。
        ellipses = list(filter(lambda e: (e['area'] < upperAreas and e['area'] > lowerAreas and e['ellipse'][2] < upperAngles and e['ellipse'][2] > lowerAngles), ellipses))
        #橢圓面積之後用不到，先移除。
        ellipses = [{k: v for k, v in e.items() if k != 'area'} for e in ellipses]
        return ellipses
    
    #找出[ellipses]橢圓的長軸(以兩個端點表示)，用來表示鋼纜上的斜紋。
    def _findEllipseMajorAxes(self, ellipses):
        for e in ellipses:
            #橢圓的中心、長短軸與角度。
            (x0, y0), (a, b), angle = e['ellipse']
            majorLength = max(a,b) / 2
            if angle > 90:
                angle = angle - 90 #左上往右下
            else:
                angle = angle + 90 #左下往右上
            #橢圓長軸的起點。
            startPoint = (int(round(x0 + majorLength * np.cos(np.radians(angle)))), int(round(y0 + majorLength * np.sin(np.radians(angle)))))
            #橢圓長軸的終點。
            endPoint = (int(round(x0 + majorLength * np.cos(np.radians(angle + 180)))), int(round(y0 + majorLength * np.sin(np.radians(angle + 180)))))
            #新增橢圓長軸。
            e['majorAxe'] = {'startPoint': startPoint, 'endPoint': endPoint}
        return ellipses
    
    #將橢圓[ellipses]分組，正常情況下鋼纜有幾條橢圓就有幾組。
    def _groupEllipses(self, ellipses):
        #存放已分組的橢圓，並將第一個橢圓直接先放到第一組。
        groupedEllipses = [[ellipses[0]]]
        for index, MA in enumerate([e['majorAxe'] for e in ellipses[1:]]):
            #該橢圓起點的X座標。
            currentStartPointX = MA['startPoint'][0]
            #該橢圓終點的X座標。
            currentEndPointX = MA['endPoint'][0]
            #該組平均長軸終點的X座標。
            meanGroupEndPointX = np.mean([e['majorAxe']['endPoint'][0] for e in groupedEllipses[-1]])
            #用來判斷該組是否已分完。
            inNextGroup = True
            #將該橢圓與該組所有現有橢圓的位置進行比較，判斷該橢圓是否應被分到該組。
            for previousMA in [e['majorAxe'] for e in groupedEllipses[-1]]:
                previousStartPointX = previousMA['startPoint'][0]
                previousEndPointX = previousMA['endPoint'][0]
                #如果該橢圓與該組任一個現有橢圓有交錯，且其長軸起點的X座標小於該組平均長軸終點的X座標，則將它分到該組。
                if ((currentStartPointX < previousStartPointX and currentEndPointX > previousStartPointX) or (currentEndPointX > previousStartPointX and currentStartPointX < previousEndPointX) or (currentStartPointX > previousStartPointX and currentEndPointX < previousEndPointX) or (currentStartPointX < previousStartPointX and currentEndPointX > previousEndPointX)) and (currentStartPointX < meanGroupEndPointX):
                    groupedEllipses[-1].append(ellipses[index + 1])
                    inNextGroup = False
                    break
            #該橢圓不屬於該組，代表該組已分完。
            if inNextGroup:
                #將該橢圓分到下一組。
                groupedEllipses.append([ellipses[index+1]])
        #少數破碎的橢圓會被演算法單獨分成一組，造成組數異常，所以將橢圓數量極少的組移除（這裡只將橢圓數量大於5的組留下）。
        groupedEllipses = list(filter(lambda g: len(g) > 5 , groupedEllipses))
        return groupedEllipses
    
    #透過[groupedEllipses]來獲得代表斜紋的直線，與該線的斜率與角度。
    def _computeSlopeAndAngle(self, groupedEllipses):
        #存放已分組的橢圓長軸。
        groupedLines = [[0] * len(group) for group in groupedEllipses]
        #存放已分組的斜紋斜率。
        groupedSlopes = [[0] * len(group) for group in groupedEllipses]
        #存放已分組的斜紋角度。
        groupedAngles = [[0] * len(group) for group in groupedEllipses]
        for index1, group in enumerate(groupedEllipses):
            for index2, e in enumerate(group):
                #橢圓長軸起點。
                startPoint = e['majorAxe']['startPoint']
                #橢圓長軸終點。
                endPoint = e['majorAxe']['endPoint']
                #斜紋斜率。
                slope = (startPoint[1] - endPoint[1]) / (endPoint[0] - startPoint[0])
                #斜紋角度。
                angle = np.arctan(slope) * 180 / np.pi
                groupedLines[index1][index2] = (startPoint, endPoint)
                groupedSlopes[index1][index2] = slope
                groupedAngles[index1][index2] = angle
        return (groupedLines, groupedSlopes, groupedAngles)
    
    #將[groupedEllipses]中代表同一個斜紋的破碎橢圓接在一起。
    def _combineSmallEllipses(self, groupedEllipses):
        #存放新的已分組的橢圓。
        newGroupedEllipses = []
        #存放新的已分組的橢圓長軸。
        newGroupedLines = []
        #存放新的已分組的斜紋斜率。
        newGroupedSlopes = []
        #存放新的已分組的斜紋角度。
        newGroupedAngles = []
        for index1, group in enumerate(groupedEllipses):
            newEllipses = []
            newLines = []
            newSlopes = []
            newAngles = []
            #用來判斷上個橢圓是否有跟目前這個橢圓合併。
            previousCombined = False
            for index2, e in enumerate(group[:-1]):
                #上個橢圓已跟目前這個橢圓合併，不用再將這個橢圓與其他橢圓合併。
                if previousCombined:
                    previousCombined = False
                    continue
                #目前這個橢圓的中心與長短軸長度。
                (x1, y1), (a1, b1), _ = e['ellipse']
                #下個橢圓的中心與長短軸長度。
                (x2, y2), (a2, b2), _ = group[index2 + 1]['ellipse']
                #兩個橢圓的中心相連為一垂直線，代表它們不代表同一個斜紋，不用合併。
                if x2 - x1 == 0:
                    #新增目前這個橢圓。
                    newEllipses.append(e)
                    #新增目前這個橢圓的長軸。
                    newLines.append(self._lines[index1][index2])
                    #新增目前這個橢圓的長軸斜率（斜紋斜率）。
                    newSlopes.append(self._slopes[index1][index2])
                    #新增目前這個橢圓的長軸角度（斜紋角度）。
                    newAngles.append(self._angles[index1][index2])
                    previousCombined = False
                    continue
                #兩個橢圓中心相連的直線的斜率。
                combinedSlope = (y1 - y2) / (x2 - x1)
                #使用 Median Absolute Deviation(MAD) 方法來檢測 combinedSlope 是否比該組其他橢圓的長軸斜率顯著大或小。
                groupSlopes = self._slopes[index1]
                groupSlopes.append(combinedSlope)
                medianSlopes = np.median(groupSlopes)
                diffSlopes = np.abs(groupSlopes - medianSlopes)
                scalingFactorSlopes = np.median(diffSlopes)
                lowerSlopes = medianSlopes - 2.0 * scalingFactorSlopes
                upperSlopes = medianSlopes + 2.0 * scalingFactorSlopes
                #combinedSlope 與該組其他橢圓的長軸斜率沒有顯著差異，代表兩個橢圓原本代表同一個斜紋，需將它們合併。
                if not (combinedSlope < lowerSlopes or combinedSlope > upperSlopes):
                    #合併後的新橢圓、新長軸、新斜率與新角度。
                    newEllipse, newLine, newSlope, newAngle = self._createCombinedEllipse(e, group[index2 + 1])
                    #新增合併後的橢圓。
                    newEllipses.append(newEllipse)
                    #新增合併後的橢圓長軸。
                    newLines.append(newLine)
                    #新增合併後的橢圓長軸斜率（斜紋斜率）。
                    newSlopes.append(newSlope)
                    #新增合併後的橢圓長軸角度（斜紋角度）。
                    newAngles.append(newAngle)
                    previousCombined = True
                #combinedSlope 與該組其他橢圓的長軸斜率有顯著差異，代表兩個橢圓原本代表不同斜紋，不需將它們合併。
                else:
                    #新增目前這個橢圓。
                    newEllipses.append(e)
                    #新增目前這個橢圓的長軸。
                    newLines.append(self._lines[index1][index2])
                    #新增目前這個橢圓的長軸斜率（斜紋斜率）。
                    newSlopes.append(self._slopes[index1][index2])
                    #新增目前這個橢圓的長軸角度（斜紋角度）。
                    newAngles.append(self._angles[index1][index2])
                    previousCombined = False
            newGroupedEllipses.append(newEllipses)
            newGroupedLines.append(newLines)
            newGroupedSlopes.append(newSlopes)
            newGroupedAngles.append(newAngles)
        self._lines = newGroupedLines
        self._slopes = newGroupedSlopes
        self._angles = newGroupedAngles
        return newGroupedEllipses
    
    #平移[groupedEllipses]中的所有橢圓，使同組的橢圓有相同的中心點X座標（這樣計數器在判別斜紋時會更精確）。
    def _translateEllipses(self, groupedEllipses):
        #初始情況，groupMeanCenterXs 尚未有任何值。
        if len(self._groupMeanCenterXs) == 0:
            #計算每條鋼纜上所有斜紋的平均中心點X座標。
            self._groupMeanCenterXs = [np.mean([e['ellipse'][0][0] for e in group]) for group in groupedEllipses]
        for index1, group in enumerate(groupedEllipses):
            #該組橢圓的平均中心點X座標，也就是要平移到的位置。
            groupMeanCenterX = self._groupMeanCenterXs[index1]
            for index2, e in enumerate(group):
                #該橢圓的中心點座標。
                centerX, centerY = e['ellipse'][0]
                #該橢圓的長軸斜率。
                slope = self._slopes[index1][index2]
                #長軸的直線方程式的常數項（設 y = ax + b，再將 centerX, centerY 代入 x 與 y，slope 代入 a）。
                b = centerY + slope * centerX
                #平移後的橢圓中心點 Y 座標。
                newCenterY = -slope * groupMeanCenterX + b
                #更新橢圓的中心點座標。
                e['ellipse'] = ((groupMeanCenterX, newCenterY), e['ellipse'][1], e['ellipse'][2])
                #更新橢圓的長軸。
                newStartPointX = int(round(e['majorAxe']['startPoint'][0] + (groupMeanCenterX - centerX)))
                newStartPointY = int(round(-slope * newStartPointX + b))
                newEndPointX = int(round(e['majorAxe']['endPoint'][0] + (groupMeanCenterX - centerX)))
                newEndPointY = int(round(-slope * newEndPointX + b))
                self._lines[index1][index2] = ((newStartPointX, newStartPointY), (newEndPointX, newEndPointY))
        return groupedEllipses
    '''
    透過[groupedEllipses]找出因光線、陰影等外部條件而沒有被辨識到的斜紋，並補齊斜紋。只有界線之前的斜紋才需要補，因為過了界線有沒有補都不影響計數。
    補斜紋其實就是在補橢圓。
    '''
    def _compensate(self, groupedEllipses):
        #存放新的已分組的橢圓。
        newGroupedEllipses = []
        #存放新的已分組的橢圓長軸。
        newGroupedLines = []
        #存放新的已分組的斜紋斜率。
        newGroupedSlopes = []
        #存放新的已分組的斜紋角度。
        newGroupedAngles = []
        for index1, group in enumerate(groupedEllipses):
            #使用 Median Absolute Deviation(MAD) 方法來判斷組內相鄰兩橢圓的間距是否異常大，是的話代表兩橢圓間有斜紋沒被辨識到。
            groupGaps = [np.abs(group[i + 1]['ellipse'][0][1] - e['ellipse'][0][1]) for i, e in enumerate(group[:-1])]
            medianGroupGaps = np.median(groupGaps)
            diffGroupGaps = np.abs(groupGaps - medianGroupGaps)
            scalingFactorGroupGaps = np.median(diffGroupGaps)
            upperGroupGaps = medianGroupGaps + 30.0 * scalingFactorGroupGaps
            try:
                #找出要從第幾個橢圓開始補斜紋。鋼纜為上行時，則從界線上方100像素開始往下補。鋼纜為下行時，則從界線下方100像素開始往上補。
                first = next(i for i, e in enumerate(group) if e['ellipse'][0][1] > (self._borderY - 100)) if self._upward else next(i for i, e in enumerate(group) if e['ellipse'][0][1] < (self._borderY + 100))
                newEllipses = group[:first]
                newLines = self._lines[index1][:first]
                newSlopes = self._slopes[index1][:first]
                newAngles = self._angles[index1][:first]
                for index2, e in enumerate(group[first:-1]):
                    newEllipses.append(e)
                    newLines.append(self._lines[index1][first + index2])
                    newSlopes.append(self._slopes[index1][first + index2])
                    newAngles.append(self._angles[index1][first + index2])
                    #目前這個橢圓與下一個橢圓的間距。
                    gap = np.abs(group[first + index2 + 1]['ellipse'][0][1] - e['ellipse'][0][1])
                    #如果間距過大則要補斜紋。
                    if gap > upperGroupGaps:
                        #計算需要補幾個斜紋。
                        nCompensate = int(round(gap / medianGroupGaps)) - 1
                        #從目前這個橢圓開始算，每隔多遠要補一個斜紋。
                        newGap = gap / (nCompensate + 1)
                        #補斜紋（補橢圓）。
                        for i in range(1, nCompensate + 1):
                            #補的橢圓的中心點座標。X座標為該組橢圓的平均中心點X座標，Y座標 = (目前這個橢圓的中心點Y座標) ＋ newGap * i 。
                            center = (self._groupMeanCenterXs[index1], e['ellipse'][0][1] + newGap * i) if self._upward else (self._groupMeanCenterXs[index1], e['ellipse'][0][1] - newGap * i)
                            #補的橢圓的短軸長度，為該組橢圓的平均短軸長度。
                            maLength = np.mean([min(e['ellipse'][1]) for e in group])
                            #補的橢圓的長軸長度，為該組橢圓的平均長軸長度。
                            MALength = np.mean([max(e['ellipse'][1]) for e in group])
                            #補的橢圓的斜率。
                            slope = np.mean(self._slopes[index1])
                            #補的橢圓的角度。
                            angle = np.mean(self._angles[index1])
                            newAngle = 90 - angle
                            if newAngle > 90:
                                newAngle = newAngle - 90
                            else:
                                newAngle = newAngle + 90
                            #補的橢圓的長軸起點。
                            startPoint = (int(round(center[0] + (MALength / 2) * np.cos(np.radians(newAngle)))), int(round(center[1] + (MALength / 2) * np.sin(np.radians(newAngle)))))
                            #補的橢圓的長軸終點。
                            endPoint = (int(round(center[0] + (MALength / 2) * np.cos(np.radians(newAngle + 180)))), int(round(center[1] + (MALength / 2) * np.sin(np.radians(newAngle + 180)))))
                            #補的橢圓的長軸。
                            majorAxe = {'startPoint': startPoint, 'endPoint': endPoint}
                            #補的橢圓本身。
                            ellipse = (center, (maLength, MALength), 90 - angle)
                            newEllipses.append({'ellipse': ellipse, 'majorAxe': majorAxe})
                            newLines.append((startPoint, endPoint, True))
                            newSlopes.append(slope)
                            newAngles.append(angle)
                    #補到該組倒數第二個果園則不用繼續補，直接新增該組最後一個橢圓。
                    if index2 == len(group[first:-1]) - 1:
                        newEllipses.append(group[first + index2 + 1])
                        newLines.append(self._lines[index1][first + index2 + 1])
                        newSlopes.append(self._slopes[index1][first + index2 + 1])
                        newAngles.append(self._angles[index1][first + index2 + 1])
            #沒有需要補的斜紋（界線之前沒有辨識到任何斜紋）。
            except StopIteration:
                newGroupedEllipses.append(group)
                newGroupedLines.append(self._lines[index1])
                newGroupedSlopes.append(self._slopes[index1])
                newGroupedAngles.append(self._angles[index1])
                continue
            newGroupedEllipses.append(newEllipses)
            newGroupedLines.append(newLines)
            newGroupedSlopes.append(newSlopes)
            newGroupedAngles.append(newAngles)
        self._lines = newGroupedLines
        self._slopes = newGroupedSlopes
        self._angles = newGroupedAngles
        return newGroupedEllipses
    
    #合併兩個橢圓[e1]與[e2]。
    def _createCombinedEllipse(self, e1, e2):
        center = None
        maLength = (min(e1['ellipse'][1]) + min(e2['ellipse'][1])) / 2
        MALength = None
        slope = None
        angle = None
        majorAxe = None
        MALength1 = np.sqrt((e1['majorAxe']['startPoint'][0] - e2['majorAxe']['endPoint'][0]) ** 2 + (e1['majorAxe']['startPoint'][1] - e2['majorAxe']['endPoint'][1]) ** 2)
        MALength2 = np.sqrt((e1['majorAxe']['endPoint'][0] - e2['majorAxe']['startPoint'][0]) ** 2 + (e1['majorAxe']['endPoint'][1] - e2['majorAxe']['startPoint'][1]) ** 2)
        if MALength1 > MALength2:
            center = (int(round((e1['majorAxe']['startPoint'][0] + e2['majorAxe']['endPoint'][0]) / 2)), int(round((e1['majorAxe']['startPoint'][1] + e2['majorAxe']['endPoint'][1]) / 2)))
            MALength = MALength1
            slope = (e1['majorAxe']['startPoint'][1] - e2['majorAxe']['endPoint'][1]) / (e2['majorAxe']['endPoint'][0] - e1['majorAxe']['startPoint'][0])
            angle = np.arctan(slope) * 180 / np.pi
            majorAxe = {'startPoint': e1['majorAxe']['startPoint'], 'endPoint': e2['majorAxe']['endPoint']}
        else:
            center = (int(round((e1['majorAxe']['endPoint'][0] + e2['majorAxe']['startPoint'][0]) / 2)), int(round((e1['majorAxe']['endPoint'][1] + e2['majorAxe']['startPoint'][1]) / 2)))
            MALength = MALength2
            slope = (e2['majorAxe']['startPoint'][1] - e1['majorAxe']['endPoint'][1]) / (e1['majorAxe']['endPoint'][0] - e2['majorAxe']['startPoint'][0])
            angle = np.arctan(slope) * 180 / np.pi
            majorAxe = {'startPoint': e2['majorAxe']['startPoint'], 'endPoint': e1['majorAxe']['endPoint']}
        return ({'ellipse': (center, (maLength, MALength), 90 - angle), 'majorAxe': majorAxe}, (majorAxe['startPoint'], majorAxe['endPoint']), slope, angle)
    
    #計算將圖片上的一點[point]以圖片中心為錨點旋轉[angle]角度後的新座標。
    def _rotatePointAroundImageCenter(self, point, angle):
        originalX, originalY = point
        imageCenterX = int(round(self._img.shape[1] / 2))
        imageCenterY = int(round(self._img.shape[0] / 2))
        originalX -= imageCenterX
        originalY -= imageCenterY
        angle = angle * np.pi / 180
        rotatedX = int(round(originalX * np.cos(angle) - originalY * np.sin(angle))) + imageCenterX
        rotatedY = int(round(originalX * np.sin(angle) + originalY * np.cos(angle))) + imageCenterY
        return (rotatedX, rotatedY)


class LineTracker:
    def __init__(self, nGroup, borderY, upward = True, maxDisappear = 5):
        #鋼纜數量。
        self._nGroup = nGroup
        #界線的位置。
        self._borderY = borderY
        #鋼纜移動方向。
        self._upward = upward
        #斜紋最多能消失幾幀還能繼續被追蹤。
        self._maxDisappear = maxDisappear
        #每組提供給新被追蹤的斜紋的Id。
        self._nextLineIds = [0] * self._nGroup
        #每組被追蹤中的斜紋。
        self._groupedLines = [OrderedDict() for i in range(0, self._nGroup)]   
        #每組被追蹤中的斜紋已消失了幾幀。
        self._groupedDisappears = [OrderedDict() for i in range(0, self._nGroup)] 
    
    #將斜紋[line]註冊到第[index]組，開始追蹤它。
    def _register(self, index, line):
        self._groupedLines[index][self._nextLineIds[index]] = line
        self._groupedDisappears[index][self._nextLineIds[index]] = 0
        self._nextLineIds[index] += 1
    
    #將第[index]組中Id為[lineId]的斜紋取消註冊，不再追蹤它。
    def _deregister(self, index, lineId):
        del self._groupedLines[index][lineId]
        del self._groupedDisappears[index][lineId]
        
    #透過斜紋的中心點在幀與幀間的變化來追蹤斜紋。
    def track(self, groupedLines):
        #沒有辨識到任何斜紋的情況，將所有現有斜紋的disappear加一，加完後若超過maxDisappear則取消註冊該斜紋。
        if sum([len(group) for group in groupedLines]) == 0:
            for index, lineIds in enumerate([list(d.keys()) for d in self._groupedDisappears]):
                for lineId in lineIds:
                    self._groupedDisappears[index][lineId] += 1
                    if self._groupedDisappears[index][lineId] > self._maxDisappear:
                        self._deregister(index ,lineId)
            return self._groupedLines
        #初始情況，替第一幀中所有斜紋註冊。
        if sum([len(lines) for lines in self._groupedLines]) == 0:
            for index, lines in enumerate(groupedLines):
                for i in range(0, len(lines)):
                    if (self._upward and Utils.getLineCentroid(lines[i])[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(lines[i])[1] < self._borderY):
                        self._register(index, lines[i])
        #若有組在第一幀中沒有任何斜紋被註冊，則在此先註冊。
        for index, lines in enumerate(self._groupedLines):
            if len(lines) == 0:
                for l in groupedLines[index]:
                    if (self._upward and Utils.getLineCentroid(l)[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(l)[1] < self._borderY):
                        self._register(index, l)
        #透過目前已被追蹤的舊斜紋與這一幀辨識到的新斜紋，其中心點之間的距離來判斷新斜紋是來自哪一條舊斜紋。
        for index, lines in enumerate(self._groupedLines):
            #新斜紋的Id。
            lineIds = list(lines.keys())
            #新斜紋的中心點。
            centroids = [Utils.getLineCentroid(l) for l in list(lines.values())]
            #如沒有辨識到任何屬於這組的新斜紋，則不處理這組。
            if len(centroids) == 0:
                continue
            #計算舊斜紋與新斜紋中心點之間的距離矩陣。
            D = dist.cdist(np.array(centroids), [Utils.getLineCentroid(l) for l in groupedLines[index]])
            #將距離矩陣從左至右，由距離小到大排序。
            rows = D.min(axis = 1).argsort()
            cols = D.argmin(axis = 1)[rows]
            usedRows = set()
            usedCols = set()
            #對於每個舊斜紋，與其距離最近的新斜紋即為同一個斜紋。
            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                lineId = lineIds[row]
                self._groupedLines[index][lineId] = groupedLines[index][col]
                self._groupedDisappears[index][lineId] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            #尚未處理到的舊斜紋（沒有新斜紋與其對應）。
            for row in unusedRows:
                lineId = lineIds[row]
                #將該斜紋已消失的幀數加一。
                self._groupedDisappears[index][lineId] += 1
                #如果該斜紋已消失的幀數超過maxDisappear，則取消追蹤它。
                if self._groupedDisappears[index][lineId] > self._maxDisappear:
                    self._deregister(index, lineId)
            #尚未處理到的新斜紋（沒有舊斜紋與其對應）。
            for col in unusedCols:
                #如果該斜紋位於界線以前，則追蹤它。
                if (self._upward and Utils.getLineCentroid(groupedLines[index][col])[1] > self._borderY) or (not self._upward and Utils.getLineCentroid(groupedLines[index][col])[1] < self._borderY): 
                    self._register(index, groupedLines[index][col])
        return self._groupedLines

class LineCounter:
    def __init__(self, nGroup, borderY, upward = True):
        #鋼纜數量。
        self._nGroup = nGroup
        #界線的位置。
        self._borderY = borderY
        #鋼纜移動方向。
        self._upward = upward
        #每組的斜紋數量。
        self._counts = [0] * self._nGroup
        #每組的累積斜紋數量。
        self._cumCounts = []
        #每組的斜紋補償數量。
        self._compensateCounts = [0] * self._nGroup
        #每組的累積斜紋補償數量。
        self._cumCompensateCounts = []
        #存放已辨識過的斜紋的Id。
        self._examinedLines = []
    
    def count(self, groupedLines):
        for index, group in enumerate(groupedLines):
            for lineId, line in group.items():
                #用來在所有斜紋間區別的Id。
                newId = '{}-{}'.format(index+1, lineId)
                #如果該斜紋的中心超過界線且該斜紋不在examinedLines中，則需更新計數。
                if ((self._upward and Utils.getLineCentroid(line)[1] < self._borderY) or (not self._upward and Utils.getLineCentroid(line)[1] > self._borderY)) and newId not in self._examinedLines:
                    #該組斜紋數量加一。
                    self._counts[index] += 1
                    #新增目前的累積斜紋數量。
                    self._cumCounts.append(self._counts.copy())
                    #該斜紋是補的。
                    if len(line) == 3:
                        #該組斜紋補償數量加一。
                        self._compensateCounts[index] += 1
                    #新增目前的累積補償斜紋數量。
                    self._cumCompensateCounts.append(self._compensateCounts.copy())
                    #將該斜紋的Id加入examinedLines，避免重複辨識。
                    self._examinedLines.append(newId)
        return (self._counts, self._cumCounts, self._compensateCounts, self._cumCompensateCounts)

class Utils:
    @staticmethod
    #每條鋼纜使用的顏色。
    def groupColors():
        return [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 140, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255), (102, 102, 255)]
    
    #將影片經過的毫秒數[milliseconds]轉為「小時：分鐘：秒」的格式。
    @staticmethod
    def milliseconds2HMS(milliseconds):
        milliseconds = int(milliseconds)
        seconds = (milliseconds / 1000) % 60
        seconds = int(seconds)
        seconds = str(seconds) if seconds > 9 else '0{}'.format(seconds)
        minutes = (milliseconds / (1000 * 60)) % 60
        minutes = int(minutes)
        minutes = str(minutes) if minutes > 9 else '0{}'.format(minutes)
        hours = (milliseconds / (1000 * 60 * 60)) % 24
        hours = int(hours)
        hours = str(hours) if hours > 9 else '0{}'.format(hours)
        return (hours, minutes, seconds)
    
    #計算線段[line]的中心點。
    @staticmethod
    def getLineCentroid(line):
        _x = line[0][0] + line[1][0]
        if type(_x) is not tuple:
            return (int(round((line[0][0] + line[1][0]) / 2)), int(round((line[0][1] + line[1][1]) / 2)))
        else:
            return 0,0

#font_tc = FontProperties("Iansui-Regular.ttf",size = 10)

class Grapher:
    def __init__(self):
        #設定字體，讓圖表能正常顯示中文。
        plt.rcParams['font.sans-serif'] = ['SimHei']
    
    #畫出斜紋累積總數[cumCounts]、斜紋累積補償數[cumCompensateCounts]的折線圖與斜紋總數[counts]、斜紋補償數[compensateCounts]的表格。
    def plotCountsGraphAndTable(self, counts, compensateCounts, cumCounts, cumCompensateCounts):
        _, axs = plt.subplots(2, sharex = True, figsize=(40, 30))
        #幀數，為X軸的單位。
        frames = list(range(1, len(cumCounts) + 1))
        #斜紋累積總數的折線圖。
        cumCounts = [list(c) for c in zip(*cumCounts)]
        #依序用不同顏色畫出每條鋼纜累積總數的折線。
        for i, c in enumerate(cumCounts):
            color = tuple(reversed(Utils.groupColors()[i]))
            color = tuple(c / 255 for c in color)
            axs[0].plot(frames, c, label = i + 1, color = color)
        #在折線上標示其代表第幾條鋼纜。
        labelLines(axs[0].get_lines(), align = False, fontsize = 36)
        #標示折線圖標題。
        axs[0].set_title('lines_total', fontsize = 40)
        #標示折線圖Y軸名稱。
        axs[0].set_ylabel('amount', fontsize = 36, rotation = 0, labelpad = 40)
        #設定折線圖Y軸僅顯示整數標線。
        axs[0].yaxis.set_major_locator(plt.MaxNLocator(integer = True))
        #斜紋累積補償數的折線圖。
        cumCompensateCounts = [list(c) for c in zip(*cumCompensateCounts)]
        #依序用不同顏色畫出每條鋼纜累積補償數的折線。
        for i, c in enumerate(cumCompensateCounts):
            color = tuple(reversed(Utils.groupColors()[i]))
            color = tuple(c / 255 for c in color)
            axs[1].plot(frames, c, label = i + 1, color = color)
        #在折線上標示其代表第幾條鋼纜。
        labelLines(axs[1].get_lines(), align = False, fontsize = 36)
        #標示折線圖標題。
        axs[1].set_title('compliment', fontsize = 40)
        #標示折線圖X軸名稱。
        axs[1].set_xlabel('frames', fontsize = 36)
        #標示折線圖Y軸名稱。
        axs[1].set_ylabel('amount', fontsize = 36, rotation = 0, labelpad = 40)
        axs[1].set_xticklabels([])
        #設定折線圖Y軸僅顯示整數標線。
        axs[1].yaxis.set_major_locator(plt.MaxNLocator(integer = True))
        #斜紋總數與補償數的表格。
        table = axs[1].table([counts, compensateCounts],
                  rowLabels = ['total', 'compliment'], 
                  colLabels = ['roll{}'.format(i + 1) for i, _ in enumerate(counts)], 
                  bbox = [0.0, -0.8, 1.0, 0.5],
                  )
        #設定折線圖字體大小。
        axs[0].tick_params(axis = 'x', labelsize = 24)
        axs[0].tick_params(axis = 'y', labelsize = 24)
        axs[1].tick_params(axis = 'x', labelsize = 24)
        axs[1].tick_params(axis = 'y', labelsize = 24)
        #設定表格字體大小。
        table.set_fontsize(36)
        #設定圖表間的垂直間距。
        plt.subplots_adjust(hspace = 0.2)

#建立一個應用程式。
app = QtWidgets.QApplication(sys.argv)
#建立介面基底(視窗)。
MainWindow = QtWidgets.QMainWindow()
#設定元件名稱(但本程式與他暫無關連)
MainWindow.setObjectName("MainWindow")
#視窗標題。
MainWindow.setWindowTitle("斜紋計數器")
#視窗大小，寬1650，高1300。
MainWindow.resize(1650,1300)

#在載入影片後顯示的影片名稱。
videoName = QtWidgets.QLabel(MainWindow)
#影片名稱標籤顯示位置：標籤左上角在視窗(1400,170)的位置上，寬200，高50。
videoName.setGeometry(1400,170,200,50)
#在載入影片後顯示影片的區域。
videoArea = QtWidgets.QLabel(MainWindow)
#影片顯示位置：影片左上角在視窗(0,0)的位置上，預設寬1000，高1000。
videoArea.setGeometry(0,0,1000,1000)
#偵測影片是上行或下行的結果並以文字顯示。
videoTypeResult = QtWidgets.QLabel(MainWindow)
#顯示的標籤左上角在視窗(25,1050)的位置上，寬500，高100。
videoTypeResult.setGeometry(25,1050,500,100)
#標籤字體用Arial，字大小為15。
videoTypeResult.setFont(QFont("Arial",15))
#預設此標籤內容為 "尚未偵測"。
videoTypeResult.setText("尚未偵測")

#先給一個檔名預設值。
filename = ""
#載入影片(其實是選取檔案路徑)。
def openFile():
    #將檔案名稱同步給其他函數使用。
    global filename
    #取得檔案名稱。
    filename, filetype = QtWidgets.QFileDialog.getOpenFileName()
    #如果檔案明並非空值:
    if filename:
        #將影片檔名填到顯示影片名稱的標籤。
        videoName.setText(os.path.basename(filename))
        #取得影片第一個frame放在UI上。
        getFrame()

#調整基準線的滑道。    
border_controller = QtWidgets.QSlider(MainWindow)
#滑道左上角在視窗(1200,40)的位置上，寬30(影響粗細)，高600(影響長度)。
border_controller.setGeometry(1200,40,30,600)
#設定滑桿值的範圍，0~100。
border_controller.setRange(0,100)
#設定滑桿一開始在35的位置(0在下，100在上，所以反過來，100-35 = 65)。
border_controller.setValue(65)
#設滑道為垂直的。
border_controller.setOrientation(2)
#設滑道上的刻度樣式。
border_controller.setTickPosition(3)
#設滑道上刻度間隔。
border_controller.setTickInterval(5)

#調整時的紅色基準線(用標籤製作)。
red_border = QtWidgets.QLabel(MainWindow)
#紅色基準線左上角在視窗(100,20)的位置上。
red_border.setGeometry(100,20,200,200)

#取得路徑後，抓第一幀作為預覽(與openFile連動)。
def getFrame():
    global filename,pic_h,pic_w,_y,cav,frame_pic
    #讀取影片。
    video = cv2.VideoCapture((filename))
    #如果影片有選取到:
    if filename:
        #讀取第一幀。
        ret, frame_pic = video.read()
        #取得影片的高度、寬度、通道數。
        pic_h,pic_w,channel = frame_pic.shape
        #將 第一幀 轉換成可以放在視窗上的圖片格式。
        pix = pic_w*channel
        frame_show = QImage(frame_pic,pic_w,pic_h,pix,QImage.Format_RGB888)
        cav = QPixmap(pic_w,pic_h).fromImage(frame_show)
        #顯示"播放影片"的按鈕。
        videoPlayer.setGeometry(QtCore.QRect(1400,270,200,60)) #left_x,top_y,width,height
        #將 顯示影片的區域 與 第一幀 同高、同寬。
        videoArea.setGeometry(QtCore.QRect(0,0,pic_w,pic_h))
        #將 調整標準線的滑道 與影片同高。
        border_controller.setGeometry(QtCore.QRect(1100,0,15,pic_h))
        #將轉好格式的第一幀放在視窗上。
        videoArea.setPixmap(cav)
        #將 影片播放前調整用的基準線 設為紅色。
        red_border.setStyleSheet('''
                        color:#f00;
                        background:#f00;
                        ''')
        #計算初始放在影片高%35的位置(由上往下，最上面是0)的pixel值。
        h__ = int(videoArea.geometry().height())
        temp_y = int(((100-border_controller.value())/100)*h__)
        #將紅色基準線預設放置在影片位置高%35處(由上往下，最上面是0)。
        red_border.setGeometry(QtCore.QRect(0,temp_y,1050,5))
#讓滑桿的數值連動getFrame函數，讓getFrame函數中紅色基準線與滑桿同步。
border_controller.valueChanged.connect(getFrame)

#連動滑桿的值，控制opencv圖片的黑色基準線。
def changeBorderLineY():
    global frame_pic,_y
    _y = border_controller.value()
    border_y =int(round((1-(_y/100)) * frame_pic.shape[0]))
    return border_y

#關閉opencv跟pyqt5的所有運行。
def CloseAllWin():
    cv2.destroyAllWindows()
    QtWidgets.QApplication.closeAllWindows()

#關閉視窗的按鈕。
CloseWindow = QtWidgets.QPushButton(MainWindow)
#按鈕左上角在(1400,470)處，寬200，高60。
CloseWindow.setGeometry(QtCore.QRect(1400,470,200,60)) 
#設定元件名稱(但本程式與他暫無關連)。
CloseWindow.setObjectName("closeWindow")
#設定按鈕上的文字為"關閉視窗"。
CloseWindow.setText("關閉視窗")
#將按鈕功能設定為CloseAllWin設定的 關閉視窗 方法。
CloseWindow.clicked.connect(CloseAllWin)

#載入影片(讀取路徑)的按鈕。
openVideo = QtWidgets.QPushButton(MainWindow)
#按鈕左上角在(1400,70)處，寬200，高60。
openVideo.setGeometry(QtCore.QRect(1400,70,200,60)) 
#設定元件名稱(但本程式與他暫無關連)。
openVideo.setObjectName("pushButton")
#設定按鈕上的文字為"匯入影片"。
openVideo.setText("匯入影片")
#將按鈕功能設定為openFile設定的關閉視窗方法。
openVideo.clicked.connect(openFile)

#統一視窗上，計數標籤的y。
count_val_y = 1050
#設定視窗上每一個計數標籤的x。
xLs = [200+150*i for i in range(8)]
#計數標籤上，字的大小統一為15。
text_size = 15
#視窗上的上行計數，預計最多有8個數字，有幾個計數就會用到幾個。
Line_1 = QtWidgets.QLabel(MainWindow)
Line_1.setGeometry(xLs[0],count_val_y,100,100)
Line_1.setText("0")
Line_1.setFont(QFont("Arial",text_size))

Line_2 = QtWidgets.QLabel(MainWindow)
Line_2.setGeometry(xLs[1],count_val_y,100,100)
Line_2.setText("0")
Line_2.setFont(QFont("Arial",text_size))

Line_3 = QtWidgets.QLabel(MainWindow)
Line_3.setGeometry(xLs[2],count_val_y,100,100)
Line_3.setText("0")
Line_3.setFont(QFont("Arial",text_size))

Line_4 = QtWidgets.QLabel(MainWindow)
Line_4.setGeometry(xLs[3],count_val_y,100,100)
Line_4.setText("0")
Line_4.setFont(QFont("Arial",text_size))

Line_5 = QtWidgets.QLabel(MainWindow)
Line_5.setGeometry(xLs[4],count_val_y,100,100)
Line_5.setText("0")
Line_5.setFont(QFont("Arial",text_size))

Line_6 = QtWidgets.QLabel(MainWindow)
Line_6.setGeometry(xLs[5],count_val_y,100,100)
Line_6.setText("0")
Line_6.setFont(QFont("Arial",text_size))

Line_7 = QtWidgets.QLabel(MainWindow)
Line_7.setGeometry(xLs[6],count_val_y,100,100)
Line_7.setText("0")
Line_7.setFont(QFont("Arial",text_size))

Line_8 = QtWidgets.QLabel(MainWindow)
Line_8.setGeometry(xLs[7],count_val_y,100,100)
Line_8.setText("0")
Line_8.setFont(QFont("Arial",text_size))

#視窗的計數的標籤，用一個List裝載，之後方便用迴圈取出。
Ls = [Line_1,Line_2,Line_3,Line_4,Line_5,Line_6,Line_7,Line_8]

#控制opencv的運行開關。
ocv = True
def defineHowtoClose(self):
    global ocv
    ocv = False
MainWindow.closeEvent = defineHowtoClose

#控制是否停止播放影片。
#一開始先設為不停止。
stop_video = False 
#一旦執行此函數，馬上停止播放影片。
def stopPlaying():
    global stop_video
    stop_video = True

#判斷上下行。
def _isUpward(fn,borderY,pic_h):
    #建立一個方向判斷器。
    dire = Direction(borderY,pic_h)
    #讀取第1幀。
    v = cv2.VideoCapture(fn)
    _, f = v.read()
    #讀取第2幀。
    v_after = cv2.VideoCapture(fn)
    v_after.set(cv2.CAP_PROP_POS_FRAMES,2)
    _, f_after = v_after.read()
    #回傳判斷結果。
    return dire.decide(f,f_after)

#播放影片。
def PlayVideo():
    global ocv,_y,filename,frame,pic_h,pic_w,borderY,stop_video
    
    #一開始先設為不停止影片播放。
    stop_video = False
    
    #基準線高度由滑桿決定。
    borderY = changeBorderLineY()
    #方向由兩幀比較後決定。
    upward = _isUpward(filename,borderY,pic_h)
    #將 顯示方向判斷結果的標籤 寫上 上行或下行。
    if upward:
        videoTypeResult.setText("上行")
    else:
        videoTypeResult.setText("下行")
    
    #讀入影片，將路徑改為影片所在路徑。
    video = cv2.VideoCapture(filename) 
    #斜紋偵測器。
    detector = None
    #斜紋追蹤器。
    tracker = None
    #斜紋計數器。
    counter = None
    #界線的位置。
    borderY = None
    #每組的斜紋數量、每組的累積斜紋數量、每組的斜紋補償數量、每組的累積斜紋補償數量。
    counts, cumCounts, compensateCounts, cumCompensateCounts = (None, None, None, None)
    #影片開始播放後，顯示"停止影片"按鈕，不顯示"開始播放"按鈕，不顯示紅色基準線。
    red_border.setStyleSheet('''''')
    videoStoper.setGeometry(QtCore.QRect(1400,370,200,60))
    videoPlayer.setGeometry(QtCore.QRect(1400,270,200,0))
    #儲存結果的資料夾設為與原影片相同夾層資料夾。
    savedir = os.path.dirname(filename)
    Fname = os.path.basename(filename).split(".")[0]+"_result.avi"
    savepath = os.path.join(savedir,Fname)

    #影片存取器設置，此存取器會存取影片為avi檔。
    four_cc = cv2.VideoWriter_fourcc(*"XVID")
    videoWriter = cv2.VideoWriter(savepath,four_cc,20.0,(pic_w,pic_h))
    
    #一幀一幀的讀取影片。
    while ocv:
        ret, frame = video.read()
        #如果影片已結束或是該幀讀取異常，則終止程式。
        if not ret:
            break
        #如果停止影片播放:
        if stop_video:
            #顯示"開始播放"按鈕。
            videoPlayer.setGeometry(QtCore.QRect(1400,270,200,60))
            #不顯示"停止影片"按鈕。
            videoStoper.setGeometry(QtCore.QRect(1400,370,200,0)) #left_x,top_y,width,height
            #顯示第一幀。
            h,w,channel = frame.shape
            pix = w*channel
            frame_show = QImage(frame,w,h,pix,QImage.Format_RGB888)
            cav = QPixmap(w,h).fromImage(frame_show)
            videoArea.setPixmap(cav)
            #顯示紅色基準線。
            h__ = int(videoArea.geometry().height())
            temp_y = int(((100-border_controller.value())/100)*h__)
            red_border.setStyleSheet('''background:#f00''')
            red_border.setGeometry(0,temp_y,1050,5)
            break 
        
        #基準線高度由滑桿決定。
        borderY = changeBorderLineY()
        #初始化斜紋偵測器。
        if detector is None:
            detector = LineDetector(borderY,upward)
        #獲得斜紋偵測結果。
        detectResult = detector.detect(frame)
        #如果能正常偵測，則進行後續的追蹤與計數，否則跳過這一幀。
        if detectResult is not None:
            #獲得斜紋、斜率、角度。
            lines, slopes, angles = detectResult
            #初始化斜紋追蹤器。
            if tracker is None:
                tracker = LineTracker(len(lines), changeBorderLineY(),upward)
            #初始化斜紋計數器。
            if counter is None:
                counter = LineCounter(len(lines), borderY,upward) 
            #獲得斜紋追蹤器正在追蹤的斜紋。
            lines = tracker.track(lines)
            #透過斜紋計數器計算每組的斜紋數量、每組的累積斜紋數量、每組的斜紋補償數量、每組的累積斜紋補償數量。
            counts, cumCounts, compensateCounts, cumCompensateCounts = counter.count(lines) 
            #分別用不同顏色標記每條鋼纜上的斜紋。
            for index, group in enumerate(lines):   
                for _, line in group.items():
                    cv2.line(frame, line[0], line[1], Utils.groupColors()[index], 1)
            #畫出界線。
            cv2.line(frame, (0, borderY), (frame.shape[1], borderY), (0, 0, 0), 2)
            #標示影片時間軸。
            h, m, s = Utils.milliseconds2HMS(video.get(cv2.CAP_PROP_POS_MSEC))
            cv2.putText(frame, '{}:{}:{}'.format(h, m, s), (10 , frame.shape[0] - 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            #標示每條鋼纜的斜紋數量。
            counts_x = [10 + 80 * (i%4) for i,item in enumerate(counts)]
            counts_y = [frame.shape[0] - 450 if i<4 else frame.shape[0] - 410 for i,item in enumerate(counts)]
            for index, c in enumerate(counts):
                cv2.putText(frame, str(c), (counts_x[index], counts_y[index]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, Utils.groupColors()[index], 2)
                #視窗上的計數標籤同步顯示斜紋數。
                Ls[index].setText(f"{c}")
                Ls[index].setStyleSheet(f'''color:rgb{(Utils.groupColors()[index])};''')
            #將每一幀調整成可以放在視窗上的格式。
            h,w,channel = frame.shape
            pix = w*channel
            frame_show = QImage(frame,w,h,pix,QImage.Format_RGB888)
            cav = QPixmap(w,h).fromImage(frame_show)
            videoArea.setPixmap(cav)
        #影片存取器寫入每一幀影像。
        videoWriter.write(frame)
        #這是影片播放的必要存在程式，但實際上按下q鍵影片不會暫停。
        if cv2.waitKey(1)==ord("q"):
            break 
    #將影片讀取的資訊放掉。
    video.release()
    #將影片存取器的資訊放掉，同時存取結果影片。
    videoWriter.release()
    #如果影片沒有暫停在這一幀，更新圖表。
    if not stop_video:
        #初始化圖表繪圖器。
        grapher = Grapher()
        #畫出斜紋累積總數、斜紋累積補償數的折線圖與斜紋總數、斜紋補償數的表格。
        grapher.plotCountsGraphAndTable(counts, compensateCounts, cumCounts, cumCompensateCounts)
 
#播放影片的按鈕。
videoPlayer = QtWidgets.QPushButton(MainWindow)
#按鈕左上角在(1400,270)處，但一開始不顯示。
videoPlayer.setGeometry(QtCore.QRect(1400,270,200,0))
#設定元件名稱(但本程式與他暫無關連)。
videoPlayer.setObjectName("playVideo")
#設定按鈕上的文字為"開始播放"。
videoPlayer.setText("開始播放")
#將按鈕功能設定為PlayVideo設定的影片播放功能。
videoPlayer.clicked.connect(PlayVideo)

#停止影片的按鈕。
videoStoper = QtWidgets.QPushButton(MainWindow)
#按鈕左上角在(1400,370)處，但一開始不顯示。
videoStoper.setGeometry(QtCore.QRect(1400,370,200,0))
#設定元件名稱(但本程式與他暫無關連)。
videoStoper.setObjectName("stopVideo")
#設定按鈕上的文字為"停止播放"。
videoStoper.setText("停止播放")
#將按鈕功能設定為stopPlaying設定的影片停止播放功能。
videoStoper.clicked.connect(stopPlaying)

#顯示視窗。
MainWindow.show()
#app的關閉(在視窗被關掉後)。
sys.exit(app.exec_())