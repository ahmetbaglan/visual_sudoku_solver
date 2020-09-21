import numpy as np
import cv2
from OCR_1 import OCR

class Extractor:


    warped_masked_original_color = None
    warped_masked_uniform_gray = None
    warped_masked_uniform_binary = None

    digit_binary = None
    digit_contours = []

    outPuzzle = np.zeros(((9, 9)), np.uint8)

    ocr = None

    inputColor = None
    toGray = None
    blurredGray = None
    uniGray = None
    uniBlurredGray = None
    thresholdOrg = None
    thresholdUni = None

    contourBiggest = None
    contour_approx = None

    puzzle_actual_mask = None
    def __init__(self):
        self.debug = True
    def extractForVideo(self, frame):
        self.debug = True
        self.inputColor = frame
        self.processImage()
        self.getBiggestContour()
        self.warpIt()
        self.extractDigits()
        self.recognizeDigits()
        return self.outPuzzle.tolist()

    def extract(self, image_filename="./images/sudoku_original.jpg"):
        self.getImage(image_filename)
        self.processImage()
        self.getBiggestContour()
        self.warpIt()
        self.extractDigits()
        self.recognizeDigits()
        self.debug =True
        return self.outPuzzle.tolist()


    def getImage(self, image_filename):
        self.inputColor = cv2.imread(image_filename)
        self.inputColor = cv2.resize(self.inputColor, (600, 600))



    def processImage(self):
        self.toGray = cv2.cvtColor(self.inputColor, cv2.COLOR_BGR2GRAY)
        self.uniGray = self.normalizeBrightness(self.toGray)

        self.blurredGray = cv2.GaussianBlur(self.toGray, (5, 5), 0)
        self.uniBlurredGray = cv2.GaussianBlur(self.uniGray, (5, 5), 0)

        self.thresholdOrg = cv2.adaptiveThreshold(self.blurredGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)
        self.thresholdUni = cv2.adaptiveThreshold(self.uniBlurredGray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)


    def normalizeBrightness(self, img_gray):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        closing_image = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        uniform_image = np.float32(img_gray)/(closing_image)
        uniform_image = np.uint8(cv2.normalize(uniform_image,uniform_image,0,255,cv2.NORM_MINMAX))
        return uniform_image


    def getBiggestContour(self):

        contours, hierarchy = cv2.findContours(self.thresholdOrg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_actual = None
        biggest_approx = None
        max_area = 0
        for i in contours:
            # Alani bul
            area = cv2.contourArea(i)
            if area > 100:
                # Cevresi nasil
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                #En buyuk alan mi
                if area > max_area and len(approx)==4:
                    biggest_approx = approx
                    biggest_actual = i
                    max_area = area

        self.contourBiggest = biggest_actual
        self.contour_approx = biggest_approx

        self.puzzle_actual_mask = np.zeros((self.toGray.shape), np.uint8)
        cv2.drawContours(self.puzzle_actual_mask, [self.contourBiggest], 0, 255, -1)
        cv2.drawContours(self.puzzle_actual_mask, [self.contourBiggest], 0, 0, 2)


    def warpIt(self):
        # Get masked uniform image
        masked_uniform_gray = cv2.bitwise_and(self.uniGray, self.puzzle_actual_mask)

        # Duzelt (sol ust nokta 1. eleman olcak sekilde)
        rectify_contour_approx = self.rectify(self.contour_approx)
        #Resize
        new_coordinates = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)
        retval = cv2.getPerspectiveTransform(rectify_contour_approx, new_coordinates)
        warped_masked_uniform_gray = cv2.warpPerspective(masked_uniform_gray, retval, (450,450))
        warped_masked_original_color = cv2.warpPerspective(self.inputColor, retval, (450, 450))

        _,warped_masked_uniform_binary = cv2.threshold(warped_masked_uniform_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        self.warped_masked_uniform_binary = warped_masked_uniform_binary
        self.warped_masked_uniform_gray = warped_masked_uniform_gray
        self.warped_masked_original_color = warped_masked_original_color

        if(self.debug):
            cv2.imshow('ll',warped_masked_original_color)
            cv2.waitKey()


    def extractDigits(self):
        warped_masked_uniform_gray_inv = 255-self.warped_masked_uniform_binary

        # Flood image
        h, w = warped_masked_uniform_gray_inv.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(warped_masked_uniform_gray_inv, mask, (0,0), 0)
        cv2.floodFill(warped_masked_uniform_gray_inv, mask, (w-1,0), 0)
        cv2.floodFill(warped_masked_uniform_gray_inv, mask, (0,h-1), 0)
        cv2.floodFill(warped_masked_uniform_gray_inv, mask, (w-1,h-1), 0)

        # Get Cont
        contours, hierarchy = cv2.findContours(warped_masked_uniform_gray_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if self.debug:
            cv2.imshow('l',warped_masked_uniform_gray_inv)
            cv2.waitKey()
        digit_contours_overlay_color = self.warped_masked_original_color.copy()
        digit_contours_mask_binary =  np.zeros((warped_masked_uniform_gray_inv.shape),np.uint8)
        digit_binary = None
        digit_contours = []

        for i in contours:
            area = cv2.contourArea(i)

            [_,_,w,h] = cv2.boundingRect(i)
            if area > 50 and w<50 and h<50:#Sayinin boyu 50den kucuk alani buyuk
                cv2.drawContours(digit_contours_overlay_color, [i], 0, (0,255,0), 2)
                cv2.drawContours(digit_contours_overlay_color, [i], 0, (0,255,0), -1)

                cv2.drawContours(digit_contours_mask_binary, [i], 0, 255, 2)
                cv2.drawContours(digit_contours_mask_binary, [i], 0, 255, -1)

                digit_contours.append(i)

        digit_binary = cv2.bitwise_and(digit_contours_mask_binary, warped_masked_uniform_gray_inv) * 255
        if self.debug:
            cv2.imshow('Draw',digit_contours_mask_binary )
            cv2.waitKey()
        self.digit_binary = digit_binary
        self.digit_contours = digit_contours


    def recognizeDigits(self):

        if self.ocr == None:
            self.ocr = OCR()
            self.ocr.loadData()

        digit_binary = self.digit_binary.copy()


        padding = 50
        digit_binary = cv2.copyMakeBorder(digit_binary,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=0)


        for i in self.digit_contours:
            # Merkezden hangi kare oldugunu cek
            M = cv2.moments(i)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cell = (int(cx/50), int(cy/50))

            # Get bounding rectangle
            [x,y,w,h] = cv2.boundingRect(i)

            #Sayiyi cek
            digit = digit_binary[y+padding:y+h+padding,x+padding:x+w+padding]

            # Skip if unable to get image
            if len(digit) == 0:
                continue

            # Set results
            self.outPuzzle[cell[1]][cell[0]] = self.ocr.recognizeCharacter(digit)

        # Draw text on image to verify
        self.showOverlayPuzzle()


    def showOverlayPuzzle(self, puzzle=None, window_name="Recognized puzzle"):
        if puzzle == None:
            puzzle = self.outPuzzle
        warped_masked_original_color = self.warped_masked_original_color.copy()
        for y in range(0,9):
            for x in range(0,9):
                num = puzzle[y][x]
                if num != 0:
                    if self.outPuzzle[y][x] != 0:
                        cv2.putText(warped_masked_original_color,str(num), (x*50+25,y*50+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
                    else:
                        cv2.putText(warped_masked_original_color,str(num), (x*50+25,y*50+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
        



    # Rectify - reshapes and sorts the order of points in a contour.
    def rectify(self, h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)
 
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
         
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
  
        return hnew


#
#   Main Entry Point
#
if __name__ == '__main__':
    Extractor().extract()
    # cv2.waitKey()
