import cv2
import copy
BLACK = (0,0,0)
GREEN = (0,155,0)
class outputImageGenerator:


    def __init__(self):
        self.baseimg = cv2.imread('./images/sudokuBase.png')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        s1 = len(self.baseimg)
        s2 = len(self.baseimg[0])
        self.lineStep = int(s1/9)
        self.columnStep = int(s2/9)
        self.outputFolder = './outputs/'




    def generate(self, solved, unsolved):
        out = copy.copy(self.baseimg)

        for l in range(9):
            for c in range(9):
                if(unsolved[l][c] != 0):
                    cv2.putText(out,str(unsolved[l][c]),(c * self.columnStep + 10 , l *self.lineStep + 40), self.font, 1.2,BLACK,3)
                else:
                    cv2.putText(out,str(solved[l][c]),(c * self.columnStep + 10 , l *self.lineStep + 40), self.font, 1.2,GREEN,3)


        cv2.imwrite(self.outputFolder+"out.jpg", out)
        cv2.imshow('out',out)
        cv2.waitKey()

