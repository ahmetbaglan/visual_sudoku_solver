from Extractor import *
from solveIt import *
from outImageGenerator import outputImageGenerator
import copy
from compareSudoku import comparer

def solveFromImage(address):
    extractor = Extractor()
    unsolved = extractor.extract(address)
    solver = sudokuSolver()
    solved = copy.deepcopy(unsolved)
    k = solver.solveSudoku(solved)

    if k == True:
        generator = outputImageGenerator()
        generator.generate(solved,unsolved)
    else:
        print 'could not solve'
        print unsolved

def solveUsingWebcam():

    extractor = Extractor()
    cap = cv2.VideoCapture(0)
    solver = sudokuSolver()
    comp = comparer()
    solvedList = []
    generator = outputImageGenerator()
    frId = 0
    m = 20
    while(True):
        frId += 1
        frId = frId%m
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('lol',frame)

        if(frId == 0):


            try:
                nowUnsolved = extractor.extractForVideo(frame)
                nowSolved = copy.deepcopy(nowUnsolved)
                k = solver.solveSudoku(nowSolved)

                if k == True:
                    print 'found someting'
                    print(nowSolved)
                    generator.generate(nowSolved,nowUnsolved)

                    if(not comp.isSudokuInList(solvedList,nowSolved)):
                        generator.generate(nowSolved,nowUnsolved)
                        solvedList.append(nowSolved)
                print 'tried but could not'
                # print nowUnsolved
            except:
                print 'failed'
                pass


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



