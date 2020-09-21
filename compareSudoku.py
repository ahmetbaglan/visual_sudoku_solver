class comparer:

    def isSameSudoku(self, a, b):
        for i in range(9):
            for j in range(9):
                if(a[i][j]!=b[i][j]):
                    return False
        return True

    def isSudokuInList(self, l, a):

        for b in l:
            if(self.isSameSudoku(a, b)):
                return False
        return True