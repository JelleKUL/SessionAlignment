class BestResult:
    testImage = 0
    refImage = 0
    transMatrix = 0
    matchScore = 0

    def __init__(self,testImage, refImage, transMatrix, matchScore):
        self.testImage = testImage
        self.refImage = refImage
        self.transMatrix = transMatrix
        self.matchScore = matchScore