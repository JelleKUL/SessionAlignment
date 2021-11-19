class BestResult:
        def __init__(self,testImage, refImage, transMatrix, matchScore):
            self.testImage = testImage
            self.refImage = refImage
            self.transMatrix = transMatrix
            self.matchScore = matchScore