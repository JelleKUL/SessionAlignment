class BestResult:
        def __init__(self,testImage, refImage, transImage, transMatrix, matchAmount):
            self.testImage = testImage
            self.refImage = refImage
            self.transImage = transImage
            self.transMatrix = transMatrix
            self.matchAmount = matchAmount