
class ImageTransform:
    id = ""
    pos = (0,0,0)
    rot = (0,0,0,0)
    fov = 0
    path = ""
    sessionDataPath = ""

    def __init__(self, id, pos, rot, fov):
        self.id = id
        self.pos = pos
        self.rot = rot
        self.fov = fov
    

