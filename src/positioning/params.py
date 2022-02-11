"""All the weight parameters that affect the pose voting"""

#File names
JSON_ID = "SessionData.json"
IMG_EXTENSION = [".jpg", ".png"]
MESH_EXTENSION = [".obj",".fbx" ]
PCD_EXTENSION = [".pcd", ".pts", ".ply"]

#2D Matching parameters

MAX_2D_FEATURES = 20000
MAX_2D_MATCHES = 1000
MAX_ERROR_2D = 50

MATCHES_2D = 1
ERROR_2D = 1

TOTAL_2D = 1

#3D Matching parameters

MAX_3D_FEATURES = 20000
MAX_3D_MATCHES = 1000
MAX_ERROR_3D = 50

MATCHES_3D = 1
ERROR_3D = 1


TOTAL_3D = 1

#Session parameters

SENSOR_TYPE = 1
SESSION_DATE = 1

#method parameters

METHOD = 1 #the generic method weight
LEAST_DISTANCE = 1
INCREMENTAL = 1
RAYCASTING = 1
