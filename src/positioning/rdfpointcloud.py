from argparse import Namespace
import open3d
from rdfobject import RdfObject

import rdflib
from rdflib import Graph, plugin
from rdflib.serializer import Serializer #pip install rdflib-jsonld https://pypi.org/project/rdflib-jsonld/
from rdflib import URIRef, BNode, Literal
from rdflib.namespace import CSVW, DC, DCAT, DCTERMS, DOAP, FOAF, ODRL2, ORG, OWL, \
                           PROF, PROV, RDF, RDFS, SDO, SH, SKOS, SOSA, SSN, TIME, \
                           VOID, XMLNS, XSD

class RdfPointCloud(RdfObject):
    """This class contains rdf meta data of a point cloud"""

    id = ""                     # the id/name of the image without it's extension
    pos = (0,0,0)               # the position of the image in sesison space
    rot = (0,0,0,0)             # the rotation quaternion in sesison space
    fov = 0                     # the full vertical field of view of the camera
    path = ""                   # the full path of the image
    cameraMatrix = None         # the 3x3 Intrinsic camera matrix K
    transformationMatrix = None # the 3x4 Extrinsic pose matrix [R T]
    keypoints = None            # the cv2 generated keypoints 
    descriptors = None          # the cv2 generated descriptors
    image = None                # the cv2_image
    accuracy = []

    def set_bounding_box(self):
        #blablabla

        bb = [[0,1], [2,6], [3,5]]
        self.g.add((self.node,Namespace.v4d.pmin, Literal(min(bb))))
        self.g.add((self.node,Namespace.v4d.pmax, Literal(max(bb))))


    
    
#    def add_to_RDF_graph(self,g):
#        """Write the obtained exif data to a json file"""
#
#        
#        channel = self.get_if_exist(self.exif_data, "DateTime")
#        g.add((imageRDF,exif.dateTime, Literal(self.get_if_exist(self.exif_data, "DateTime"))))
#        #imageWidth=self.get_if_exist(self.exif_data,"ExifImageWidth")
#        g.add((imageRDF,exif.imageWidth, Literal(self.get_if_exist(self.exif_data,"ExifImageWidth"))))
#        g.add((imageRDF,exif.imageLength, Literal(self.get_if_exist(self.exif_data,"ExifImageHeight"))))
#        g.add((imageRDF,exif.xResolution, Literal(self.get_if_exist(self.exif_data,"XResolution"))))
#        g.add((imageRDF,exif.yResolution, Literal(self.get_if_exist(self.exif_data,"YResolution"))))
#        g.add((imageRDF,exif.resolutionUnit, Literal(self.get_if_exist(self.exif_data,"ResolutionUnit"))))
#        # 'exif:imageWidth'
#        # 'exif:imageLength'
#        # 'exif:orientation'
#        # 'exif:xResolution'
#        # 'exif:yResolution'        
#        # 'exif:fNumber'
#        # 'exif:exifVersion'
#        # 'exif:apertureValue'
#        # 'exif:focalLength'
#        # 'exif:imageUniqueID'
#        if 'GPSInfo' in self.exif_data:
#            gps_info = self.exif_data["GPSInfo"]
#            g.add((imageRDF,exif.gpsLatitude, Literal(self.get_if_exist(gps_info, "GPSLatitude"))))
#            g.add((imageRDF,exif.gpsLatitudeRef, Literal(self.get_if_exist(gps_info, "GPSLatitudeRef"))))
#            g.add((imageRDF,exif.gpsLongitude, Literal(self.get_if_exist(gps_info, "GPSLongitude"))))
#            g.add((imageRDF,exif.gpsLongitudeRef, Literal(self.get_if_exist(gps_info, "GPSLongitudeRef"))))
#            g.add((imageRDF,exif.gpsAltitude, Literal(self.get_if_exist(gps_info, "GPSAltitude"))))
#            g.add((imageRDF,exif.gpsAltitudeRef, Literal(self.get_if_exist(gps_info, "GPSAltitudeRef"))))
#        # 'exif:gpsVersionID'
#        # 'exif:gpsLatitudeRef'
#        # 'exif:gpsLatitude'
#        # 'exif:gpsTimeStamp'
#        # 'exif:gpsDOP'
#        # 'exif:gpsMapDatum'
#        # 'exif:gpsDifferential'
#        # 'exif:model'
#        # 'exif:resolutionUnit'
#        # 'exif:exif_IFD_Pointer' #everything about camera
#        # 'exif:IFD'
#        return g
    