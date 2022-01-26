# this exist to convert the existing refernece data to rdf compliant data

import open3d
import rdflib
from rdflib import Graph, plugin
from rdflib.serializer import Serializer #pip install rdflib-jsonld https://pypi.org/project/rdflib-jsonld/
from rdflib import URIRef, BNode, Literal
from rdflib.namespace import CSVW, DC, DCAT, DCTERMS, DOAP, FOAF, ODRL2, ORG, OWL, \
                           PROF, PROV, RDF, RDFS, SDO, SH, SKOS, SOSA, SSN, TIME, \
                           VOID, XMLNS, XSD

class RdfObject():
    """This base class contains rdf meta data and methods to import/ export the data to RDF"""

#    g = Graph()
#    node = None
#    
#    
#    def __init__(self, path):
#        
#        if(self.check_RDF_path(path)):
#            g = Graph()
#            g = g.parse(path)
#            self.g = g
#        pass
#
#    def __init__(self, pcd, id):
#
#        self.g = Graph()
#        self.set_context()
#        self.add_resource(pcd, id)
#        pass
#
#    def check_RDF_path(self, pathName):
#        """Checks if the path name is valid"""
#
#        if (not pathName.EndsWith(".ttl") and not pathName.EndsWith(".xml")):
#            print("Invalid file path. Enter a filename with .ttl or .xml extension.")
#            return False
#        if (not pathName.IsValidFileName(True)):
#            print("filename contains invalid character.")
#            return False
#
#        return True
#
#    def set_context(self):
#
#        #bind the various ontologies that you want to use in your schema
#        self.g.bind("rdf", RDF)
#        self.g.bind("rdfs", RDFS)
#        self.g.bind("foaf", FOAF)
#        self.g.bind("owl", OWL)
#        # bind additional ontologies that aren't in rdflib
#        exif = rdflib.Namespace('http://www.w3.org/2003/12/exif/ns')
#        self.g.bind('exif', exif)
#
#    def add_resource(self, pcd, id):
#        
#        self.node = URIRef(id)
#        self.g.add((self.node,RDFS.Resource, Literal(BNode()))) # a GUID is generated
#        pass
#
#    def write_graph(self, path):
#
#        if(self.check_RDF_path(path)):
#            self.g.serialize(destination = path, format='ttl')
#            return True
#        return False


    