# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-18
# version: 0.0.1

from molpy.forcefield import Node
import xml.etree.ElementTree as ET
import xml.dom.minidom

class XMLParser:
    def __init__(self, ffTree: Node):

        self.ff = ffTree

    def parse_node(self, root):

        node = Node(tag=root.tag, **root.attrib)
        children = list(map(self.parse_node, root))
        if children:
            node.add_children(children)
        return node

    def parse(self, *xmls):
        for xml in xmls:
            root = ET.parse(xml).getroot()
            for leaf in root:
                n = self.parse_node(leaf)
                ifExist = False
                for nchild, child in enumerate(self.ff.children):
                    if child.tag == n.tag:
                        ifExist = True
                        break
                if ifExist:
                    self.ff.children[nchild].add_children(n.children)  # TODO: update all attrs or just children node?
                else:
                    self.ff.add_child(n)

    def write_node(self, parent, node):
        parent = ET.SubElement(parent, node.tag, node.attrs)
        for sibiling in node:
            tmp = ET.SubElement(parent, sibiling.tag, sibiling.attrs)
            for child in sibiling:
                self.write_node(tmp, child)

    @staticmethod
    def pretty_print(element):
        initstr = ET.tostring(element, "unicode")
        pretxml = xml.dom.minidom.parseString(initstr)
        pretstr = pretxml.toprettyxml()
        return pretstr

    def write(self, path):

        root = ET.Element('Forcefield')

        for child in self.ff:
            if child.tag == 'Residues':
                Residues = ET.SubElement(root, 'Residues')
                for residue in child:
                    self.write_node(Residues, residue)
            else:
                self.write_node(root, child)
        outstr = self.pretty_print(root)
        with open(path, "w") as f:
            f.write(outstr)
