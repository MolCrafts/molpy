# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-22
# version: 0.0.1

from molpy.base import Item
from molpy.group import Group
from molpy.bond import Bond

class ForceField(Item):
    
    def __init__(self, name) -> None:
        super().__init__(name)
        self.templates = {}
        
    def registerTemplate(self, group: Group):
        self.templates[group] = group.properties
        
    def getTemplateByName(self, groupName: str):
        for t in self.templates:
            if t.name == groupName:
                return t
            
    def matchTemplate(self, group: Group):
        """TODO: True template matching

        Args:
            group (Group): group to be matched

        Returns:
            Group: matched template
        """
        for template in self.templates:
            if template.type == group.type:
                return template
        
    def matchGroupOfBonds(self, group: Group, template: Group):
        groupBonds = sorted(group.getBonds())
        tempBonds = sorted(template.getBonds())
        
        for g, t in zip(groupBonds, tempBonds):
            g.update(**t.properties)
            
    def matchBond(self, bond: Bond, template: Group):
        for tarbond in template.getBonds():
            if tarbond.type == bond.type:
                bond.update(**tarbond.properties)
        return bond