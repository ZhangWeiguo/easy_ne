# -*- encoding:utf-8 -*-

import configparser,os
import xml.etree.ElementTree as ET

'''
IniConfiger 能从ini配置文件读取内容/修改内容
XmlConfiger 能从xml配置文件/字符串读取/修改内容
'''

class IniConfiger:
    def __init__(self, config_filename):
        self.config_path = config_filename
        self.configer = configparser.ConfigParser()
        if os.path.exists(config_filename):
            self.configer.read(config_filename)
        else:
            raise Exception("Config File Is Not Exists!")

    def get_sections(self):
        # 获取所有section的名字
        return self.configer.sections()

    def get_keys(self, section):
        # 获取对应section的所有key
        return self.configer.options(section)

    def get(self, section, key, kind=None):
        # 获取对应section的key-value
        #  kind = "int"/"float"/None
        if kind == "int":
            return self.configer.getint(section, key)
        elif kind == "float":
            return self.configer.getfloat(section, key)
        else:
            return self.configer.get(section, key)

    def set(self, section, key, value):
        # 更新对应section的key-value
        if section in self.get_sections():
            self.configer.set(section,key, value)
        else:
            self.configer.add_section(section)
            self.configer.set(section,key, value)
    
    def remove_section(self, section):
        self.configer.remove_section(section)
    def remove_key(self, section, key):
        self.configer.remove_option(section, key)


    def flush(self,config_filename=""):
        # 用当前配置更新配置文件内容
        if config_filename == "":
            self.configer.write(open(self.config_path,'w'))
        else:
            self.configer.write(open(config_filename,'w'))
    

class XmlConfiger:
    def __init__(self,config_filename="", config_string=""):
        self.config_path = config_filename
        self.config_string = config_string
        if os.path.exists(config_filename):
            try:
                self.tree = ET.parse(self.config_path)
                self.root = self.tree.getroot()
            except:
                try:
                    self.root = ET.fromstring(self.config_string)
                except:
                    raise Exception("Config Data Is Not Exists!")
        else:
            raise Exception("Config File Is Not Exists!")
    
    def __recursion(self,root):
        data = {
            "text"  :   root.text,
            "attrib":   root.attrib,
            "tag"   :   root.tag,
            "data"  :   []
            }
        if len(root) == 0:
            return data
        else:
            for unit in root:
                data["data"].append(self.__recursion(unit))
            return data

    def __recursion_inv(self, data, parent):
        nodedata = data["data"]
        if len(nodedata) == 0:
            return None
        else:
            for unit in nodedata:
                node = ET.SubElement(parent, 
                                    tag = unit["tag"], 
                                    attrib = unit["attrib"], 
                                    text = unit["text"])
                self.__recursion_inv(unit, node)
            return None

    
    def find_all(self, tag=None):
        data = []
        for unit in self.root.findall(tag):
            data = self.__recursion(unit)
        return data


    def get_data(self):
        data = self.__recursion(self.root)
        return data
    

    def set_data(self, data):
        if isinstance(data, dict):
            root = ET.Element(
                                tag=self.root.tag,
                                attrib = data["attrib"],
                                text = data["text"] )
            self.__recursion_inv(data, root)
            return ET.ElementTree(root)
        else:
            raise Exception("Data Should Be A Dict")

    def flush(self, tree = None, config_filename=""):
        tree = ET.ElementTree(self.root)
        if config_filename == "":
            tree.write(self.config_path)
        else:
            tree.write(config_filename)


    


if __name__ == "__main__":
    # configer = IniConfiger("test.ini")
    # print configer.get_sections()
    # print configer.get_keys("app1")
    # print configer.get("app1","name")
    # configer.set("app2","name","app2_test")
    # configer.set("app2","port",1234)
    # configer.set("app2","ip","127.0.0.1")
    # configer.remove_key("app2","name")
    # configer.remove_section("app1")
    # configer.flush()

    configer = XmlConfiger(config_filename="test.xml")
    data = configer.get_data()
    data1 = {
            "text"  :   "test_text",
            "attrib":   {"test_attrib":"test"},
            "tag"   :   "test_tag",
            "data"  :   []
            }
    data["data"].append(data1)
    tree = configer.set_data(data)
    configer.root = tree.getroot()
    configer.flush("test_copy.xml")