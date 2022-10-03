# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-03-12
# version: 0.0.1

import linecache
import re

class FileHandler:
    
    def __init__(self, filepath, mode='r'):
        
        self.filepath = filepath
        self.fp = open(self.filepath, mode)
        
    def readline(self):
        return self.fp.readline()
    
    def writeline(self, line):
        return self.fp.write(line)
        
    def getline(self, lineno):
        return linecache.getline(self.filepath, lineno)
    
    def readlines(self):
        
        return self.fp.readlines()
    
    def readchunks(self, start, end=None):
        
        chunks = Chunks(self.filepath)
        chunks.scan(start, end)
        
        return chunks
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        line = self.fp.readline()
        if line:
            return line
        else:
            raise StopIteration
        
    def reset_fp(self):
        self.close()
        self.fp = open(self.filepath, 'r')   
        
    def close(self):

        if not self.fp.closed:
              self.fp.close()  
    
    def __del__(self):
        self.close()
    

class Chunks:
    
    def __init__(self, filepath):
        
        self.filepath = filepath
        self.fp = open(self.filepath, 'r')
        self.start_byte = []
        self.end_byte = []
        self.start_lineno = []
        self.end_lineno = []
        self._scan_flag: bool = False
        
    @property
    def nchunks(self):
        return len(self.start_lineno)
    
    @property
    def isScan(self):
        return self._scan_flag
    
    def scan(self, start, end=None):
        
        readline = self.fp.readline
        line = readline()
        lineno = 1
        
        start_byte = self.start_byte
        end_byte = self.end_byte
        start_lineno = self.start_lineno
        end_lineno = self.end_lineno
        
        start_partten = re.compile(start)
        if end:
            end_partten = re.compile(end)

            while line:
                
                if re.match(start_partten, line):
                    start_lineno.append(lineno)
                    start_byte.append(self.fp.tell()-len(line))
                elif re.match(end_partten, line):
                    end_lineno.append(lineno)
                    end_byte.append(self.fp.tell()-len(line))
                    
                line = readline()
                lineno += 1
                    
        else:
            
            while line:
            
                if re.match(start_partten, line):
                    start_lineno.append(lineno)
                    start_byte.append(self.fp.tell()-len(line)) 
                    
                line = readline()
                lineno += 1
            
            self.end_lineno = start_lineno[1:]
            self.end_lineno.append(lineno)
            self.end_byte = start_byte[1:]
            self.end_byte.append(self.fp.tell())
        
        self._scan_flag = True
    
    def __del__(self):
        
        try:
            self.fp.close()
        except:
            pass
            
    def __getitem__(self, index):
        
        if isinstance(index, slice):
            start, end = index.start, index.stop
            if start is None:
                start = 0
            if end is None:
                end = len(self.start_lineno)
            return self.readlines(start, end)
        elif isinstance(index, int):
            
            return self.getChunk(index)
            
    def getchunk(self, index):
        
        self.fp.seek(self.start_byte[index])
        return self.fp.read(self.end_byte[index] - self.start_byte[index] - 1).split('\n')
            