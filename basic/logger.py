# -*- encoding:utf-8 -*-
import logging,time,sys,os,platform
from logging.handlers import TimedRotatingFileHandler,RotatingFileHandler

'''
使用的时候只能使用单进程写，可以使用消息队列读取，避免写冲突
logger = logClient("Test","test", rotate = "Time", when = 'H', keep_num = 48)
logger = logClient("Test","test", rotate = "Size", max_bytes = 1028, keep_num = 48)
logger = logClient("Test","test", rotate = "None")
'''




def Logger(app_name, 
            file_name, 
            rotate       =   "None", 
            when         =   'H', 
            keep_num     =   24, 
            max_bytes    =   1024*1024*10,
            max_buffer   =   100 ):
    logger = logging.getLogger(app_name)
    formater = logging.Formatter(
        fmt         = "%(asctime)s %(filename)10s[line:%(lineno)5d] %(levelname)-8s %(message)s",
        datefmt     = "%Y-%m-%d %H:%M:%S")
    
    if rotate == 'Time':
        file_handler = TimedRotatingFileHandler(file_name, 
                                                when        =   when, 
                                                interval    =   1, 
                                                backupCount =   keep_num)
        if when == 'H':
            file_handler.suffix = "%Y%m%d%H.log"
        elif when  == 'M':
            file_handler.suffix = "%Y%m%d%H%M.log"
        elif when  == 'S':
            file_handler.suffix = "%Y%m%d%H%M%S.log"
        elif when  == 'D':
            file_handler.suffix = "%Y%m%d.log"
        else:
            raise Exception("when Must in (S,M,H,D)")
    elif rotate == 'Size':
        file_handler = RotatingFileHandler(filename = file_name, 
                                        maxBytes = max_bytes, 
                                        backupCount = keep_num)
    else:
        file_handler = logging.FileHandler(filename = file_name)

    file_handler.formatter = formater
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger   


class LoggerCustom:
    def __init__(self, app_name, keep_num = 24, buffer_num = 100):
        self.buffer_num = buffer_num
        self.buffer = []
        self.app_name = app_name
        self.file_name = "%s.%s.log"
        if 'Windows' in platform.system():
            self.is_unix = False
        else:
            import fcntl
            self.is_unix = True
    
    def log(self,data):
        num = len(self.buffer)
        if num <= self.buffer_num:
            if data.strip() != "":
                name = sys._getframe().f_back.f_code.co_filename
                line = str(sys._getframe().f_back.f_lineno)
                dt = time.strftime("%Y-%m-%d %H:%M:%S")
                s = "log_time=%s`locate=%s`data=%s\n"%(dt, name+":"+line, data)
                self.buffer.append(s)
        else:
            self.flush()
    
    def flush(self):
        ymdh = time.strftime("%Y%m%d%H")
        file_name = self.file_name%(self.app_name, ymdh)
        f = file(file_name, 'a+')
        if self.is_unix:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.writelines(self.buffer)
            fcntl.flock(f,fcntl.LOCK_UN)
        else:
            # windows 下未找到文件锁方式
            f.writelines(self.buffer)
        f.close()    
        self.buffer = []

            
        
