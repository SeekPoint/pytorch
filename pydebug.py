import inspect
def debuginfo(prefix):
    level=0
    #http://stackoverflow.com/q/6810999
    callerframerecord = inspect.stack()[level]
    frame = callerframerecord[0]
    frame_info = inspect.getframeinfo(frame)
    #print frame_info.filename                      # __FILE__
    #print frame_info.function                      # __FUNCTION__
    #print frame_info.lineno                        # __LINE__

    #使用机器名称，pid，项目或者模块名称,文件名，函数名，行号作为key，记录触发次数
    fname = frame_info.filename.split('/')[-1]

    print(frame_info.filename + ' L#: ' + str(frame_info.lineno) + ' f# ' + frame_info.function)
