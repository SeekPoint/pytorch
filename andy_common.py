import inspect
def printLineFileFunc():
    callerframerecord = inspect.stack()[1]    # 0代表当前行  , 1当前调用
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    filename = info.filename[info.filename.rfind('/')+1:]
    print("on func:"+ info.function + " at file:"+ filename + " #" + str(info.lineno))