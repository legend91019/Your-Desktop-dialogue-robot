import datetime

def get_current_time_str():
    """
    获取当前现实世界绝对时间并转化为字符串,
    格式:2026-05-16 20:51 (星期六)
    """
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    now = datetime.datetime.now()
    
    return f"{now.strftime('%Y-%m-%d %H:%M')} ({weekdays[now.weekday()]})"


    