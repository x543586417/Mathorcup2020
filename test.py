import re
a="018-12-05/2018/3/5"
b="2018/3/5"

day = int((re.split('[-/]', a))[2])
print(day)