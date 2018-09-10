def KMP_Match_2(s, t):
    slen = len(s)
    tlen = len(t)
    if slen >= tlen:
        i = 0
        j = 0
        next_list = [-2 for i in range(len(t))]
        getNext_2(t, next_list)
        #print next_list
        while i < slen:
            if j == -1 or s[i] == t[j]:
                i = i + 1
                j = j + 1
            else:
                j = next_list[j]
            if j == tlen:
                return i - tlen
    return -1

def getNext_2(t, next_list):
    next_list[0] = -1
    next_list[1] = 0
    for i in range(2, len(t)):
        tmp = i -1
        for j in range(tmp, 0, -1):
            if equals(t, i, j):
                next_list[i] = j
                break
            next_list[i] = 0

def equals(s, i, j):
    k = 0
    m = i - j
    while k <= j - 1 and m <= i - 1:
        if s[k] == s[m]:
            k = k + 1
            m = m + 1
        else:
            return False
    return True

a = 'singer_周杰|周杰伦|刘德华|王力宏;song_冰雨|北京欢迎你|七里香;actor_周杰伦|孙俪'
b = '请播放周杰伦的七里香给我听'

items = a.split(';')
d = {}
for item in items:
    name = item.split('_')[0]
    values = item.split('_')[1].split('|')
    for v in values:
        if v not in d.keys():
            d[v] = [name]
        else:
            d[v].append(name)




for key in d.keys():
    if KMP_Match_2(b, key):
        print(key)