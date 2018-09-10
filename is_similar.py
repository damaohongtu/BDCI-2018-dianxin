def is_similar(A,B):
    tmp="abcdefghijklmnopqrstuvwxyz"
    key=['0']*26
    for i in range(len(A)):
        if key[tmp.find(A[i])]=='0':
            key[tmp.find(A[i])]=B[i]
        else:
            if key[tmp.find(A[i])]!=B[i]:
                return False
    return True

def how_many(S,T):
    result=0
    len_T=len(T)
    len_S=len(S)
    if len_S<len_T:
        return 0
    for i in range(0,len_S-len_T):
        if is_similar(S[i:i+len_T-1],T):
            result=result+1
    return result
print(how_many("ababcb","xyz"))