import torch as T

def adjOfMat(t):
    t_size=t.size()
    if(list(t_size) == [2,2]):
        adj_t = T.zeros(2,2)
        adj_t[0][1],adj_t[1][0] = -1*t[0][1],-1*t[1][0]
        adj_t[0][0],adj_t[1][1] = t[1][1],t[0][0]
        return adj_t
    elif (list(t_size) == [3,3]):
        adj_t = T.tensor([[1,-1,1],[-1,1,-1],[1,-1,1]])
        t = transOfMat(t)
        for i in [0,1,2]:
            temp = T.zeros(2,2)
            temp = t[0:2,[j for j in [0,1,2] if i!=j ]]
            adj_t[2][i]=adj_t[2][i]*detOfMat(temp)

        for i in [0,1,2]:
            temp = T.zeros(2,2)
            temp = t[1:,[j for j in [0,1,2] if i!=j ]]
            adj_t[0][i]=adj_t[0][i]*detOfMat(temp)

        adj_t[1][0]=adj_t[1][0]*detOfMat(T.tensor([
            [t[0][1],t[0,2]],
            [t[2][1],t[2,2]],
        ]))
        adj_t[1][1]=adj_t[1][1]*detOfMat(T.tensor([
            [t[0][0],t[0,2]],
            [t[2][0],t[2,2]],
        ]))
        adj_t[1][2]=adj_t[1][2]*detOfMat(T.tensor([
            [t[0][0],t[0,1]],
            [t[2][0],t[2,1]],
        ]))
        return adj_t

def transOfMat(t):
    t_size=list(t.size())
    t_transpose = T.zeros(t_size[0],t_size[1])
    for i in range(t_size[0]):
        for j in range(t_size[0]):
            t_transpose[i][j]=t[j][i]
    return t_transpose

def detOfMat(t):
    t_size=t.size()
    if(list(t_size) == [2,2]):
        return t[0][0]*t[1][1] - t[1][0]*t[0][1]
    elif (list(t_size) == [3,3]):
        det_t=T.tensor([0])
        for i in [0,1,2]:
            temp = T.zeros(2,2)
            temp = t[1:,[j for j in [0,1,2] if i!=j ]]
            if(i==1):
                det_t -= t[0][i]*detOfMat(temp)
            else:
                det_t += t[0][i]*detOfMat(temp)
            temp= T.zeros(2,2)
        return det_t

def invOfMat(t):
    adj_t = adjOfMat(t)
    inv_t = T.zeros(3,3)
    det_t = detOfMat(t)
    for i in [0,1,2]:
        for j in [0,1,2]:
            inv_t[i][j] = adj_t[i][j] / det_t
    return inv_t

t1=T.tensor([[2,14,6],
             [8,10,12],
             [14,16,18]])

print(invOfMat(t1),'inv')
