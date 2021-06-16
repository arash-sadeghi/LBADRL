import numpy as np
import os
import matplotlib.pyplot as plt

def DirLocManage(returnchar=False):
    ''' with this segment code is callable from any folder '''
    if os.name=='nt':
        dirChangeCharacter='\\'
    else:
        dirChangeCharacter='/'
    if returnchar==False:
        scriptLoc=__file__
        for i in range(len(scriptLoc)):
            # if '/' in scriptLoc[-i-2:-i]: # in running
            if dirChangeCharacter in scriptLoc[-i-2:-i]: # in debuging
                scriptLoc=scriptLoc[0:-i-2]
                break
        # print('[+] code path',scriptLoc)
        os.chdir(scriptLoc)
    return dirChangeCharacter
    ''' done '''
DirLocManage()

with open('critic_mem.npy','rb') as f:
    ar=np.load(f)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(ar[:,1],ar[:,2],ar[:,3],marker='o')
ax.set_xlabel('l')
ax.set_ylabel('a')
ax.set_zlabel('r')

plt.show()

print('hi')