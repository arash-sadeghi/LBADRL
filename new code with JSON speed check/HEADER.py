import os
import time
if os.name=="nt": os.system('cls')
else: os.system('clear')
# print('[+] updating pip')
# os.system('python -m pip install --upgrade pip') # upgrading pip
''' installing dependincies '''
# dependincies=['opencv-python','numpy','matplotlib','termcolor','psutil']
# availble_pkg=os.popen('pip list').read()
# for d in dependincies:
#     if not(d in availble_pkg):
#         ans=input('[!] pakage '+d+' must be install [y/n]')
#         if ans=='y':
#             os.system('pip install '+d)

import sys
import cv2 as cv
import numpy as np
import random as rnd
from math import radians, sin,cos,sqrt,atan2,exp
from time import time as TIME
from time import ctime
import signal # used in main
import pickle # used in main
import matplotlib.pyplot as plt
from psutil import disk_usage
from termcolor import colored 
from itertools import combinations  as comb , product

################################################################################################################################
################################################################################################################################
################################################################################################################################
def set_seed(sd=None):
    if (type(sd) is float) or (type(sd) is int):
        print(colored("\n\n[!]>>>>>>>>>>>> SEED GIVEN. NOT RANDOM<<<<<<<<<<<<<<\n\n","red"))
        rnd.seed(sd)
        return sd
    else:
        seed=int(TIME()%1000)
        print(colored("[+] seed: ","red"),seed)
        rnd.seed(seed)
        return seed

# ------------------------------------------------------------------------------------------------------------------------------
def warningSupress():
    from sys import warnoptions
    from warnings import simplefilter
    if not warnoptions:
        simplefilter("ignore")
# ------------------------------------------------------------------------------------------------------------------------------
def checkHealth():
    MinDiskCap=30
    # health=shutil.disk_usage('/')
    # if health[-1]/(2**30)<=5:
    #     raise NameError('[-] disk is getting full')
    # disk='/media/arash/7bee828d-5758-452e-956d-aca000de2c81'
    disk=os.getcwd()

    try:
        hdd=disk_usage(disk)
        total,used,free=hdd.total / (2**30),hdd.used / (2**30),hdd.free / (2**30)
        if free<MinDiskCap:
                print (colored('[-] disk is almost full',"red"))
                exit(1)

    except Exception as E:
        print(colored("[+] Error not in "+disk+" ERROR:"+E,'red'))

    s=__file__
    s=s.replace(__name__,"")
    s=s.replace('.py',"")
    if '\\' in s:
        s=s.replace('\\','/')
    hdd2=disk_usage(s)
    total2,used2,free2=hdd2.total / (2**30),hdd2.used / (2**30),hdd2.free / (2**30)

    if free2<MinDiskCap:
            print (colored('[-] disk is almost full',"red"))
            exit(1)
    print(colored('\t[+] Disk health checked. free2: ','green'),str(int(free2)),' GB')
# ------------------------------------------------------------------------------------------------------------------------------
def m2px(inp):
    return int(inp*512/2)
# ------------------------------------------------------------------------------------------------------------------------------
def px2m(inp):
    return inp*2/512
# ------------------------------------------------------------------------------------------------------------------------------
def RotStandard(inp):
    if inp<0: inp+=360
    if inp>360: inp-=360
    return inp
# ------------------------------------------------------------------------------------------------------------------------------
def dist(delta):
    if np.size(delta)>2:
        # return np.sqrt(np.square(delta[:,0])+np.square(delta[:,1]))
        return np.linalg.norm(delta,axis=1)
    else:
        return np.linalg.norm(delta)
        # return np.sqrt(np.square(delta[0])+np.square(delta[1]))
# ------------------------------------------------------------------------------------------------------------------------------
def TableCompare(table1,table2):
    return np.round((table1+table2)/2,3)
# ------------------------------------------------------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------------------------------------------------------
def sigmoid(x):
    return 1/(1+exp(-10*x+5))
# ------------------------------------------------------------------------------------------------------------------------------
def saturate(x):
    if x>=1: return 1
    elif x<=0: return 0
    else: return x
# ------------------------------------------------------------------------------------------------------------------------------
def quadratic(x):
    return saturate(x**2)
# ------------------------------------------------------------------------------------------------------------------------------
################################################################################################################################
################################################################################################################################
################################################################################################################################
class SUPERVISOR:
    def __init__(self,ROBN,codeBeginTime,showFrames,globalQ,record,Lx,Ly,cueRadius,visibleRaduis,\
        paramReductionMethod,PRMparameter,noise,localMinima,method):

        self.Etol=4
        self.Lx=m2px(Lx)
        self.Ly=m2px(Ly)
        self.cueRadius=m2px(cueRadius)
        self.visibleRaduis=m2px(visibleRaduis)
        self.QRloc={'QR1':(self.Lx//2,0),'QR2':(self.Lx,self.Ly//4),\
            'QR3':(self.Lx,self.Ly//4*3),'QR4':(self.Lx//2,self.Ly),'QR5':(0,3*self.Ly//4),'QR6':(0,self.Ly//4)}
        '''self.localMinima: flag to determine if local minima will exist or not '''
        self.localMinima=localMinima
        self.ground=self.generateBackground(self.Lx,self.Ly,self.cueRadius,self.visibleRaduis)
        self.codeBeginTime=codeBeginTime
        self.sharedParams()
        self.showFrames=showFrames
        self.record=record
        self.vizFlag=record or showFrames
        self.fps=int(self.timeStep*100*2)
        self.ROBN=ROBN
        self.collisionDist=m2px(0.05)
        self.swarm=[0]*self.ROBN
        self.wallNum=4
        self.time=0
        self.flagsR=np.zeros((self.ROBN,self.ROBN))
        self.counterR=np.zeros((self.ROBN,self.ROBN))
        self.collisionDelay=1
        self.detectRad=0.06
        self.robotRad=0.06
        self.robotSenseRad=m2px(self.robotRad+self.detectRad)
        self.collisionDetectDist=self.robotSenseRad+self.collisionDist
        self.Wmax=120 if not self.debug else 0
        self.paramReductionMethod=paramReductionMethod
        if self.debug: print('\t[+] Debugging mode: Wmax=0')
        self.log=0
        videoRecordTime=ctime(TIME()).replace(':','_')
        FPS=5
        '''
        size=(self.Xlen,self.Ylen)
        this wont work you must inverse it. an empty video will be saved
        '''
        size=(self.Ylen,self.Xlen)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        if self.vizFlag:
            self.video = cv.VideoWriter(codeBeginTime+DirLocManage(returnchar=True)+videoRecordTime+'.mp4',fourcc, FPS, size,True)

        self.allRobotIndx=np.arange(0,self.ROBN)
        self.globalQ=globalQ
        self.allnodes=np.array(list(comb(self.allRobotIndx,2)))
        self.colliders=[]
        self.all_poses=[]
        self.allRobotQRs=np.array(list(product(self.allRobotIndx,np.arange(0,len(self.QRloc)))))
        self.QRpos_ar=np.array(list(self.QRloc.values()))
        # self.NASfunction=np.vectorize(lambda x: x.groundSensorValue)
        self.NASfunction=[]


        '''for local and global NAS '''
        if self.localMinima:
            self.NASGfunction=np.vectorize(lambda x: x.groundSensorValueG)
            self.NASLfunction=np.vectorize(lambda x: x.groundSensorValueL)

        if self.showFrames:
            ''' positioning window to make it easier to watch '''
            cv.namedWindow('background')
            cv.moveWindow('background',1000,0)
        self.noise=noise
        # self.sigma={"angle":int(180*0.25),"length":int(self.maxlen//4*0.25)}
        noise_ratio=self.noise/180
        self.sigma={"angle":int(180*noise_ratio),"length":int(self.maxlen//4*noise_ratio*0)}

        self.PRMparameter=PRMparameter
        self.method=method
# sharedParams .................................................................................................................
    def sharedParams(self):
        """
        These parameters are constant
        Other wise it will cause conflict between
        supervisor and robots
        """
        self.Xlen=np.shape(self.ground)[0]  
        self.Ylen=np.shape(self.ground)[1]
        ''' places of Xlen and Ylen must been changed but dont 
        touch it. now Xlen=1024>Ylen=512 but is should have been 
        other way. any way! '''  

        self.EpsilonDampRatio=0.999 #######

        '''self.desiredPosDistpx the radius of a circle in which I say desired point has reached 
        This parameter depends on the pixel distance that each robot travels in each cycle.'''
        self.desiredPosDistpx=10


        # angles=np.arange(0,180+18,18)
        angles=np.array([35,70,70+35])
        # self.maxlen=int(16*512/9.2)*sqrt(2) # caviat
        self.maxlen=int(1*512/2)

        lens=[self.maxlen//2,2*self.maxlen//2]
        self.actionSpace=list(product(lens,angles))

        self.numberOfStates=7
        self.NumberOfActions=len(self.actionSpace)
        
        '''self.RLparams:  start value of parameters'''
        self.RLparams={"epsilon":1,"alpha":0.1}

        ''' if supervisor is executing this method, add two parameters to params file'''
        if not hasattr(self,'robotName'):
            with open(self.codeBeginTime+DirLocManage(returnchar=True)+'params.txt','a') as paramfile :
                paramDict={"RLparams":self.RLparams,"EpsilonDampRatio":self.EpsilonDampRatio}
                paramfile.write(str(paramDict))
        else:
            ''' if robot is running this method, get QRlocs from supervisor'''
            self.QRloc=self.SUPERVISOR.QRloc    
        self.velocity=14
        self.timeStep=512*0.001
        ''' printFlag: True: robot0 will print content False: No one will print any shit  '''
        self.printFlag= not True 
        '''self.debug: in debugging mode Wmax will be zero and robot 0 will be indicated with green color in visualize'''
        self.debug=not True ##########
# generateBackground ...........................................................................................................
    def generateBackground(self,Lx,Ly,cueRadius,visibleRaduis):
        Lxpx=Lx
        Lypx=Ly
        R=cueRadius
        def gauss(x):
            a = 1.0 # amplititude of peak
            b = Lxpx/2.0 # center of peak
            c = Lxpx/11# standard deviation
            return a*exp(-((x-b)**2)/(2*(c**2)))
        im=np.zeros((Lxpx,Lypx))
        for i in range(0,R):
            cv.circle(im,(int((Lypx/2)),int(Lxpx/2)),i,gauss(Lxpx/2-i),2)

        if self.localMinima:
            self.cueGcenter=np.flip(np.array([int((Lypx/4)),int(Lxpx/2)]))
            self.cueLcenter=np.flip(np.array([int((3*Lypx/4)),int(Lxpx/2)]))

            for i in range(0,R):
                cv.circle(im,(int((3*Lypx/4)),int(Lxpx/2)),i,gauss(Lxpx/2-i)/2,2)

        ''' until here dim(x)>dim(y). after here it changes '''
        im=cv.rotate(im, cv.ROTATE_90_CLOCKWISE)


        for i in self.QRloc.values():
            cv.circle(im,tuple(i),10,(255,255,255),-1)
            cv.circle(im,tuple(i),visibleRaduis,(255,255,255),1)
        im=255-255*im
        '''writing and reading back the image to have a 3 channel image with pixels between 0-255'''
        cv.imwrite("BackgroundGeneratedBySim.png",im)
        im=cv.imread("BackgroundGeneratedBySim.png")
        im=cv.rectangle(im,(0,0),(Lxpx,Lypx),(0,255,0),3)
        return im
# generateRobots ...............................................................................................................
    def generateRobots(self):
        margin=self.collisionDetectDist*2
        possibleY=np.arange(0+margin,self.Xlen-margin,margin)
        possibleX=np.arange(0+margin,self.Ylen-margin,margin)
        possiblePos=list(product(possibleX,possibleY))
        possibleRot=np.linspace(0,360,10)
        for i in range(self.ROBN):
            self.swarm[i]=ROBOT(self,str(i))
            chosen=rnd.sample(possiblePos,1)[0]
            '''  delete the selected position from all posible poses ''' 
            possiblePos.remove(chosen) 
            self.swarm[i].position=np.ravel(np.array(chosen))
            self.swarm[i].rotation2B=np.asarray([rnd.sample(list(possibleRot),1)[0]])
            """ 
            adding one dimention to rotation2B is because if you say rotation2B=2 its 
            address will change but if you say rotation2B[0]=2 the addres wont be touched
            """
            self.swarm[i].position=self.swarm[i].position.astype('float64')
            ''' all robots share the same arena texture. if one changes sth, the arena will 
            be changed in every one. equalled by address'''
            self.swarm[i].ground=self.ground 
            
            ''' sharing the Qtable addres for all robots'''
            if self.globalQ:
                if i==0: tmp=self.swarm[i].Qtable
                else: self.swarm[i].Qtable=tmp


        '''vectorized functions to get position and rotation of all robots'''
        self.swarmPosList=[]
        for i in range(self.ROBN):
             self.swarmPosList.append(self.swarm[i].position) 
        ''' ^ its a list that tracks the changes in position. Pointer stuff. if you make it array,
        it will stop tracking the changes and only will store the value at the moment'''
        self.swarmRotList=[]
        for i in range(self.ROBN):
             self.swarmRotList.append(self.swarm[i].rotation2B)
        
        for i in range(self.ROBN):
            self.NASfunction.append(np.asarray(self.swarm[i].groundSensorValue))

# visualize ....................................................................................................................
    def visualize(self):
        if self.vizFlag:
            background=np.copy(self.ground)
            for i in range(self.ROBN):
                vizPos=[]
                for ii in range(len(self.swarm[i].position)): vizPos.append(int(self.swarm[i].position[ii]))
                RobotColor=(255,100,100)
                if i==0:
                    ''' robot0 is in purple for debug purposes '''
                    RobotColor=(128,0,128)

                if self.method=="RL":
                    if self.swarm[i].inAction== True:
                        I=tuple(map(lambda x: int(round(x)),self.swarm[i].initialPos))
                        E=tuple(map(lambda x: int(round(x)),self.swarm[i].desiredPos))
                        Q=tuple(map(lambda x: int(round(x)),self.swarm[i].QRloc[self.swarm[i].lastdetectedQR]))
                        cv.arrowedLine(background,Q,E,(0,0,255),2) # action space vector
                        cv.arrowedLine(background,I,E,(0,255,0),2) # vector that must be traversed
                        cv.arrowedLine(background,I,Q,(100,100,0),2) # sudo vec

                    if self.swarm[i].ExploreExploit=='Exploit' and self.swarm[i].inAction== True:
                        RobotColor=(100,255,100) # color of robot will be green if exploits. otherwise blue again


                elif self.method=="LBA":
                    if self.swarm[i].checkArrived== True:
                        I=tuple(map(lambda x: int(round(x)),self.swarm[i].initialPos))
                        E=tuple(map(lambda x: int(round(x)),self.swarm[i].desiredPos))
                        Q=tuple(map(lambda x: int(round(x)),self.swarm[i].QRloc[self.swarm[i].lastdetectedQR]))
                        cv.arrowedLine(background,Q,E,(0,0,255),2) # action space vector
                        cv.arrowedLine(background,I,E,(0,255,0),2) # vector that must be traversed
                        cv.arrowedLine(background,I,Q,(100,100,0),2) # sudo vec
                    
                cv.circle(background,tuple(vizPos),self.robotSenseRad,RobotColor,1)
                cv.circle(background,tuple(vizPos),m2px(self.robotRad),RobotColor,-1)

                direction=np.array([int(m2px(self.robotRad)*sin(np.radians(self.swarm[i].rotation))) \
                    , int(m2px(self.robotRad)*cos(np.radians(self.swarm[i].rotation)))])
                cv.putText(background,str(i),tuple([vizPos[0]-10,vizPos[1]+10]),cv.FONT_HERSHEY_SIMPLEX\
                    ,0.75,(0,0,0),3 )
                cv.line(background,tuple(vizPos),tuple(np.array(vizPos)+direction),(0,0,200),3)

            cv.putText(background,str(int(self.time))+' s',(20,20),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,100,0),3 )

            if self.record:
                self.video.write(background)
            if self.showFrames:
                ''' with openCV '''
                cv.imshow("background",background)
                cv.waitKey(self.fps)
                ''' with matplotlib '''
                # plt.imshow(background)
                # plt.pause(self.fps*0.0000000000000000000001)
                # plt.draw()
# moveAll ......................................................................................................................
    def moveAll(self):
        for i in range(self.ROBN):
            if self.swarm[i].delayFlag==False:
                self.swarm[i].move()
                pass
            else :
                self.swarm[i].waitingTime-=self.timeStep
                if self.swarm[i].waitingTime<=0:
                    self.checkCollision(specific=True,robotNum=i)            
                    self.swarm[i].delayFlag=False
                    self.swarm[i].move()
        self.time+=self.timeStep
# avoid ........................................................................................................................
    def avoid(self,cols,specific=False):
        if specific==False:
            c=np.reshape(cols,(np.size(cols)//2,2))
            for pairs in c:
                i,j=pairs
                collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] \
                    , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                self.swarm[i].rotation2B[0]=np.asarray(collisionAngle+rnd.randint(90,270) )
                self.swarm[i].rotation2B[0]=np.asarray(RotStandard(self.swarm[i].rotation2B))
                self.flagsR[i,j]=1
                if self.method=="LBA":
                    self.swarm[i].checkArrived=False
                    if self.swarm[i].waitingCue:
                        self.swarm[i].turnPoint+=1
                elif self.method=="RL":
                    self.swarm[i].inAction=False
                i,j=j,i
                collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] \
                    , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                self.swarm[i].rotation2B[0]=np.asarray(collisionAngle+rnd.randint(90,270) )
                self.swarm[i].rotation2B[0]=np.asarray(RotStandard(self.swarm[i].rotation2B))
                self.flagsR[i,j]=1
                if self.method=="LBA":
                    self.swarm[i].checkArrived=False
                    if self.swarm[i].waitingCue:
                        self.swarm[i].turnPoint+=1
                elif self.method=="RL":
                    self.swarm[i].inAction=False
        else:
            c=np.reshape(cols,(np.size(cols)//2,2))
            for pairs in c:
                i,j=pairs
                collisionAngle=RotStandard(np.degrees(atan2( self.swarm[j].position[0]-self.swarm[i].position[0] \
                    , self.swarm[j].position[1]-self.swarm[i].position[1] )))
                self.swarm[i].rotation2B[0]=np.asarray(collisionAngle+rnd.randint(90,270) )
                self.swarm[i].rotation2B[0]=np.asarray(RotStandard(self.swarm[i].rotation2B))
                self.flagsR[i,j]=0
# checkCollision ...............................................................................................................
    def checkCollision(self,specific=False,robotNum=None):
        ''' flagsR is always numerical not boolien'''
        '''specific: if collision avoidence is forced. this onle happens when you want
        to leave aggregation '''
        if specific==False:
            ''' collision with wall detection '''
            self.all_poses=np.array(self.swarmPosList)
            all_rots=np.array(self.swarmRotList)
            # all_rots=self.rot_getter(self.allRobotIndx)

            '''list order: 1 >| , 2 |< , 3 _ , 4 - '''
            loc_conds=[np.where( self.all_poses[:,0]>=self.Ylen-self.robotSenseRad)[0],\
                np.where(self.all_poses[:,0]<=self.robotSenseRad)[0],\
                np.where(self.all_poses[:,1]>=self.Xlen-self.robotSenseRad)[0],\
                np.where(self.all_poses[:,1]<=self.robotSenseRad)[0] ]

            rot_conds=[ np.where(np.logical_and(all_rots<=180,all_rots>=0))[0],\
                np.where(np.logical_and(all_rots<=360,all_rots>=180))[0],\
                np.where(np.logical_or(np.logical_and(all_rots<=360,all_rots>=270),np.logical_and(all_rots<=90,all_rots>=0)))[0],\
                np.where(np.logical_and(all_rots<=270,all_rots>=90))[0]]

            ranges=[(180,360),(0,180),(90,270),(270,360+90)]

            if np.size(loc_conds)>0:
                for i in range(len(loc_conds)):
                    if np.size(loc_conds[i])>0:
                        for j in loc_conds[i]:
                            if j in rot_conds[i]:
                                self.swarm[j].rotation2B[0]=np.asarray(RotStandard(rnd.randint(ranges[i][0],ranges[i][1])))
                                if self.method=='RL':
                                    if self.swarm[j].inAction==True:
                                        self.swarm[j].actAndReward(-1) 
                                        y=self.swarm[j].actionIndx
                                        x=self.swarm[j].state
                                        '''V to catch alpha=1 bug if it happens again '''
                                        # if y<10 and y>0 :
                                            # print('catched',j,self.getTime(),x,y)
                                elif self.method=='LBA' :
                                    if self.swarm[j].waitingCue:
                                        self.swarm[j].turnPoint+=1
                                        
                                    if self.swarm[j].checkArrived: 
                                        ''' update error vector '''
                                        self.swarm[j].checkArrived=False
                                        LQR=int(self.swarm[j].lastdetectedQR[-1])
                                        self.swarm[j].e[LQR-1]+=1
                                        if self.swarm[j].e[LQR-1]>=self.Etol:
                                            self.swarm[j].vecs[LQR-1]*=0

            
            ''' collision with robot detection '''
            Robot1=self.all_poses[self.allnodes[:,0]]
            Robot2=self.all_poses[self.allnodes[:,1]]
            dists=dist(Robot1-Robot2)
            indexes=np.where(dists<=self.collisionDetectDist+self.velocity*self.timeStep) ##############
            colliders=[]
            if np.size(indexes)>0:
                indexes=indexes[0]
                colliders=self.allnodes[indexes]
                '''i want collider to choose from flagsR '''
                flags=self.flagsR[colliders[:,0],colliders[:,1]]==0
                '''i want flags to choose form collider_filterer '''
                colliders=colliders[flags]
                self.avoid(colliders)
            self.colliders=colliders


            cond=self.flagsR==1
            '''if any collision has happend dont touch their matrices'''
            if np.size(indexes)>0: 
                '''dont touch the recently decided ones'''
                cond[colliders[:,0],colliders[:,1]]=False
                '''also reverse of tuple  since matrix is symmetric'''
                cond[colliders[:,1],colliders[:,0]]=False
            self.counterR[cond]+=1
            self.flagsR[self.counterR>=self.collisionDelay]=0
            self.counterR[self.counterR>=self.collisionDelay]=0

        else:
            ''' here collision avoidence is forced with a known robot name: robotNum '''
            temp=np.zeros((self.ROBN-1,2),dtype=int)
            temp[:,0]+=robotNum
            yaxis=self.allRobotIndx
            temp[:,1]+=yaxis[yaxis!=robotNum]
            self.all_poses=np.array(self.swarmPosList)
            Robot1=self.all_poses[temp[:,0]]
            Robot2=self.all_poses[temp[:,1]]
            dists=Robot1-Robot2
            dists=dist(dists)
            indexes=np.where(dists<=self.collisionDetectDist+self.velocity*self.timeStep) #################
            if np.size(indexes)>0:
                indexes=indexes[0]
                colliders_forced=temp[indexes]
                '''i want collider to choose from flags'''
                flags=self.flagsR[colliders_forced[:,0],colliders_forced[:,1]]==0 
                '''i want flags to choose form collider_filterer'''
                colliders_forced=colliders_forced[flags] 
                self.avoid(colliders_forced,specific)
# aggregateSwarm ...............................................................................................................
    def aggregateSwarm(self):
        '''
        a try to vectorized here but better not since some other 
        functions also use the aggregate method of each robot
        y=self.all_poses.astype(int)
        y[:,0],y[:,1]=np.copy(y[:,1]),np.copy(y[:,0])
        groundValues=255-self.ground[y]
        '''
        for i in range(self.ROBN):
            self.swarm[i].aggregate()
# getTime ......................................................................................................................
    def getTime(self):
        return int(self.time)
# getNAS .......................................................................................................................
    def getNAS(self,weighted=False):
        if self.localMinima:
            if weighted:
                return np.sum(self.NASGfunction(self.swarm))/(self.ROBN*255),np.sum(self.NASLfunction(self.swarm))/(self.ROBN*255)
            else:
                return np.count_nonzero(self.NASGfunction(self.swarm))/self.ROBN,np.count_nonzero(self.NASLfunction(self.swarm))/self.ROBN

        else:
            ''' no discrimination of local and global cue '''
            if weighted==False:
                return np.count_nonzero(np.array(self.NASfunction))/self.ROBN
            elif weighted==True:
                return np.sum(np.array(self.NASfunction))/(self.ROBN*255)
# getQRs .......................................................................................................................
    def getQRs(self):
        '''
        old fashion non vectorized way:
        for i in range(self.ROBN):
            self.swarm[i].detectQR()
        '''
        robots=self.all_poses[self.allRobotQRs[:,0]]
        QRs=self.QRpos_ar[self.allRobotQRs[:,1]]
        dists=dist(robots-QRs)
        indx=np.where(dists<=self.visibleRaduis)
        detects=self.allRobotQRs[indx]
        allRobotIndx=np.copy(self.allRobotIndx)
        if len(detects)>0:
            for i,j in detects:
                ''' these robots has detected QR. so change their QR parameter and
                delete their indexes from  allRobotIndx so the rest gets QR0'''
                self.swarm[i].detectedQR='QR'+str(j+1)
                allRobotIndx=np.delete(allRobotIndx,allRobotIndx==i)
                '''lastdetectedQR only saves the non zero values of QR. for RL it is only used for visualization'''
                self.swarm[i].lastdetectedQR=self.swarm[i].detectedQR

            
        for i in allRobotIndx:
            self.swarm[i].detectedQR='QR0'
# swarmRL ......................................................................................................................
    def swarmRL(self):
        for i in range(self.ROBN):
            '''first you must not recently collided. scond you must be either doing an action >or< you must be out of QR area
            the first part is to avoid a bug which happens when robots collide inside QR-code'''
            if (not i in self.colliders) and (self.swarm[i].detectedQR != 'QR0' or self.swarm[i].inAction==True)  :
                self.swarm[i].RL()
# talk .........................................................................................................................
    def talk(self): ##### print must be fixed ####### epsilon sharing policy must be changed
        for i in range(self.ROBN):
            if any(self.flagsR[i]):
                tobesharedRobots=np.where(self.flagsR[i])[0]
                for j in tobesharedRobots:
                    temp=TableCompare(self.swarm[i].Qtable,self.swarm[j].Qtable)
                    if self.vizFlag : print('table shared among ',i,j,np.all(self.swarm[i].Qtable==self.swarm[j].Qtable))
                    self.swarm[i].Qtable=np.copy(temp)
                    self.swarm[j].Qtable=np.copy(temp)
                    if self.vizFlag : print(np.all(self.swarm[i].Qtable==self.swarm[j].Qtable),\
                        self.swarm[i].RLparams['epsilon'],self.swarm[j].RLparams['epsilon'])
                    
                    temp=min(self.swarm[i].RLparams['epsilon'],self.swarm[j].RLparams['epsilon'])
                    self.swarm[i].RLparams['epsilon']=temp
                    self.swarm[j].RLparams['epsilon']=temp
                    if self.vizFlag : print('eps after',self.swarm[i].RLparams['epsilon'],self.swarm[j].RLparams['epsilon'])
        if self.vizFlag : print("------------------------------------------------------")
# getLog .......................................................................................................................
    def getLog(self):
        location=[]
        for i in range(self.ROBN):
            location.append(np.append(self.swarm[i].position,self.swarm[i].rotation))
        return np.array(location)
# getQtables ...................................................................................................................
    def getQtables(self):
        Qtables=[]
        for i in range(self.ROBN):
            Qtables.append(self.swarm[i].Qtable)
        return np.array(Qtables)
# getReward ....................................................................................................................
    def getReward(self):
            return np.array([self.swarm[_].rewardMemory[-1] if len(self.swarm[_].rewardMemory)>0 else 0 for _ in range(self.ROBN)]) 
# getEps .......................................................................................................................
    def getEps(self):
        if self.paramReductionMethod=='classic' or self.paramReductionMethod=='cyclical':
            return np.array([self.swarm[_].RLparams['epsilon'] for _ in range(self.ROBN)]) 
        elif self.paramReductionMethod=='VDBE':
            return np.array([self.swarm[_].eps_1d for _ in range(self.ROBN)]) #[1:] is because discarding state 0
# getAlpha .....................................................................................................................
    def getAlpha(self):
        return np.array([np.mean(self.swarm[_].alpha) for _ in range(self.ROBN)]) 
# changeGround .................................................................................................................
    def changeGround(self):
        ''' this code will ruin address exchange'''
        self.ground=cv.rotate(self.ground,cv.ROTATE_180)
        if self.localMinima: self.cueLcenter,self.cueGcenter=np.copy(self.cueGcenter),np.copy(self.cueLcenter)
        for i in range(self.ROBN):
            self.swarm[i].ground=self.ground
        ''' now address equality returned '''
# LBA .................................................................................................................
    def LBA(self):
        for i in range(self.ROBN):
            if self.swarm[i].detectedQR!='QR0':
                ''' a QR is detected '''
                QR=int(self.swarm[i].detectedQR[-1])
                if np.all(self.swarm[i].vecs[QR-1]==0):
                    """ we see the QR for the first time. so we are WAITING TO LEARN """
                    self.swarm[i].waitingCue=True

                elif self.swarm[i].checkArrived==False:
                    angle=0;length=0
                    """ this QR have been seen before. EXECUTE VECTOR """
                    sudoVec=self.QRpos_ar[QR-1]-self.swarm[i].position
                    actionXY_SudoVec=self.swarm[i].vecs[QR-1]+sudoVec
                    angle=np.degrees(atan2(actionXY_SudoVec[0],actionXY_SudoVec[1]))+self.Noise("angle")
                    length=sqrt(actionXY_SudoVec[0]**2+actionXY_SudoVec[1]**2)+self.Noise("length")
                    self.swarm[i].rotation2B[0]=np.asarray(angle)
                    self.swarm[i].initialPos=np.copy(self.swarm[i].position)
                    # self.swarm[i].desiredPos=self.swarm[i].initialPos+actionXY_SudoVec
                    self.swarm[i].desiredPos=self.swarm[i].initialPos+\
                        np.array([length*sin(np.radians(angle)),length*cos(np.radians(angle))])
                    self.swarm[i].checkArrived=True


            elif self.swarm[i].waitingCue and self.swarm[i].groundSensorValue>0 and np.any(self.flagsR[i]==1):
                ''' robot has collided inside the cue and has seen QR previously. so we LEARN '''
                self.swarm[i].waitingCue=False
                angle_n=0
                length_n=0
                for _ in range(self.swarm[i].turnPoint):
                    angle_n+=self.Noise("angle")
                    length_n+=self.Noise("length")
                self.swarm[i].turnPoint=0
                LQR=int(self.swarm[i].lastdetectedQR[-1])
                p1=self.QRpos_ar[LQR-1]
                p2=self.swarm[i].position
                connector=p2-p1
                angle=np.degrees(atan2(connector[0],connector[1]))+angle_n
                length=sqrt(connector[0]**2+connector[1]**2)+length_n
                connector=np.array([length*sin(np.radians(angle)),length*cos(np.radians(angle))])
                self.swarm[i].vecs[LQR-1]=np.copy(connector)
# Noise .................................................................................................................
    def Noise(self,varname):
        if not self.noise:
            return 0
        return self.sigma[varname]*(2*rnd.random()-1)
################################################################################################################################
################################################################################################################################
################################################################################################################################
class ROBOT(SUPERVISOR):
    def __init__(self,SUPERVISOR,name):
        self.SUPERVISOR=SUPERVISOR # address of supervisor hass passed here
        self.ground=self.SUPERVISOR.ground # grounds will also have same address since it is array
        self.robotName=name
        super().sharedParams()
        self.ground=0 # initilizing for robots
        self.rotation=[0]
        self.rotation2B=[0]
        '''self.position[1]>self.position[0] -> first element of position is for x axis of the
        image of the vertical rectangle. for self.ground it is inverse since its first element
        is bigger than the second one'''
        self.position=[0,0]
        self.position2B=[0,0]
        self.groundSensorValue=0
        self.waitingTime=0
        self.delayFlag=False

        ''' only robot 0 will talk '''
        self.printFlag=True if self.robotName=='0' and self.printFlag else False 

        '''self.rewardNoise: flag that says if reward will have noise or not 
        self.rewardNoise=self.SUPERVISOR.noise
        self.noiseStrength=self.rewardNoise
        '''
        ''' for local and global minima '''
        if self.SUPERVISOR.localMinima:
            self.groundSensorValueG=0
            self.groundSensorValueL=0

        if self.SUPERVISOR.method=="LBA" or self.SUPERVISOR.method=="RL":
            ''' common parameters of LBA and RL '''
            self.detectedQR='QR0'
            self.lastdetectedQR='QR0'
            self.initialPos=0
            self.sudoVec=0
            self.desiredPos=0
            if self.SUPERVISOR.method=="RL":
                ''' parameters of just RL '''
                self.Qtable=np.zeros((self.numberOfStates,self.NumberOfActions))
                self.prevQtable=np.zeros((self.numberOfStates,self.NumberOfActions))
                self.QtableCheck=np.zeros((self.numberOfStates,self.NumberOfActions))
                self.exploredAmount=0
                self.inAction=False
                self.action=0
                self.state=0
                self.rewardMemory=[]
                self.ExploreExploit=''
                self.actionIndx=0
                ''' start with the predetermined initial values '''
                self.epsilon=np.zeros(np.shape(self.Qtable))+self.RLparams['epsilon']
                self.alpha=np.zeros(np.shape(self.Qtable))+self.RLparams['alpha']
                self.eps_1d=np.zeros((np.shape(self.Qtable)[0],))+self.RLparams['epsilon']
                self.prev_eps_1d=np.zeros((np.shape(self.Qtable)[0],))+self.RLparams['epsilon']

                ''' making first row striped'''
                x=np.arange(0,len(self.epsilon[0]))
                self.epsilon[0,x%2==0]=255

                self.SAR=[]
                ''' for cyclical '''
                self.epoch=0
            elif self.SUPERVISOR.method=="LBA":
                ''' parameters of just LBA '''
                self.turnPoint=0
                self.e=np.zeros(len(self.SUPERVISOR.QRloc))
                self.vecs=np.zeros((len(self.SUPERVISOR.QRloc),2))
                self.checkArrived=False
                self.waitingCue=False
# move .........................................................................................................................
    def move(self):
            self.rotation[0]=self.rotation2B[0]
            ''' assume you are gone/ 2* is that we want the robot to move two times
            to avoid repititive cols '''
            self.position2B[0]=self.position[0]+self.velocity*sin(np.radians(self.rotation))*self.timeStep
            self.position2B[1]=self.position[1]+self.velocity*cos(np.radians(self.rotation))*self.timeStep
            self.position2B[0]=min(self.position2B[0],self.Ylen-1)
            self.position2B[1]=min(self.position2B[1],self.Xlen-1)
            self.position2B[0]=max(self.position2B[0],0+1)
            self.position2B[1]=max(self.position2B[1],0+1)
            if self.robotName=='0' and self.SUPERVISOR.getTime()<1:
                print(colored("\t[+] traveled distance in each cycle in px:","green"),round(dist(self.position-self.position2B)))
            ''' check col with 2B poses'''
            temp=np.zeros((self.SUPERVISOR.ROBN-1,2),dtype=int)
            robotNum=int(self.robotName)
            temp[:,0]+=robotNum
            yaxis=np.arange(0,self.SUPERVISOR.ROBN)
            temp[:,1]+=yaxis[yaxis!=robotNum]
            Ys=np.array(self.SUPERVISOR.swarmPosList)[temp[:,1]]
            Xs=np.zeros(Ys.shape)
            Xs[:]=self.position2B
            dists=dist(Xs-Ys)
            indexes=np.where(dists<=self.SUPERVISOR.collisionDetectDist)
            if np.size(indexes)>0:
                ''' cant leave the aggregation with this angle '''
                self.aggregate(forced=True)
            else:
                ''' easily leave the aggregation ''' 
                # self.position=np.copy(self.position2B) # this one distorts the address
                self.position[0],self.position[1]=self.position2B[0],self.position2B[1]

# groundSense ..................................................................................................................
    def groundSense(self):
        temp=self.ground[int(round(self.position[1])),int(round(self.position[0]))]
        if self.position[0]<= self.Ylen-(self.SUPERVISOR.visibleRaduis+2) and self.position[0]>= (self.SUPERVISOR.visibleRaduis+2)\
            and self.position[1]>=(self.SUPERVISOR.visibleRaduis+2) and self.position[1]<=self.Xlen-(self.SUPERVISOR.visibleRaduis+2):

            self.groundSensorValue=255-temp[0]
        else: self.groundSensorValue=0
        
        if self.SUPERVISOR.localMinima:
            ''' if discrimination of local and global cue is needed'''
            if dist(self.position-self.SUPERVISOR.cueGcenter)<=self.SUPERVISOR.cueRadius:
                self.groundSensorValueG=self.groundSensorValue
                self.groundSensorValueL=0

            elif dist(self.position-self.SUPERVISOR.cueLcenter)<=self.SUPERVISOR.cueRadius:
                self.groundSensorValueL=self.groundSensorValue
                self.groundSensorValueG=0
            else:
                self.groundSensorValueL=0
                self.groundSensorValueG=0
# aggregate ....................................................................................................................
    def aggregate(self,forced=False):
        self.groundSense()
        if ( any(self.SUPERVISOR.flagsR[int(self.robotName)]) and self.groundSensorValue>0 and not self.delayFlag ) or forced:
            self.waitingTime=self.SUPERVISOR.Wmax*((self.groundSensorValue**2)/((self.groundSensorValue**2) + 5000))
            self.delayFlag=True
            forced=False
# RL ...........................................................................................................................
    def RL(self):
        if self.inAction==False :
            self.state=int(self.detectedQR[-1])
            if self.SUPERVISOR.paramReductionMethod=='classic' or self.SUPERVISOR.paramReductionMethod=='cyclical':
                eps=self.RLparams['epsilon']
            elif self.SUPERVISOR.paramReductionMethod=='VDBE':
                eps=self.eps_1d[self.state]
            else: raise NameError('[-] method could not be found')

            
            if rnd.random()<= eps: ###################
                self.action=rnd.sample(self.actionSpace,1)[0]
                self.ExploreExploit='Explore'
            else:
                actionIndx=np.argmax(self.Qtable[self.state,:])
                self.action=self.actionSpace[actionIndx]
                self.ExploreExploit='Exploit'
        self.actAndReward()
# actAndReward .................................................................................................................
    def actAndReward(self,rewardInp=None):
        if self.inAction==False:
            angle=self.action[1]
            length=self.action[0]
            if self.state<=3: angle=180+angle # caviat

            actionXY=np.array([length*sin(np.radians(angle)),length*cos(np.radians(angle))])
            self.sudoVec= np.array(self.QRloc[self.detectedQR])-self.position
            actionXY_SudoVec=actionXY+self.sudoVec
            angle=np.degrees(atan2(actionXY_SudoVec[0],actionXY_SudoVec[1]))+self.SUPERVISOR.Noise("angle")
            length=sqrt(actionXY_SudoVec[0]**2+actionXY_SudoVec[1]**2)+self.SUPERVISOR.Noise("length")
            actionXY_SudoVec=np.array([length*sin(np.radians(angle)),length*cos(np.radians(angle))])

            self.rotation2B[0]=np.asarray(angle)
            self.inAction=True
            self.initialPos=np.copy(self.position)
            self.reward=0
            self.desiredPos=self.initialPos+actionXY_SudoVec


        elif (self.inAction==True and dist(self.position-self.desiredPos)<=self.desiredPosDistpx) or rewardInp!=None:
            ''' elif goal reached or a reward is forced '''
            self.inAction=False
            if rewardInp==None:
                self.groundSense()
                self.reward=self.groundSensorValue
                ''' add noise to reward (old way)
                if self.rewardNoise:
                    self.reward+=rnd.randint(-1*self.noiseStrength, self.noiseStrength)
                    self.reward=max(0,self.reward)
                    self.reward=min(self.reward,255)
                '''
            else: self.reward=rewardInp
            self.actionIndx=self.actionSpace.index(self.action)
            x=self.state
            y=self.actionIndx

            self.prevQtable[x,y]=self.Qtable[x,y]

            ''' update rule: bellman '''
            self.Qtable[x,y]+=self.RLparams['alpha']*(self.reward-self.Qtable[x,y]) ########################

            self.updateRLparameters(x,y)

            self.QtableCheck[x,y]=1
            self.exploredAmount=(self.QtableCheck==1).sum()/np.size(self.Qtable[1:]) # discarding data of state 0

            ''' logging reward and SAR '''
            self.rewardMemory.append(self.reward)
            if self.robotName=='0':
                self.SAR.append(np.array([self.SUPERVISOR.getTime(),x,y,self.reward]))

            '''for debugging alpha effect
            if y<10 and y>0 and self.reward==-1:
                print('catched',self.robotName,self.SUPERVISOR.getTime(),x,y,self.reward)'''
# updateRLparameters ...........................................................................................................
    def updateRLparameters(self,x=None,y=None):
        if self.SUPERVISOR.paramReductionMethod=='classic':
            self.RLparams["epsilon"]*=self.EpsilonDampRatio ####
        elif self.SUPERVISOR.paramReductionMethod=='VDBE':
            ''' assuming sigma=1'''
            sigma=self.SUPERVISOR.PRMparameter

            delta=1/self.NumberOfActions # caviat: inverse of number of actions. number of actions is known so cheated
            ''' less delta means slower updates so small delta will change eps slowly '''
            comm_term=np.exp(-np.abs(self.Qtable[x,y]-self.prevQtable[x,y])/sigma)
            f=(1-comm_term)/(1+comm_term)
            self.prev_eps_1d[x]=self.eps_1d[x]
            self.eps_1d[x]=delta*f+(1-delta)*self.eps_1d[x]

            if self.printFlag:
                print('\t\t[+] state {} f {:2.2f} comm_term {:2.2f} Qtable {:2.2f} prevQtable {:2.2f} \n\t\t prev_eps {:2.2f} new_eps {:2.2f}'\
                    .format(x,f,comm_term,self.Qtable[x,y],self.prevQtable[x,y],self.prev_eps_1d[x],self.eps_1d[x]))
        elif self.SUPERVISOR.paramReductionMethod=='cyclical':
            self.epoch+=1
            s=self.SUPERVISOR.PRMparameter # period 
            self.RLparams["epsilon"]=(cos(self.epoch*2*np.pi/s)+1)/2
