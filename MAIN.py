from HEADER import *
import json

#...............................................................................................................................
def saveData(caller=None):
    print(colored("\n\t\t[+] data saved ",'yellow'))
    dataType='process data'
    if method=="RL":
        QtableMem[it,:,:,:]=sup.getQtables()
    if localMinima:
        data2BsavedStr=["NASwG","NASG","NASwL","NASL","log","Qtable","rewards","eps","alpha"]
        data2Bsaved=[NASwG,NASG,NASwL,NASL,log,QtableMem,reward,eps,alpha]
    else:
        data2BsavedStr=["NASw","NAS","log","Qtable","rewards","eps","alpha"]
        data2Bsaved=[NASw,NAS,log,QtableMem,reward,eps,alpha]
    fileName=codeBeginTime+dirChangeCharacter+dataType+dirChangeCharacter+str(it)+' '+str(sup.getTime())+' s '+ctime(TIME()).replace(':','_')+' '
    for i in range(len(data2Bsaved)):
        with open(fileName+data2BsavedStr[i]+commentDividerChar+comment+'.npy','wb') as f:
            np.save(f,data2Bsaved[i])

    if caller=='iterationDone':

        with open(fileName+commentDividerChar+comment+'.sup', 'wb') as supSaver:
            try:
                if hasattr(sup,'video'):
                    del sup.video
                del sup.NASfunction
                if localMinima:
                    del sup.NASGfunction
                    del sup.NASfunction
                pickle.dump(sup, supSaver)
            except Exception as E: 
                print(colored('\t\t[-] error in saving class: '+str(E),'red'))
    elif caller=='interupt':
        ''' you should not delete any thing here, for code will continue after here'''
        with open(fileName+commentDividerChar+comment+'.sup', 'wb') as supSaver:
            try:
                pickle.dump(sup, supSaver)
            except Exception as E: 
                print(colored('\t\t[-] error in saving class: '+str(E),'red'))
#...............................................................................................................................
def keyboardInterruptHandler(signal, frame):
    saveData('interupt')
    print(colored('\t\t[+] half data saved. Time: '+str(sup.getTime())+', ratio '+str(int(sup.getTime()/FinalTime*100))+'%','green'))
    ans=input(colored('\t\t[+] continue/quit? [c/q]','green'))
    if ans=='q': exit(0)
    else:print(colored('\t\t[+] continuing','green'))
signal.signal(signal.SIGINT, keyboardInterruptHandler)                  
#...............................................................................................................................
def clearTerminal(): 
    ''' check and make call for specific operating system '''
    # call('clear' if os.name =='posix' else 'cls')
    if os.name=='nt':os.system('cls')
    else:os.system('clear')
#...............................................................................................................................
def LOG():
    global sup,samplingPeriodSmall,it,sampled,save_csv,codeBeginTime,dirChangeCharacter,FinalTime,save_tables_videos,videoList,record,showFrames,SAR,localMinima,t
    global NASw,NAS,log,eps,alpha,reward # caviat: if localMinima is true, its matrices are not globalized here

    # sup.visualize()
    if sup.getTime()%samplingPeriodSmall==0 and sup.getTime()-t>1:
        '''logs specail for first iteration'''
        if method=='RL':
            if it==0: # save these in the first iteration only
                if sampled % 100==0 :
                    # sup.visualize() # moved for less file size >>>>>>>>>>>>>> alert
                    '''save csvs '''
                    if save_csv:
                        QtableRob0=sup.getQtables()[0]
                        np.savetxt(codeBeginTime+dirChangeCharacter+'csvs'+dirChangeCharacter+str(sup.getTime())+".csv", np.round(QtableRob0,2), delimiter=",")

                    '''save tables videos '''
                    if save_tables_videos:
                        imsMat=[sup.swarm[0].epsilon*255,sup.swarm[0].QtableCheck*255,QtableRob0]
                        for count_,v in enumerate(imsMat):
                            v=np.minimum(v,np.ones(v.shape)*255)
                            v[0]=strip
                            im=255-v
                            im.astype(int)
                            canvas=np.zeros((im.shape[0],im.shape[1],3))
                            canvas[:,:,0]=im
                            canvas[:,:,1]=im
                            canvas[:,:,2]=im
                            canvas=cv.resize(canvas,tableImSize)
                            videoList[count_].write(np.uint8(canvas))

                elif abs(FinalTime-sup.getTime())<1:
                    '''iteration 0 is about to end. so release the video and turn of record
                    flag so we dont have any video attribute afterwards and save SARs as npy ans csv '''
                    if save_tables_videos:
                        for _ in range(len(videoList)):
                            videoList[_].release()
                    record=False
                    showFrames=False
                    if hasattr(sup,'video'): sup.video.release()
                    ''' save SAR '''
                    SAR=np.stack(sup.swarm[0].SAR)
                    with open(codeBeginTime+dirChangeCharacter+'SAR_robot0.npy','wb') as SAR_f:
                        np.save(SAR_f,SAR)
                    np.savetxt(codeBeginTime+dirChangeCharacter+'SAR_robot0.csv',np.round(SAR), delimiter=",")

            ''' in every iteration, log the vital performance indexes with frequency of samplingPeriodSmall''' 
            if localMinima:
                NASG[it,sampled],NASL[it,sampled]=sup.getNAS()                    
                NASwG[it,sampled],NASwL[it,sampled]=sup.getNAS(weighted=True)                    
            else:
                # NASw[it,sampled]=sup.getNAS(weighted=True)
                NAS[it,sampled]=sup.getNAS()
            # log[it,sampled,:,:]=sup.getLog()
            eps[it,sampled,:]=sup.getEps()
            # alpha[it,sampled,:]=sup.getAlpha()
            reward[it,sampled,:]=sup.getReward()
        elif method=='BEECLUST' or method=='LBA':
            if it==0: # save these in the first iteration only
                sup.visualize() 
                if abs(FinalTime-sup.getTime())<1:
                    '''iteration 0 is about to end. so release the video and turn of record
                    flag so we dont have any video attribute afterwards and save SARs as npy ans csv '''
                    record=False
                    showFrames=False
                    if hasattr(sup,'video'): sup.video.release()
            # NASw[it,sampled]=sup.getNAS(weighted=True)
            NAS[it,sampled]=sup.getNAS()

            if abs(FinalTime-sup.getTime())<1:
                '''iteration 0 is about to end. so release the video and turn of record
                flag so we dont have any video attribute afterwards and save SARs as npy ans csv '''
                record=False
                showFrames=False
                if hasattr(sup,'video'): sup.video.release()
        else:
            raise NameError("[-] METHOD NOT RECOGNIZED")
        sampled+=1
        t=sup.getTime()
#...............................................................................................................................
def init_params():
    # print(">>>>",sys.argv[1])

    with open('params.JSON') as json_file: 
        data = json.load(json_file) 
    for key in data.keys():
        globals()[key+"_dict"]=data[key]
    flags=[]
    for key, value in flags_dict.items():
        flags.append([key, value])
    vals=[]
    for key, value in vals_dict.items():
        vals.append([key, value])


    # flags=[
    # ["dynamic",True],
    # ["localMinima",False],
    # # ["noise",True],
    # ["showFrames",False],
    # ["record",True],
    # ["globalQ",False],
    # ["communicate",False],
    # ["save_csv",False],
    # ["save_tables_videos",False]]
    # vals=[
    # ["Lx",round(1.35,2)],
    # ["Ly",round(2.7,2)],
    # ["cueRaduis",round(0.3,2)],
    # ["visibleRaduis",round(0.3,3)],
    # ["iteration",5],
    # ["samplingPeriodSmall",10],
    # ["FinalTime",int(90*60)],
    # ["HalfTime",int(90*60//2)],
    # ["ROBN",4],
    # ["paramReductionMethod","cyclical"],
    # # ["PRMparameter",float(sys.argv[1])],
    # ["PRMparameter",500],
    # ["comment","real robot simulation (arena and action space changed)"],
    # ["commentDividerChar"," x "],
    # ["method","RL"],
    # ["noise",15],
    # ["seed_value","x"]]

    """ make every one str """
    for c in range(len(vals)):
        if not(type(vals[c][1]) is str):
            vals[c][1]=str(vals[c][1])

    # GUI_flag=not True
    # if GUI_flag:
    #     from GUI import parameterGUI
    #     pgui=parameterGUI(flags,vals)
    #     vals,flags=pgui.vals,pgui.flags
    #     del pgui
    #     del parameterGUI
    global parameters
    parameters=[]
    for valg_name, val_value in vals:
        if val_value.replace(".","").isdigit():
            if "." in val_value:
                val_value=float(val_value)
            elif val_value.isdigit():
                val_value=int(val_value)
        else:
            try:
                ''' if I have written an exdpression calculate it '''
                val_value=eval(val_value)
                ''' if it is int change its type to int '''
                if val_value%1==0:
                    val_value=int(val_value)
            except: 
                pass
        ''' eval function diffrentiates / and //'''
        globals()[valg_name]=val_value
        parameters.append([valg_name,val_value])
    print(colored("[+] parameters:","green"),parameters)
    for flag_name, flag_value in flags:
        globals()[flag_name]=True if (flag_value=="True" or flag_value=="true") else False

    global comment
    if method=="RL":
        comment=paramReductionMethod+' '+str(PRMparameter)+' '+comment+' noise '+str(noise)
    else:
        comment=method+' '+comment+' noise '+str(noise)

#...............................................................................................................................
if __name__ == "__main__" or True:
    print(colored("VVVVVVVVVVVVVVVVVV STARTED VVVVVVVVVVVVVVVVVV","yellow"))
    # print(colored("[!] be carefull avout sup.visualuz","red"))
    # print(colored("[!] action space changed","red"))
    DirLocManage()
    init_params()    

    '''initiate seed'''
    seed=set_seed(seed_value)
    param0=[_[0] for _ in parameters]
    indx=param0.index("seed_value")
    parameters[indx][1]=seed
    # indx=parameters.in


    t1_=TIME()
    # clearTerminal()
    ''' call wsential functions '''
    warningSupress()
    dirChangeCharacter=DirLocManage()

    ''' parameter value assigning '''
    print(colored('[+] '+comment,'green'))
    print(colored('[+] paramReductionMethod','green'),paramReductionMethod,PRMparameter)

    # LOGthrd=threading.Thread(target=LOG)

    codeBeginTime=ctime(TIME()).replace(':','_')+'_'+method+'_'+comment
    if globalQ and communicate:
        '''local and global communication cant be toghether '''
        raise NameError('[-] what do you want?')

    ''' preparing dirs '''
    os.makedirs(codeBeginTime)
    os.makedirs(codeBeginTime+dirChangeCharacter+'process data')

    ''' save parameters into a file '''
    with open(codeBeginTime+dirChangeCharacter+'params.txt','w') as paramfile :
        paramfile.write(str(parameters))

    ''' for saving csvs which is Q-table of robot 0 for iteration 0 '''
    if save_csv: os.makedirs(codeBeginTime+dirChangeCharacter+'csvs') 


    ''' initilization '''
    sampledDataNum=FinalTime//samplingPeriodSmall
    saved=0
    print(colored('[+] '+method,'green'))
    print(colored('[+] press ctrl+c for saving data asynchronously','green'))
    # QtableMem=np.zeros((iteration,ROBN,7,44)) ##### caviat
    QtableMem=np.zeros((iteration,ROBN,7,6)) ##### caviat
    
    log=np.zeros((iteration,sampledDataNum,ROBN,3))
    if paramReductionMethod=='classical' or paramReductionMethod=='cyclical':
        eps=np.zeros((iteration,sampledDataNum,ROBN))##### caviat
    elif paramReductionMethod=='VDBE':
        eps=np.zeros((iteration,sampledDataNum,ROBN,7))##### caviat

    alpha=np.zeros((iteration,sampledDataNum,ROBN))
    reward=np.zeros((iteration,sampledDataNum,ROBN))
    NAS=np.zeros((iteration,sampledDataNum))
    NASw=np.zeros((iteration,sampledDataNum))
    if localMinima:
        NASG=np.zeros((iteration,sampledDataNum))
        NASwG=np.zeros((iteration,sampledDataNum))
        NASL=np.zeros((iteration,sampledDataNum))
        NASwL=np.zeros((iteration,sampledDataNum))

    # strip=np.arange(0,44) ##### caviat: table dimentions pre known
    strip=np.arange(0,6) ##### caviat: table dimentions pre known

    strip[strip%2==0]=0
    strip[strip%2==1]=255
    # tableImSize=(7*20,44*20)[::-1] ##### caviat: table dimentions pre known
    tableImSize=(7*20,6*20)[::-1] ##### caviat: table dimentions pre known


    if save_tables_videos:
        imsName=["epsilon","QtableCheck","QtableRob0"]
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        FPS=1
        videoList=[]
        for _ in range(len(imsName)):
            videoList.append(cv.VideoWriter(codeBeginTime+DirLocManage(returnchar=True)+imsName[_]+'.mp4',fourcc, FPS, tableImSize,True))


    for it in range(iteration):
        iteration_duration=TIME()
        print(colored("\t[+] iteration: ",'blue'), it)
        t=0;tt=0;sampled=0
        sup=SUPERVISOR(ROBN,codeBeginTime,showFrames,globalQ,record,Lx,Ly,cueRaduis,visibleRaduis,paramReductionMethod,PRMparameter,noise,localMinima,method)
        sup.generateRobots()
        sup.moveAll() # to make initilazation happen
        GroundChanged=False # to make sure ground is changed only once in each iteration
        checkHealth()
        while sup.getTime()<=FinalTime:
            ''' start of main loop '''

            # t1=TIME()
            sup.checkCollision()
            # print("checkCollision ",(TIME()-t1)*10000)

            # t1=TIME()
            sup.aggregateSwarm()
            # print("aggregateSwarm ",(TIME()-t1)*10000)

            if method=='RL':
                # t1=TIME()
                sup.getQRs()
                # print("getQRs ",(TIME()-t1)*10000)

                # t1=TIME()
                sup.swarmRL()
                # print("swarmRL ",(TIME()-t1)*10000)

            elif method=='LBA':
                # t1=TIME()
                sup.getQRs()
                # print("getQRs ",(TIME()-t1)*10000)

                # t1=TIME()
                sup.LBA()
                # print("LBA ",(TIME()-t1)*10000)

            # t1=TIME()
            sup.moveAll()
            # print("moveAll ",(TIME()-t1)*10000)

            if abs(HalfTime-sup.getTime())<1 and GroundChanged==False:
                GroundChanged=True
                print(colored('\t[+] half time reached','green'))
                if dynamic:
                    sup.changeGround()

            # t1=TIME()
            LOG()
            # print("LOG ",(TIME()-t1)*10000)
            # print("---------------------------------------")

            # lp_wrapper = lp(sup.moveAll)
            # lp_wrapper()
            # lp.print_stats()
            # cp.run('sup.moveAll()')
        if method=="RL":
            QtableMem[it,:,:,:]=sup.getQtables()
        '''V: -1 is for that 'it' at max will be iteration-1 and after that code will exit the loop'''
        if it==iteration-1: saveData("iterationDone") 
        del sup
        print(colored("\t[+] iteration duration: ",'blue'),int(TIME()-iteration_duration))
    print(colored('[+] duration','green'),int(TIME()-t1_))
    print(colored('[+] goodbye  ^^',"green"))


