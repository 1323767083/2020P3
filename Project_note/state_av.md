#purpose:
> understand the av component for V2


* env.py
> Simulator_LHPP2V2 
> > lhd=[0 for _ in range(lc.LHP+1)]
> > lhd[self.LHP_CHP] = 1 
> > lav=lhd+[potential_profit]

* Buffer_comm
> no change

#proposal
* V2 has av use OS_AV_shape
* V3 and V4 not have av
* marked ##av___change
#action 
1. remove all other AV shape related (Done)
2. add setattr(self.specific_param, "OS_AV_shape", (self.LHP + 1,)) (Done)
3. env, TD_buffer(always send av with lc.specific_param.OS_AV_shape), a3c_worker not change
4. v2 check 
    * agent  (Done)
    * trainer (Done)
    * choose action (Done) 
5. V3 check
    * agent  seems V3 V32 V33 agent is the same only difference is trainer (Done)
    * trainer (Done)
    * choose action (Done)
6. V4 check
    * agent (Done)
    * trainer (Done)
    * choose action (Done)
    


