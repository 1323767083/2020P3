import os
import pandas as pd
class WR_handler:
    def __init__(self, lc,process_name, process_group_name, logger):
        self.lc=lc
        self.process_name=process_name
        self.logger=logger
        self.log_WR_column=["BW","BZ","BR","NW","NZ","NR","NA"]
        self.log_WRs=[]
        self.log_PAs=[]

        self.WR_working_dir,self.PA_working_dir="",""
        for tag in ["WR","PA"]:
            wdir= lc.system_working_dir
            for sub_dir in ["Classifiaction",tag, process_group_name]:
                wdir = os.path.join(wdir, sub_dir)
                if not os.path.exists(wdir): os.mkdir(wdir)
            setattr(self,"{0}_working_dir".format(tag), wdir)


    def save(self, eval_loop_count):
        fnwp=os.path.join(self.WR_working_dir, "ET{0}.csv".format(eval_loop_count))
        pd.DataFrame(self.log_WRs, columns=self.log_WR_column).to_csv(fnwp,index=False)

        fnwp=os.path.join(self.PA_working_dir, "ET{0}.csv".format(eval_loop_count))
        pd.DataFrame(self.log_PAs).to_csv(fnwp,index=False, float_format="%.2f")

        self.reset_logs()
    def reset_logs(self):
        del self.log_WRs[:]
        del self.log_PAs[:]

    def Fabricate_PA(self, profit_log):
        return profit_log

    def Fabricate_WR(self, l_WR):
        # ["BW", "BZ", "BR", "NW", "NZ", "NR", "NA"]
        return [l_WR.count(0),l_WR.count(1),l_WR.count(2),l_WR.count(10),l_WR.count(11),l_WR.count(12),l_WR.count(-1)]

    def add_log(self,ll_log):
        l_WR,l_PA=ll_log
        self.log_WRs.append(self.Fabricate_WR(l_WR))
        self.log_PAs.append(self.Fabricate_PA(l_PA))
