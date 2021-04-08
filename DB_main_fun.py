import sys,os
import pandas as pd
from datetime import date
from DB_Base import DB_Base
from DBI_Base import DBI_init,StockList
from DBTP_Creater import DBTP_Creater,DBTP_creator_on_SLperiod,DBTP_creator
from DB_FTP import Get_Data_After_closing
def find_addon_last_day(param_name):
    Addon_IN_One_dnwp = "/home/rdchujf/DB_raw_addon/Config"
    param_fnwp = os.path.join(Addon_IN_One_dnwp, "{0}.csv".format(param_name))
    df = pd.read_csv(param_fnwp)
    TPDB_update_log_base_dir = "/home/rdchujf/n_workspace/data/RL_data/TP_DB/Update_Log"
    UpdateToDayIs = []
    for idx, row in df.iterrows():
        dnwp = os.path.join(TPDB_update_log_base_dir, row["DBTP_Name"], "{0}_{1}_{2}".
                            format(row["SL_Name"], row["SL_tag"], int(row["SL_idx"])))
        UpdateToDayIs.append(max([int(dn[9:]) for dn in os.listdir(dnwp)]))
    lastdayIs = list(set(UpdateToDayIs))
    return lastdayIs
    '''
    if len(lastdayIs) == 1:
        print("Update to {0}".format(lastdayIs[0]))
    else:
        print("can not decide in {0}, please check following folders:".format(lastdayIs))
        for idx, row in df.iterrows():
            print(os.path.join(TPDB_update_log_base_dir, row["DBTP_Name"], "{0}_{1}_{2}".
                               format(row["SL_Name"], row["SL_tag"], int(row["SL_idx"]))))
    '''
def addon_in_one(param_name, DateI,flag_Print_on_screen_or_file):
    Addon_IN_One_dnwp = "/home/rdchujf/DB_raw_addon/Config"
    param_fnwp = os.path.join(Addon_IN_One_dnwp, "{0}.csv".format(param_name))
    lastdayIs = find_addon_last_day(param_name)
    if len(lastdayIs) != 1:
        print("can not decide in {0}, please check following folders:".format(lastdayIs))
        print("Check /home/rdchujf/n_workspace/data/RL_data/TP_DB/Update_Log")
        return
    else:
        if lastdayIs[0] >= DateI:
            print("DBI has updated to {0} and should not call Addon_In_One with {1}".format(lastdayIs[0], DateI))
            return

    from contextlib import redirect_stdout, redirect_stderr
    if flag_Print_on_screen_or_file:
        newstdout = sys.__stdout__
        newstderr = sys.__stderr__
        stdoutfnwp, stderrfnwp = "", ""
    else:
        Addon_IN_One_log_dnwp = Addon_IN_One_dnwp
        for subdir in [param_name, str(DateI)]:
            Addon_IN_One_log_dnwp = os.path.join(Addon_IN_One_log_dnwp, subdir)
            if not os.path.exists(Addon_IN_One_log_dnwp): os.mkdir(Addon_IN_One_log_dnwp)
        stdoutfnwp = os.path.join(Addon_IN_One_log_dnwp, "Output.txt")
        stderrfnwp = os.path.join(Addon_IN_One_log_dnwp, "Error.txt")
        print("Output will be direct to {0}".format(stdoutfnwp))
        print("Error will be direct to {0}".format(stderrfnwp))
        newstdout = open(stdoutfnwp, "w")
        newstderr = open(stderrfnwp, "w")

    with redirect_stdout(newstdout), redirect_stderr(newstderr):

        i = Get_Data_After_closing()
        if not i.get_qz_data(DateI):
            print("Fail in downloading qz for {0}".format(DateI))
            return
        else:
            print("Success downloading qz for {0}".format(DateI))
        if not i.get_HFQ_index(DateI):
            print("Fail in downloading HFQ_index for {0}".format(DateI))
            return
        else:
            print("Success downloading HFQ_index for {0}".format(DateI))

        i = DBI_init()
        flag, mess = i.Update_DBI_addon(DateI)
        print("Success" if flag else "Fail", "  ", mess)
        if not flag:
            return
        df = pd.read_csv(param_fnwp)
        for idx, row in df.iterrows():
            print(row["DBTP_Name"], row["SL_Name"], row["SL_tag"], int(row["SL_idx"]))
            DBTP_creator(row["DBTP_Name"], row["SL_Name"], row["SL_tag"], int(row["SL_idx"]), DateI, DateI, 10, False)
    if not flag_Print_on_screen_or_file:
        print("Output will be direct to {0}".format(stdoutfnwp))
        print("Error will be direct to {0}".format(stderrfnwp))
