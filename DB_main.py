from DB_main_fun import *

commmand_smaples=[
    "Initial_DBI",
    "Reset_DBI",
    "Create_Total_SL SLV1",
    "Create_Sub_SL SLV1",
    "Get_SL_Exceed_MaxPrice SLV1 500",
    "Generate_DBTP TPVTest1 SH600000 20200101 20200110",
    "Generate_DBTP_Process TPVTest1 SLV1 4 True/False #True means overwrite create log",
    "Create_List_Stock_Fail_Generate_TPDB SLV1",
    "Addon_Download  20210301",
    "Addon_Update_DBI 20210301",
    "Addon_Generate_DBTP DBTP_Name, SL_Name, SL_tag, SL_idx, StartI, EndI, NumP, flag_overwrite ",
    "Addon_In_One Config_name DateI Screen_or_file",
    "Addon_In_One DateI  # means use default DBTP1M5M10MSL300SL500",
    "Addon_In_One   # means today",
    "Addon_Update_To_Date Config_name",
    "Addon_Update_To_Date #means use defult DBTP1M5M10MSL300SL500"
]

def main(argv):
    if len(argv)==0:
        print ("Command Format Sample")
        for comand_sample in commmand_smaples:
            print ("\t python DB_main.py ", comand_sample)
        return
    command=argv[0]
    if command == "Reset_DBI":
        i=DBI_init()
        i.Reset_Init_Index_HFQ()
    elif command == "Initial_DBI":
        i=DBI_init()
        flag, mess = i.init_DBI_lumpsum_Indexes()
        assert flag, ("inital lumpsum index in DBI fail with {0}".format(mess))
        i.init_DBI_lumpsum_HFQs()
    elif command == "Generate_DBTP":
        DBTP_Name, Stock, StartS, EndS=argv[1:]
        i = DBTP_Base(DBTP_Name)
        tag = i.CLN_DBTPCreater[-2:]
        assert tag in ["V1", "V2"], tag
        if tag=="V1":
            i = DBTP_CreaterV1(DBTP_Name)
            i. DBTP_generator(Stock, int(StartS), int(EndS))
        else:
            i = DBTP_CreaterV2(DBTP_Name,[Stock],"Try")
            i. DBTP_generator(int(StartS), int(EndS))
    elif command == "Generate_DBTP_Process":
        DBTP_Name, SL_Name, NumPS,str_flag_overwrite= argv[1:]
        DBTP_creator_on_SLperiod(DBTP_Name, SL_Name, int(NumPS),eval(str_flag_overwrite))
    elif command == "Create_Total_SL":
        SL_Name=argv[1]
        i=StockList(SL_Name)
        i.Get_Total_SL()
    elif command == "Create_Sub_SL":
        SL_Name=argv[1]
        i=StockList(SL_Name)
        i.generate_Train_Eval_SL()
    elif command == "Get_SL_Exceed_MaxPrice":
        SL_Name = argv[1]
        max_price=eval(argv[2])
        i=StockList(SL_Name)
        i.get_exceed_max_price_sl(max_price)
    elif command == "Create_List_Stock_Fail_Generate_TPDB":
        SL_Name=argv[1]
        i=StockList(SL_Name)
        i.Get_Stocks_Error_Generate_DBTP()
    elif command == "Addon_Download":
        DateI=eval(argv[1])
        i=Get_Data_After_closing()
        print("Success" if i.get_qz_data(DateI) else "Fail", " Download {0} QZ data".format(DateI))
        print("Success" if i.get_HFQ_index(DateI) else "Fail", " Download {0} HFQ and Index data".format(DateI))
    elif command == "Addon_Update_DBI":
        DateI = eval(argv[1])
        i=DBI_init()
        flag, mess=i.Update_DBI_addon(DateI)
        print ("Success" if flag else "Fail", "  ",mess)
    elif command=="Addon_Generate_DBTP":
        DBTP_Name, SL_Name, SL_tag, SL_idx, StartI, EndI, NumP, flag_overwrite=argv[1],argv[2],argv[3],\
                                                eval(argv[4]),eval(argv[5]),eval(argv[6]),eval(argv[7]),eval(argv[8])
        DBTP_creator(DBTP_Name, SL_Name, SL_tag, SL_idx, StartI, EndI, NumP, flag_overwrite)
    elif command=="Addon_Update_To_Date":
        if len(argv)==1:
            param_name = "DBTP1M5M10MSL300SL500"
        else:
            param_name = argv[1]
        lastdayIs=find_addon_last_day(param_name)
        if len(lastdayIs)==1:
            print ("Update to {0}".format(lastdayIs[0]))
        else:
            print ("can not decide in {0}, please check following folders:".format(lastdayIs))
            print ("Check /home/rdchujf/n_workspace/data/RL_data/TP_DB/Update_Log")

    elif command=="Addon_In_One":
        if len(argv)==1:
            param_name = "DBTP1M5M10MSL300SL500"
            today = date.today()
            DateI = int(today.strftime("%Y%m%d"))
            flag_Print_on_screen_or_file = False
        elif len(argv)==2:
            param_name = "DBTP1M5M10MSL300SL500"
            DateI = eval(argv[1])
            flag_Print_on_screen_or_file = False
        else:
            param_name=argv[1]
            DateI=eval(argv[2])
            flag_Print_on_screen_or_file=eval(argv[3])
        addon_in_one(param_name, DateI, flag_Print_on_screen_or_file)
    else:
        print ("Command {0} is not supported".format(command))
    print("Finished")
    return

if __name__ == '__main__':
    main(sys.argv[1:] )