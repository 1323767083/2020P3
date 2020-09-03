import sys
from DBI_Base import DBI_init,StockList
from DBTP_Creater import DBTP_Creater,DBTP_main
commmand_smaples=[
    "Initial_DBI",
    "Reset_DBI",
    "Update_DBI  20200601",
    "Generate_DBTP TPVTest1 SH600000 20200101 20200110",
    "Generate_DBTP_Process TPVTest1 SLV1 4",
    "Create_Total_SL SLV1",
    "Create_Sub_SL SLV1"
]
def main(argv):
    if len(argv)==0:
        print ("Command Format Sample")
        for comand_sample in commmand_smaples:
            print ("\t python DB_main.py ", comand_sample)
        return
    command=argv[0]
    if command == "Initial_DBI":
        i=DBI_init()
        flag, mess = i.init_DBI_lumpsum_Indexes()
        assert flag, ("inital lumpsum index in DBI fail with {0}".format(mess))
        i.init_DBI_lumpsum_HFQs()
    elif command == "Reset_DBI":
        i=DBI_init()
        i.Reset_Init_Index_HFQ()
    elif command == "Update_DBI":
        DayI=int(argv[1])
        i=DBI_init()
        flag, mess=i.Update_DBI_addon_HFQ_Index(DayI)
        print (flag, mess)
    elif command == "Generate DBTP":
        DBTP_Name, Stock, StartS, EndS=argv[1:]
        i = DBTP_Creater(DBTP_Name)
        i. DBTP_generator(Stock, int(StartS), int(EndS))
    elif command == "Generate_DBTP_Process":
        DBTP_Name, SL_Name, NumPS= argv[1:]
        DBTP_main(DBTP_Name, SL_Name, NumP=int(NumPS))
    elif command == "Create_Total_SL":
        SL_Name=argv[1]
        i=StockList(SL_Name)
        i.get_total_stock_list()
    elif command == "Create_Sub_SL":
        SL_Name=argv[1]
        i=StockList(SL_Name)
        i.generate_Train_Eval_stock_list()
    else:
        print ("Command {0} is not supported".format(command))
    print("Finished")
    return

if __name__ == '__main__':
    main(sys.argv[1:] )