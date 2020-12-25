import sys
from DBI_Base import DBI_init,StockList
from DBTP_Creater import DBTP_Creater,DBTP_main
commmand_smaples=[
    "Initial_DBI",
    "Reset_DBI",
    "Update_DBI  20200601",
    "Create_Total_SL SLV1",
    "Create_Sub_SL SLV1"
    "Generate_DBTP TPVTest1 SH600000 20200101 20200110",
    "Generate_DBTP_Process TPVTest1 SLV1 4 True/False #True means overwrite create log",
    "Create_List_Stock_Fail_Generate_TPDB SLV1"
]

'''
Guide to update DBI

一。拷贝原始文件 （raw data）
1. 把交易的 原始压缩数据按月拷贝到  /home/rdchujf/DB_raw/Normal
2. index文件：
    a。删除 /home/rdchujf/DB_raw/HFQ_Index/Stk_Day_Idx 下的所有文件
    b。压缩文件解压到/home/rdchujf/DB_raw/HFQ_Index/Stk_Day_Idx
3. hfq文件：
    a。删除 /home/rdchujf/DB_raw/HFQ_Index/Stk_Day_FQ_WithHS 下的所有文件
    b。压缩文件解压到/home/rdchujf/DB_raw/HFQ_Index/Stk_Day_FQ_WithHS
二。更新 DBI
1.python DB_main.py Reset_DBI  #删除 DBI 里的 HFQ 和 index

2.python DB_main.py Initial_DBI  #把raw data 里的 HFQ 和 index 数据录入 DBI  
    要把DB_base.py 的 def __init__(self,Raw_lumpsum_End_DayI=20201130): 里的Raw_lumpsum_End_DayI 改到现在lumpsum data的最后一天

    生成 /home/rdchujf/n_workspace/data/RL_data/I_DB/Update_Log_HFQ_Index/lumpsum_HFQ_Inited_log.csv 
    该文件有两个作用
    a. 该文件的存在与否是是否 DBI index 和 HFQ init 的 标志
    b. 在 TSL_from_caculate 生成 stock list 时也用到
3.python DB_main.py Generate_DBTP TPVTest1 SH600000 20200601 20201131
  这一步是先把所有的raw data 都解压了， 否则 后面多进程同时解压一个文件会出各种莫名奇妙的错 比如 以下两种错误， 
  第一种是多次启动解压程序引起的
  ERROR: Can't allocate required memory!
  第二种是多次同时解压， 后面解压的output 和前面正在解压生成的文件冲突了 
  ERROR: ERROR: Can not delete output file : No such file or directory : /
  home/rdchujf/DB_raw/Normal/decompress/202006/20200612/000001.csv

3.python DB_main.py Generate_DBTP_Process TPVTest1 SLV1 4 True
  根据
  a. 最后的 True 只能 用在 SL_Definition.json 之定义一个 list， 否则 第二次process 启动会覆盖第一次的结果
  b. SLV1 的 SL_Definition.json 里定义的 train 、eval stock list 和 起始终止时间
  c.TPVTest1 里 定义的数据结构 生成 DBTP 数据
  d.log /mnt/data_disk2/n_workspace/data/RL_data/I_DB/Stock_List/SLV300/CreateLog里
  e. 结束后要查看 /mnt/data_disk/DB_raw/list_error_raw_fnwp.csv 文件， 这里 列的是DB_Base 中get_qz_df eception 的 raw 文件
     如果process terminate， 那么最后一个就是连ecemption 也处理不了的
     现在能处理的是两种情况
     1. 整数传成浮点数  如100  传成 99.99999999999999 或 100.0000000000001
     2. 有NaN 和 undefined
     
4.python DB_main.py Create_List_Stock_Fail_Generate_TPDB SLV300  
   根据 /home/rdchujf/n_workspace/data/RL_data/I_DB/Stock_List/SLV300/CreateLog里 的 error log 生成
   /home/rdchujf/n_workspace/data/RL_data/I_DB/Stock_List/SLV300/Adj_to_Remove.csv
   由于格式原因， 这个文件要打开后手工整理， 如有糊涂地方， 把每个error log 打开记下所有出错的股票代码

   不如如下：   
   Decompress File Not Found****/home/rdchujf/DB_raw/Normal/decompress/201807/20180720/000671.csv
   qz_mess = self.IRD.get_qz_df_inteface( Stock
   Not Enough Record

   这里面000671 也是出错的股票代码
   
   
关于stock list
1.create stock list 
    这两个选择只有 stock list 里  SL_Definition.json 定义了TSL_from_caculate 才和Initial_DBI 结果有关， 否则它们是独立的任何时间都能执行
    "Create_Total_SL SLV1",
    "Create_Sub_SL SLV1"
    DBTP 结果对它们的修正是在读取它们的list 时， 查找 Adj_to_Remove.csv （DBTP Error） 和 Price_to_Remove.csv （手工生成）

'''
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
    elif command == "Update_DBI":
        DayI=int(argv[1])
        i=DBI_init()
        flag, mess=i.Update_DBI_addon_HFQ_Index(DayI)
        print (flag, mess)
    elif command == "Generate_DBTP":
        DBTP_Name, Stock, StartS, EndS=argv[1:]
        i = DBTP_Creater(DBTP_Name)
        i. DBTP_generator(Stock, int(StartS), int(EndS))
    elif command == "Generate_DBTP_Process":
        DBTP_Name, SL_Name, NumPS,str_flag_overwrite= argv[1:]
        DBTP_main(DBTP_Name, SL_Name, int(NumPS),eval(str_flag_overwrite))
    elif command == "Create_Total_SL":
        SL_Name=argv[1]
        i=StockList(SL_Name)
        i.Get_Total_SL()
    elif command == "Create_Sub_SL":
        SL_Name=argv[1]
        i=StockList(SL_Name)
        i.generate_Train_Eval_SL()
    elif command == "Create_List_Stock_Fail_Generate_TPDB":
        SL_Name=argv[1]
        i=StockList(SL_Name)
        i.Get_Stocks_Error_Generate_DBTP()
    else:
        print ("Command {0} is not supported".format(command))
    print("Finished")
    return

if __name__ == '__main__':
    main(sys.argv[1:] )