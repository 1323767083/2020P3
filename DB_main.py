import sys,os
import pandas as pd
from DB_Base import DB_Base
from DBI_Base import DBI_init,StockList
from DBTP_Creater import DBTP_Creater,DBTP_creator_on_SLperiod,DBTP_creator
from DB_FTP import Get_Data_After_closing

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

    3. 加入停盘的HFQ
        TUSHARE查 当天停盘的list
            pro = ts.pro_api()
            df = pro.suspend(ts_code='', suspend_date='20210226', resume_date='', fields='')

       然后通过 DB_Base 里的read df 读，去除duplicated 后，写入/home/rdchujf/n_workspace/data/RL_data/I_DB/HFQ
           stocks=["SZ000976","SH600687","SZ002071","SH600978","SH600247"]
    
            raw_hfq_base_dn="/mnt/data_disk/DB_raw/HFQ_Index/Stk_Day_FQ_WithHS/"
            DBI_hfq_base_dn="/mnt/pdata_disk2Tw/RL_data_additional/HFQ/"
            
            for stock in stocks:
                raw_fnwp=os.path.join(raw_hfq_base_dn,"{0}.csv".format(stock))
                DBI_fnwp=os.path.join(DBI_hfq_base_dn,"{0}.csv".format(stock))
                flag,dfraw, mess=i.get_hfq_df(raw_fnwp)
                assert flag,mess
                df=dfraw
                df.drop_duplicates(subset=["date"],inplace=True)
                df.reset_index(inplace=True,drop=True)
                df.to_csv(DBI_fnwp, index=False)
                print (DBI_fnwp)
        手工改update log    /home/rdchujf/n_workspace/data/RL_data/I_DB/Update_Log_HFQ_Index/lumpsum_HFQ_Inited_log.csv
        
        生成后手工加入的HFQ
            part 1 因为文件名sh 和 sz 小写， 所以被程序认为不存在
            sh603967.csv
            sz300766.csv
            sz300769.csv
            sz300768.csv
            sz300771.csv
            sz002950.csv
            sz300772.csv
            sh603317.csv
            sz300770.csv
            sz300773.csv
            sh603068.csv
            
            
            part 2  因为20210226 停盘， 所以当天qz 数据没有， 而且在20210310 手工改的时候还未复盘
            
            SZ000976 False HFQ Not Have Lumpsum End****SZ000976 not have 20210226 data
            SH600687 False HFQ Not Have Lumpsum End****SH600687 not have 20210226 data
            SZ002071 False HFQ Not Have Lumpsum End****SZ002071 not have 20210226 data
            SH600978 False HFQ Not Have Lumpsum End****SH600978 not have 20210226 data
            SH600247 False HFQ Not Have Lumpsum End****SH600247 not have 20210226 data
            
            part 3 因为20210226 停盘， 所以当天qz 数据没有， 但是在20210310 前已经复盘， 所以addon HFQ有， 20210226前面HFQ要加入
            SZ000603 True Success
            SZ000803 True Success
            SZ300949 True Success
            SZ000032 True Success
            SZ002024 True Success
        
            只有这两个股票数据是 500 和 300的
            {'SH603317'} 500
            {'SZ002024'} 300
    3.python DB_main.py Generate_DBTP TPVTest1 SH600000 20200601 20201131
      这一步是先把所有的raw data 都解压了， 否则 后面多进程同时解压一个文件会出各种莫名奇妙的错 比如 以下两种错误， 
      第一种是多次启动解压程序引起的
      ERROR: Can't allocate required memory!
      第二种是多次同时解压， 后面解压的output 和前面正在解压生成的文件冲突了 
      ERROR: ERROR: Can not delete output file : No such file or directory : /
      home/rdchujf/DB_raw/Normal/decompress/202006/20200612/000001.csv
    
    4.python DB_main.py Generate_DBTP_Process TPVTest1 SLV300 4 True
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
     
       f: 如果有进程中间出错中断， 修补方法：
           以Output 的最后修改时间来确定哪个进程， 哪个文件  一般 打印出错的process 代号比文件的标号大 1
           找到原因并解决
           从相应的 SL 里找到后面没处理的stock
           类似以下的行命令一个个生成， 并保留log            

                python DB_main.py Generate_DBTP DBTP_5MV1 SH600176 20180101 20210226 >/home/rdchujf/t.txt
                python DB_main.py Generate_DBTP DBTP_5MV1 SH600177 20180101 20210226 >>/home/rdchujf/t.txt
                python DB_main.py Generate_DBTP DBTP_5MV1 SH600183 20180101 20210226 >>/home/rdchujf/t.txt
                python DB_main.py Generate_DBTP DBTP_5MV1 SH600188 20180101 20210226 >>/home/rdchujf/t.txt
                python DB_main.py Generate_DBTP DBTP_5MV1 SH600196 20180101 20210226 >>/home/rdchujf/t.txt
                python DB_main.py Generate_DBTP DBTP_5MV1 SH600208 20180101 20210226 >>/home/rdchujf/t.txt
                
            然后拼装 log
                cat /home/rdchujf/t.txt >>Process2Output.txt
     
    5.python DB_main.py Create_List_Stock_Fail_Generate_TPDB SLV300  
       根据 /home/rdchujf/n_workspace/data/RL_data/I_DB/Stock_List/SLV300/CreateLog里 的 error log 生成
       /home/rdchujf/n_workspace/data/RL_data/I_DB/Stock_List/SLV300/Adj_to_Remove.csv
       由于格式原因， 这个文件要打开后手工整理， 如有糊涂地方， 把每个error log 打开记下所有出错的股票代码
    
       不如如下：   
       Decompress File Not Found****/home/rdchujf/DB_raw/Normal/decompress/201807/20180720/000671.csv
       qz_mess = self.IRD.get_qz_df_inteface( Stock
       Not Enough Record
    
       这里面000671 也是出错的股票代码
            use following command instead 
            SLV300_TPV3/CreateLog$ grep "SZ" *Error.txt
            SLV300_TPV3/CreateLog$ grep "SH" *Error.txt
update Addon data
    1. FTP
        a. to homeserver
            python DB_main.py Addon_Download 20210305
        b. to local then to home server
    
            #using following way in V70 local jupiter run the program
            import sys
            sys.path.append("D:\\user\\Hu\\workspace_gp\\2020P3")
            import DB_FTP
            i=DB_FTP.FTP_base()
            i.local_get_qz_data(20210303)
            i.local_get_HFQ_index(20210303)
            
            存在 C:\\Users\\lenovo\\202103
    
    2. Addon_Update_DBI
        python DB_main.py Addon_Update_DBI 20210305
    3. generate DBTP for the date on SL_name
        python DB_main.py Addon_Generate_DBTP DBTP_Name, SL_Name, SL_tag, SL_idx, StartI, EndI, NumP, flag_overwrite
        现阶段：
        python DB_main.py Addon_Generate_DBTP TPV5_10M_5MNprice SLV500_10M Train 0 20210303 20210305 10 True
        python DB_main.py Addon_Generate_DBTP TPVTest1 SLV300 Train 0 20210301 20210305 10 True
        
        use
        ls -l *Error.txt check result
    4. Addon_In_One
        python DB_main.py Addon_In_One Config_name DateI Screen_or_file   # True means Screen
        
        Config_name.csv format
        DBTP_Name, SL_Name, SL_tag, SL_idx
        
        Example:
        python DB_main.py Addon_In_One DBTP1M5M10MSL300SL500 20210310 False      # False 是输入文件不在屏幕上显示
    5. Addon_Update_To_Date
        python DB_main.py Addon_Update_To_Date Config_name
        Example:
        python DB_main.py Addon_Update_To_Date DBTP1M5M10MSL300SL500
   
关于stock list
    1.create stock list 
        这两个选择只有 stock list 里  SL_Definition.json 定义了TSL_from_caculate 才和Initial_DBI 结果有关， 否则它们是独立的任何时间都能执行
        "Create_Total_SL SLV1",
        "Create_Sub_SL SLV1"
        DBTP 结果对它们的修正是在读取它们的list 时， 查找 Adj_to_Remove.csv （DBTP Error） 和 Price_to_Remove.csv （手工生成）
    
    添加新的element DBI和DBTP
    1. 在DBI_Creater 里建立新的element函数
        Norm_Average_Nprice_And_Mount_Whole_Day_1M in DBI_Creater
    2. 在 I_DB 和 TP_DB 里建立 目录 和json file （如考虑硬盘大小， 可以是目录链接）
        SV1M 目录下 DBI_Definition.json
        TPV3 目录下 DBTP_Definition.json
    3. 单个股票试
        python DB_main.py Generate_DBTP TPV3 SH600000 20200601 20201131
    4. 按stock list 建立 DBTP
        python DB_main.py Generate_DBTP_Process TPV3 SLV300_TPV3 30 True
        #注意SLV300 配置文件里的时间段信息，不同的DBTP 要用不同的 stocklist名字 即使是同样的stocklist
    5. python DB_main.py Create_List_Stock_Fail_Generate_TPDB SLV300_TPV3
            use following command instead 
            SLV300_TPV3/CreateLog$ grep "SZ" *Error.txt
            SLV300_TPV3/CreateLog$ grep "SH" *Error.txt
    6. python DB_main.py Get_SL_Exceed_MaxPrice SLV1 500
            Generate Price_to_Remove.csv for stock price exceed 500

添加新的DBTP可以通过添加DBI element， 也可以通过在 Filter list 增加filter 来 调整数据
    例如
    TPHFD：的DBTP_Definition.json
    {
        "DataFromDBI":
        {
            "LV":{
                "VTest1":["Price_VS_Mount", "Sell_Dan", "Buy_Dan","Exchange_Ratios"]
            },
            "SV":{
                "VTest1":["Norm_Average_Nprice_And_Mount_Whole_Day"]
            },
            "Reference":{
                "VTest1": ["DateI","HFQ_Ratio"],
                "NPrice1300_I5":["Potential_Nprice_1300"]
             }
        },
        "Param":
        {
            "LV": {
                "Filters_In_order":["LV_HFD","LV_NPrice","LV_Volume","LV_HFD_Sanity"]
            },
            "SV": {
                "Filters_In_order":["SV_HFD","SV_NPrice","SV_Volume","SV_HFD_Sanity"]
            },
            "Reference": {
                "Filters_In_order":["AV_HFD","AV_HFD_Sanity"]
            }
    
        }
    }
    
    注意这些filter的函数 都要返回的


'''

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
    "Addon_Update_To_Date Config_name"
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
        i = DBTP_Creater(DBTP_Name)
        i. DBTP_generator(Stock, int(StartS), int(EndS))
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
        Addon_IN_One_dnwp = "/home/rdchujf/DB_raw_addon/Config"
        param_fnwp = os.path.join(Addon_IN_One_dnwp, "{0}.csv".format(argv[1]))
        df = pd.read_csv(param_fnwp)
        TPDB_update_log_base_dir="/home/rdchujf/n_workspace/data/RL_data/TP_DB/Update_Log"
        UpdateToDayIs=[]
        for idx, row in df.iterrows():
            dnwp=os.path.join(TPDB_update_log_base_dir,row["DBTP_Name"],"{0}_{1}_{2}".
                              format(row["SL_Name"], row["SL_tag"], int(row["SL_idx"])))
            UpdateToDayIs.append(max([int(dn[9:])for dn in os.listdir(dnwp)]))
        lastdayIs=list(set(UpdateToDayIs))
        if len(lastdayIs)==1:
            print ("Update to {0}".format(lastdayIs[0]))
        else:
            print ("can not decide in {0}, please check following folders:".format(lastdayIs))
            for idx, row in df.iterrows():
                print(os.path.join(TPDB_update_log_base_dir,row["DBTP_Name"],"{0}_{1}_{2}".
                                   format(row["SL_Name"], row["SL_tag"], int(row["SL_idx"]))))

    elif command=="Addon_In_One":
        Addon_IN_One_dnwp="/home/rdchujf/DB_raw_addon/Config"
        param_fnwp=os.path.join (Addon_IN_One_dnwp, "{0}.csv".format(argv[1]))
        DateI=eval(argv[2])
        flag_Print_on_screen_or_file=eval(argv[3])

        from contextlib import redirect_stdout, redirect_stderr
        if flag_Print_on_screen_or_file:
            newstdout = sys.__stdout__
            newstderr = sys.__stderr__
            stdoutfnwp,stderrfnwp="",""
        else:
            Addon_IN_One_log_dnwp=Addon_IN_One_dnwp
            for subdir in [argv[1],str(DateI)]:
                Addon_IN_One_log_dnwp=os.path.join(Addon_IN_One_log_dnwp, subdir)
                if not os.path.exists(Addon_IN_One_log_dnwp): os.mkdir (Addon_IN_One_log_dnwp)
            stdoutfnwp=os.path.join(Addon_IN_One_log_dnwp,"Output.txt")
            stderrfnwp=os.path.join(Addon_IN_One_log_dnwp,"Error.txt")
            print ("Output will be direct to {0}".format(stdoutfnwp))
            print ("Error will be direct to {0}".format(stderrfnwp))
            newstdout = open(stdoutfnwp, "w")
            newstderr = open(stderrfnwp, "w")

        with redirect_stdout(newstdout), redirect_stderr(newstderr):

            i = Get_Data_After_closing()
            if not i.get_qz_data(DateI):
                print ("Fail in downloading qz for {0}".format(DateI))
                return
            else:
                print("Success downloading qz for {0}".format(DateI))
            if not i.get_HFQ_index(DateI):
                print("Fail in downloading HFQ_index for {0}".format(DateI))
                return
            else:
                print("Success downloading HFQ_index for {0}".format(DateI))

            i=DBI_init()
            flag, mess=i.Update_DBI_addon(DateI)
            print ("Success" if flag else "Fail", "  ",mess)

            df = pd.read_csv(param_fnwp)
            for idx, row in df.iterrows():
                print (row["DBTP_Name"], row["SL_Name"], row["SL_tag"], int(row["SL_idx"]))
                DBTP_creator(row["DBTP_Name"], row["SL_Name"], row["SL_tag"], int(row["SL_idx"]), DateI, DateI, 10, False)
        if not flag_Print_on_screen_or_file:
            print ("Output will be direct to {0}".format(stdoutfnwp))
            print ("Error will be direct to {0}".format(stderrfnwp))

    else:
        print ("Command {0} is not supported".format(command))
    print("Finished")
    return

if __name__ == '__main__':
    main(sys.argv[1:] )