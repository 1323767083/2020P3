
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
       0.### 注意不要同时运行两个 上面的指令， 比如
        python DB_main.py Generate_DBTP_Process DBTP_1M1D SL300V1 20 True
        python DB_main.py Generate_DBTP_Process DBTP_1M1D SL500V1 20 True
        SL300V1 和 SL500V1 有重叠的股票， 会导致DBI 的 log （。csv） 同时写而出错
        出错信息：
            Process Process_Generate_DBTP-18:
                Traceback (most recent call last):
                  File "/home/rdchujf/anaconda3/envs/p37/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
                    self.run()
                  File "/home/rdchujf/remote_sw/DBTP_Creater.py", line 206, in run
                    flag, mess=self.iDBTP_Creater.DBTP_generator(Stock, self.StartI, self.EndI)
                  File "/home/rdchujf/remote_sw/DBTP_Creater.py", line 151, in DBTP_generator
                    flag, mess=self.buff.Add(Stock,DayI)
                  File "/home/rdchujf/remote_sw/DBTP_Creater.py", line 46, in Add
                    flag,mess=iDBI.Generate_DBI_day( Stock, DayI)
                  File "/home/rdchujf/remote_sw/DBI_Creater.py", line 149, in Generate_DBI_day
                    self.log_append_keep_new([[True, DayI, "Success" + "Generate" ]], logfnwp, ["Result", "Date", "Message"])
                  File "/home/rdchujf/remote_sw/DB_Base.py", line 197, in log_append_keep_new
                    dfo.sort_values(by=[unique_check_title],inplace=True)
                  File "/home/rdchujf/anaconda3/envs/p37/lib/python3.7/site-packages/pandas/core/frame.py", line 4933, in sort_values
                    k, kind=kind, ascending=ascending, na_position=na_position
                  File "/home/rdchujf/anaconda3/envs/p37/lib/python3.7/site-packages/pandas/core/sorting.py", line 274, in nargsort
                    indexer = non_nan_idx[non_nans.argsort(kind=kind)]
                TypeError: '<' not supported between instances of 'int' and 'str'
        出错处理：
            根据Process Process_Generate_DBTP-18 找到 process17 Output log 最后一行股票， 到DBI 该股票的 log 中删掉最后一行
            然后把17stocklist 剩下的股票以 f 中讲的补全 
        
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
        DBTP 结果对它们的修正是在读取它们的list 时， 查找 Adj_to_Remove.csv （DBTP Error） 和 Price_to_Remove.csv （Get_SL_Exceed_MaxPrice）
    
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

DBI 里增加指数
    第一步 在/mnt/pdata_disk2Tw/RL_data_additional/index/ 生成指数文件：
        from DB_main_fun import Add_new_index_to_DBI
        l_df=Add_new_index_to_DBI(["SH000300", "SH000905"])
    第二步 修改DB_Base.py 把指数加入 DBI_Index_Code_List
        DBI_Index_Code_List=["SH000001","SZ399001""SH000300","SH000905"]   #SH000905 is 500