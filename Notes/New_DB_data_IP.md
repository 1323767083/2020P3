Data Structure
raw_DB Structure
YYYYMM/YYYYMMDD.7z->YYYYMM/YYYYMMDD/YYYY-MM-DD/XXXXXX.csv
E_YYYYMM/DD/XXXXXX.csv

I_DB Structure  ( I mean Inrermediate)

~/n_workspace/data/RL_data/I_DB/
    IV1/StockID/YM/day1.pickle
    IV1/StockID/YM//day2.pickle  
    IV1/StockID/Generate_Days.csv
    IV1/StockID/Missing_Days.csv
    IV1/I_Type_Definition.json
    #IV2/StockID/YM/day1.pickle
    #IV2/StockID/YM/day2.pickle
    #IV2/StockID/Generate_Days.csv
    #IV2/StockID/Missing_Days.csv
    #IV2/I_Type_Definition.json
    
~/n_workspace/data/RL_data/I_DB/    
    Daily_Summary/StockID/To_YYMMDD.csv
    Daily_Summary/StockID/Addon_YYMM.csv  

~/n_workspace/data/RL_data/I_DB/
    Iindex/IndexID/Daily_Summary/To_YYMMDD.csv    
    Iindex/IndexID/Daily_Summary/Addon_YYMM.csv
~/n_workspace/data/RL_data/I_DB/
    Raw_Source.jason


TP_DB Structure (TP mean Train and Predict)

~/n_workspace/data/RL_data/TP_DB/
    DatName(like V6)/StockID/YM/day1.pickle
    DatName(like V6)/StockID/YM/day2.pickle
    DatName(like V6)/StockID/Generate_Days.csv
    DatName(like V6)/StockID/Missing_Days.csv
~/n_workspace/data/RL_data/
    DatName(like V6)/TP_type_definition.json
~/n_workspace/data/RL_data/
    DatName(like V6)/SL_Train_YYMMDD(Start)_YYMMDD(End)_XX(number).csv
    DatName(like V6)/SL_Eval_YYMMDD(Start)_YYMMDD(End)_XX(number).csv
    
File Format
Raw_Source.jason
{"legacy":{
    "type":"lumpsum",
    "StartDate":"XXXXXX",
    "End_date":"XXXXXX",
    Source_dir:
    Class_to_handle:
    }
"normal":{
    "type":"lumpsum",
    "StartDate":"XXXXXX",
    "End_date":"XXXXXX",
    Source_dir:
    Class_to_handle:
    }  
"addon":{    
    "type":"addon",
    "StartDate":"XXXXXX",
    "End_date":"XXXXXX",
    Source_dir:
    Class_to_handle:
    }  
"lumpsum_support_inform":{
    "index_dir"
    "HFQ_dir"
}   
 

    *only one row could be addon
I_Type_Definition.csv
    {"IV1" :{"LV":
                {
                    "Title1","fun1",
                    "Title2","fun2"},
            "SV":
                {
                    "Title1","fun"},
            "Av":
                {"Title1","fun"}
            },
     }
    *name of fun  title_dimention1_(single_dimention) or  title_dimention1——dimention2(multiple dimention)    

Generate_Days.csv and Missing_Days.csv
    YYMMDD
    
TP_type_definition.csv
{"TPV2":{"LV":{"title1","IV1",
               "title2","IV2",  
               "title3","IV2",
               "title4","TPV1" 
               }
         {"SV":{"title1","IV1",
               "title2","IV2",  
               "title3","IV2",
               "title4","TPV1" 
               }
         {"AV":{"title1","IV1",
               "title2","IV2",  
               "title3","IV2",
               "title4","TPV1" 
               }