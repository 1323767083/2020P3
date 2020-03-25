

#Introduced
##data_T5.py introduced:
* stock
* date
* last_day_flag
* this_trade_day_Nprice
* this_trade_day_hfq_ratio
* stock_SwhV1       #this is used by class simulator; not used by Simulator_LHPP2V2 and Simulator_LHPP2V3 

##env_get_data.py introduced
* last_day_flag,
* flag_all_period_explored (LHP  only used in eval to end the eval for specific ET)
* period_idx  (LHP only used for recorder to compress un compress state)
* idx_in_period (LHP only used for recorder to compress un compress state)

##env.py introduced
* action_taken
* action_return_message
* holding
* potential_profit
* flag_force_sell

##Buffer_comm.py introduced
* _support_view_dic  #this is used by record to recover data
* SdisS_

##A3C_workers introduced
* old_ap

#A3C_workers cleaned before send to server
* kept
    * stock
    * date
    * action_return_message
    * action_taken
    * holding
    * potential_profit
    * idx_in_period
    * period_idx
    * old_ap
* droped
    * last_day_flag
    * this_trade_day_Nprice
    * this_trade_day_hfq_ratio
    * stock_SwhV1
