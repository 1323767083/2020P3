import os, pickle,sys
import numpy as np
import pandas as pd
from DBTP_Base import DBTP_Base
from DBI_Base import StockList,hfq_toolbox

import DBI_Generator_LVs
import DBI_Generator_SVs
import DBI_Generator_Refs
from DBTP_Creater_Comm import Memory_For_DBTP_Creater

class DBTP_CreaterV2_get_Nprice:
    def __init__(self,Swith_BuyvsSell):
        self.iM15DBIGen=DBI_Generator_Refs.Potential_Nprices_M15()
        #print (self.iM15DBIGen.TitilesD)
        #print (np.where(self.iM15DBIGen.TitilesD =="Flag_Tradable"))
        #self.flag_idx=np.where(self.iM15DBIGen.TitilesD =="Flag_Tradable")[0][0]
        self.flag_idx =self.iM15DBIGen.TitilesD.index("Flag_Tradable")

        assert Swith_BuyvsSell in ["Buy", "Sell"]
        self.buffidx= 1 if Swith_BuyvsSell=="Buy" else 2

    def get_Nprice_on_operationtime(self,buff, OperationTime): #OperationTime=94500
        #self.price_idx = np.where(self.iM15DBIGen.TitilesD == f"Potential_NPrice_{OperationTime}")[0][0]
        price_idx = self.iM15DBIGen.TitilesD.index(f"Potential_NPrice_{OperationTime}")
        PNprices_M15s = buff.get_np_item("Potential_Nprices_M15", [17])
        Nprice=float("NaN")
        if PNprices_M15s[self.buffidx][self.flag_idx]:
            for M15Priceidx in list(range(price_idx, len(self.iM15DBIGen.TitilesD))):
                if PNprices_M15s[self.buffidx][M15Priceidx] == PNprices_M15s[self.buffidx][M15Priceidx]:  # need choose TinpaiNAN in DBI_Definiotn
                    Nprice = PNprices_M15s[self.buffidx][M15Priceidx]
                    return Nprice
            else:
                return float("NaN")
        else:
            return float("NaN")

    def get_average_Nprice(self, buff, OperationTimes):
        assert len(OperationTimes)>0
        l_Nprice=[]
        for OperationTime in OperationTimes:
            Nprice=self.get_Nprice_on_operationtime(buff, OperationTime)
            if Nprice!=Nprice:
                return float("NaN")
            l_Nprice.append(Nprice)
        return sum(l_Nprice) / len(l_Nprice)

    def get_tradable_flag(self,buff):
        PNprices_M15s = buff.get_np_item("Potential_Nprices_M15", [17])
        return  PNprices_M15s[1][self.flag_idx]

    def get_HFQ_ratio(self,buff):
        HFQs = buff.get_np_item("HFQ_Ratio", [1])
        return HFQs[self.buffidx]
class DBTP_CreaterV2(DBTP_Base):
    def __init__(self, DBTP_Name,Stocks,stock_list_name):
        DBTP_Base.__init__(self,DBTP_Name)
        self.Stocks=Stocks
        self.stock_list_name=stock_list_name
        self.BuyAtTimes=self.DBTP_Definition["V2Param"]["BuyAtTimes"]
        self.SellAtTimes=self.DBTP_Definition["V2Param"]["SellAtTimes"]
        self.Profit_Scale=self.DBTP_Definition["V2Param"]["Profit_Scale"]
        assert all( [iDBI.TypeDefinition["TinpaiFun"]=="TinpaiNAN" for iDBI in self.iDBIs])

        self.buffs=[Memory_For_DBTP_Creater(self.npMemTitlesM, self.iDBIs,self.NumDBIDays) for _ in self.Stocks]
        self.ibuyNprice = DBTP_CreaterV2_get_Nprice("Buy")
        self.isellNprice = DBTP_CreaterV2_get_Nprice("Sell")

        self.iHFQTool=hfq_toolbox()
    def Generate_DBTP_Data(self, DayI):
        fnwp = self.get_DBTP_data_fnwpV2(DayI)
        if os.path.exists(fnwp):
            print("DBTP from DBI {0} already exists ".format(DayI))
            return False  # means already exist
        assert all([buff.daysI[0]==DayI for buff in self.buffs])

        result_table=np.zeros((len(self.Stocks),))
        for buffidx,buff in enumerate(self.buffs):
            if self.ibuyNprice.get_tradable_flag(buff)  and self.isellNprice.get_tradable_flag(buff):
                buy_Nprice_average  =   self.ibuyNprice.get_average_Nprice(buff,self.BuyAtTimes)
                sell_Nprice_average =   self.isellNprice.get_average_Nprice(buff,self.SellAtTimes)
                if buy_Nprice_average!=buy_Nprice_average or sell_Nprice_average!=sell_Nprice_average:
                    result_table[buffidx] = 0

                buy_HFQ= self.ibuyNprice.get_HFQ_ratio(buff)
                sell_HFQ= self.isellNprice.get_HFQ_ratio(buff)
                result_table[buffidx]=(hfq_toolbox().get_hfqprice_from_Nprice(sell_Nprice_average,sell_HFQ)/
                                      hfq_toolbox().get_hfqprice_from_Nprice(buy_Nprice_average, buy_HFQ)
                                      -1)*self.Profit_Scale

            else:
                result_table[buffidx]=0
        pickle.dump(result_table, open(fnwp, "wb"))
        print("DBTP from DBI {0} success generated".format(DayI))
        return True

    def DBTP_generator(self, StartI, EndI):
        AStart_idx, AStartI=self.get_closest_TD(StartI, True)
        AEnd_idx, AEndI = self.get_closest_TD(EndI, False)
        if AStartI<=AEndI:
            assert AStart_idx>self.NumDBIDays-1
            period=self.nptd[AStart_idx-(self.NumDBIDays-1):AEnd_idx+1]
        else:
            print("{0},{1}".format(StartI, "No trading day between {0} {1}".format(StartI, EndI)), file=sys.stderr)
            return False, "No trading day between {0} {1}".format(StartI, EndI)

        for DayI in period:
            logfnwp = self.get_DBTP_data_log_fnwpV2(DayI, self.stock_list_name)
            for idx,Stock in enumerate(self.Stocks):
                flag, mess=self.buffs[idx].Add(Stock,DayI)
                if not flag:
                    self.log_append_keep_new([[flag,Stock,mess]], logfnwp, ["Result", "Stock", "Message"],unique_check_title="Stock")
                    #self.buff.Reset()   #already called in Add dicontinues in self.memory , so should reset meomory
                    print("{0},{1},{2}".format(Stock,DayI,mess ), file=sys.stderr)
                    continue  # need add other stock
                else:
                    self.log_append_keep_new([[flag, Stock, mess]], logfnwp, ["Result", "Stock", "Message"],unique_check_title="Stock")
            flags_buff_ready=[buff.Is_Ready() for buff in self.buffs]
            if not all(flags_buff_ready):
                assert not any(flags_buff_ready)
                self.log_append_keep_new([[False, "AllStock","Not_Enough_Record_to_generate_DBTP"]], logfnwp, ["Result", "Stock", "Message"],unique_check_title="Stock")
                print("{0},{1}".format( DayI, "Not Enough Record"), file=sys.stderr)
                continue


            assert all ([buff.Is_Last_Day(DayI) for buff in self.buffs])
            flag_return = self.Generate_DBTP_Data(self.buffs[0].Get_First_Day())
            self.log_append_keep_new([[True, "AllStock", "Generate _to_generate_DBTP" if flag_return else "Already Exists DBTP"]], logfnwp, ["Result", "Stock", "Message"],unique_check_title="Stock")

        for buff in self.buffs:
            buff.Reset()
        return True, "Success"
