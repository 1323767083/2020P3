from Agent_Trader import *
class debug_strategy(Strategy_agent):
    def __init__(self, portfolio_name, strategy_name,experiment_name):
        Strategy_agent.__init__(self, portfolio_name, strategy_name,experiment_name)
        #debug
        self.i_RawData=RawData()

    def debug_load_df_a2e(self, DateI):
        fnwp_action2exe = self.iFH.get_a2e_fnwp(DateI)
        assert os.path.exists(fnwp_action2exe), "{0} does not exists".format(fnwp_action2exe)
        df_a2e = pd.read_csv(fnwp_action2exe)
        df_a2e = df_a2e.astype(self.a2e_types)
        df_a2e.set_index(["Stock"], drop=True, inplace=True,verify_integrity=True)
        print("Loaded a2e from ", fnwp_action2exe)
        return df_a2e

    def debug_sim(self, YesterdayI, DateI):
        df_a2e = self.debug_load_df_a2e(YesterdayI)
        df_a2e.to_csv(self.iFH.get_a2eDone_fnwp(DateI))
        df_aresult = pd.DataFrame(columns=self.aresult_Titles)
        # roughly buy 0.0003
        # roughly sell 0.0013
        for idx, row in df_a2e.iterrows():
            #print (row)
            stock, gu = row.name, row["Gu"]  # stock is index in Seris it is name
            flag, dfhqr, message = self.get_hfq_df(self.get_DBI_hfq_fnwp(stock))
            assert flag
            a = dfhqr[dfhqr["date"] == str(DateI)]
            if not a.empty:
                flag, dfqz, message = self.i_RawData.get_qz_df_inteface(row.name, DateI)
                assert flag
                for low, high in [[93000, 93500], [93500, 94000], [94000, 94500], [94500, 95000], [95000, 95500],
                                  [95500, 96000]]:
                    a = dfqz[(dfqz["Time"] >= 93000) & (dfqz["Time"] < 93500)]
                    if not a.empty: break
                num_trans = min(np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3]), len(a))
                NPrices = a["Price"].to_list()
                random.shuffle(NPrices)

                gu_avg = gu // num_trans
                l_Trans_Gu = [gu_avg if idx < num_trans - 1 else gu - (num_trans - 1) * gu_avg for idx in
                              list(range(num_trans))]
                l_Trans_Price = NPrices[:num_trans]
                if row["Action"] == "Buy":
                    for trans_Gu, trans_Price in zip(l_Trans_Gu, l_Trans_Price):
                        df_aresult.loc[len(df_aresult)] = [stock, "Buy", "Success", trans_Gu,trans_Gu * trans_Price * 1.0003, 0.0]
                elif row["Action"] == "Sell":
                    for trans_Gu, trans_Price in zip(l_Trans_Gu, l_Trans_Price):
                        df_aresult.loc[len(df_aresult)] = [stock, "Sell", "Success", 0, 0.0,trans_Gu * trans_Price * (1 - 0.0013)]
                else:
                    assert False, "Action only can by Buy or Sell not {0}".format(row["Action"])
            else:
                df_aresult.loc[len(df_aresult)] = [stock, row["Action"], "Tinpai", 0, 0.0, 0.0]
        df_aresult.to_csv(self.iFH.get_aresult_fnwp(DateI), index=False)
        return

    def debug_main(self, StartI, EndI):
        AStart_idx, AStartI = self.get_closest_TD(StartI, True)
        AEnd_idx, AEndI = self.get_closest_TD(EndI, False)

        assert AStartI <= AEndI
        self.start_strategy(AStartI)
        print ("Init strategy at ", AStartI)
        YesterdayI = AStartI
        period = self.nptd[AStart_idx + 1:AEnd_idx+1]
        for DateI in period:
            print("Run strategy at ", DateI)
            if DateI==AEndI:
                self.Sell_All(YesterdayI)
            self.debug_sim(YesterdayI, DateI)
            self.run_strategy(YesterdayI,DateI)
            YesterdayI = DateI



if __name__ == '__main__':
    debug_strategy("Portfolio_try1","Strategy_1", "experience1").debug_main(20201101, 20201110)