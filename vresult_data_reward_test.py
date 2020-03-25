def test_read_are():
    import vresult_data_com as c
    import vresult_data_reward as m
    system_name = "tryB3"
    eval_process_name = "Eval_0"
    Lstock, LEvalT, LYM, lgc = c.get_addon_setting(system_name, eval_process_name)
    i = m.ana_reward_data(system_name, eval_process_name, Lstock, LEvalT, LYM, lgc)
    stock = "SH600009"
    evalT = 3000
    flag_opt, df = i._read_stock_are(stock, evalT)
