from DBTP_Base import DBTP_Base
from DBI_Base import StockList,hfq_toolbox
from sklearn.preprocessing import minmax_scale,MinMaxScaler
import numpy as np
from collections import OrderedDict


class DBTP_Filters:
    def __init__(self,FT_TypesD_Flat):
        self.iHFQTool=hfq_toolbox()
        self.FT_TypesD_Flat =FT_TypesD_Flat

    def LV_NPrice(self,result_datas,HFQ_Ratio):
        FT_idx=0
        SelectedIdxs=[idx for idx, typeD in enumerate(self.FT_TypesD_Flat[FT_idx]) if typeD=="NPrice"]
        SelectedData=result_datas[FT_idx][:,SelectedIdxs]
        adj_HFQ_Ratio=[]
        for _ in range(len(SelectedIdxs)):
            adj_HFQ_Ratio.extend(HFQ_Ratio.tolist())
        SelectedDataHprice=self.iHFQTool.get_hfqprice_from_Nprice(SelectedData.reshape((-1,)),np.array(adj_HFQ_Ratio).reshape((-1,)))
        SelectedDataHpriceNorm = minmax_scale(SelectedDataHprice.reshape((-1,1)), feature_range=(0, 1), axis=0).ravel()
        adjSelectedData=SelectedDataHpriceNorm.reshape((-1,len(SelectedIdxs)))
        for adj_idx, source_idx in enumerate (SelectedIdxs):
            result_datas[FT_idx][:,source_idx] =adjSelectedData[:,adj_idx]
        return result_datas,HFQ_Ratio

    def LV_Volume(self,result_datas):
        FT_idx=0
        SelectedIdxs = [idx for idx, typeD in enumerate(self.FT_TypesD_Flat[FT_idx]) if typeD == "Volume"]
        assert len(SelectedIdxs)==0,"Volume type in lv  has not tested"
        return result_datas

    def LV_HFD(self,result_datas,HFQ_Ratio):
        FT_idx=0
        result_datas[FT_idx]=result_datas[FT_idx][:-1,:]
        new_HFQ_Ratio=HFQ_Ratio[:-1]
        return result_datas,new_HFQ_Ratio

    def LV_HFD_Sanity(self,result_datas):
        FT_idx = 0
        assert result_datas[FT_idx].shape==(19,15),result_datas[FT_idx].shape
        return result_datas


    def SV_NPrice(self,result_datas,HFQ_Ratio):
        FT_idx = 1
        HFQ_Ratio20_25 = []
        for ia in HFQ_Ratio:
            HFQ_Ratio20_25.extend([ia[0] for _ in range(result_datas[FT_idx].shape[1])])
        npHFQ_Ratio20_25=np.array(HFQ_Ratio20_25)
        assert len(self.FT_TypesD_Flat[FT_idx]) == result_datas[FT_idx].shape[2], "len(self.FT_TypesD_Flat[1])= {0} result_datas[1].shape= {1}".format(
            self.FT_TypesD_Flat[FT_idx],  result_datas[FT_idx].shape)

        SelectedIdxs = [idx for idx, typeD in enumerate(self.FT_TypesD_Flat[FT_idx]) if typeD == "NPrice"]
        assert len(SelectedIdxs)==1, "only support sv has one column Nprice type SelectedIdxs= {0} ".format(SelectedIdxs)
        SelectedData = result_datas[FT_idx][:,:,SelectedIdxs]
        SelectedDataHprice = self.iHFQTool.get_hfqprice_from_Nprice(SelectedData.reshape((-1,)),npHFQ_Ratio20_25)
        SelectedDataHpriceNorm = minmax_scale(SelectedDataHprice.reshape((-1, 1)), feature_range=(0, 1), axis=0).ravel()

        adjSelectedData=SelectedDataHpriceNorm.reshape((result_datas[FT_idx].shape[0],result_datas[FT_idx].shape[1]))
        for idx in list(range(result_datas[FT_idx].shape[0])):
            result_datas[FT_idx][idx,:,SelectedIdxs[0]]=adjSelectedData[idx]
        return result_datas,HFQ_Ratio


    def SV_Volume(self, result_datas):
        FT_idx = 1
        assert len(self.FT_TypesD_Flat[FT_idx]) == result_datas[FT_idx].shape[2], "len(self.FT_TypesD_Flat[1])= {0} result_datas[1].shape= {1}".format(
            self.FT_TypesD_Flat[FT_idx],  result_datas[FT_idx].shape)

        SelectedIdxs = [idx for idx, typeD in enumerate(self.FT_TypesD_Flat[FT_idx]) if typeD == "Volume"]
        assert len(SelectedIdxs)==1, "sv only support has one column Volume type"
        SelectedData = result_datas[FT_idx][:, :, SelectedIdxs]
        #SelectedDataVolume = minmax_scale(SelectedData.reshape((-1, 1)), feature_range=(0, 1), axis=0).ravel()
        SelectedDataVolume = minmax_scale(SelectedData.astype(float).reshape((-1, 1)), feature_range=(0, 1), axis=0).ravel()

        adjSelectedDataVolume = SelectedDataVolume.reshape((result_datas[FT_idx].shape[0], result_datas[FT_idx].shape[1]))
        for idx in list(range(result_datas[FT_idx].shape[0])):
            result_datas[FT_idx][idx, :, SelectedIdxs[0]] = adjSelectedDataVolume[idx]
        return result_datas

    def SV_HFD(self,result_datas,HFQ_Ratio):
        FT_idx=1
        a=result_datas[FT_idx][:,1:,:] # this is to remove the 9:25 part
        assert a.shape[1]%2==0
        a=a.reshape(a.shape[0]*2,a.shape[1]//2,a.shape[2] )
        result_datas[FT_idx]=a[:-1,:,:] # this is to remove lastday afternoon part
        l_new=[]
        for ratio in HFQ_Ratio:
            l_new.append(ratio)
            l_new.append(ratio)
        l_new.pop()  # pop the lastday  onece
        new_HFQ_Ratio=np.array(l_new)
        return result_datas,new_HFQ_Ratio


    def SV_HFD_Sanity(self,result_datas):
        FT_idx = 1
        assert result_datas[FT_idx].shape==(39,12,2),result_datas[FT_idx].shape
        return result_datas

    def AV_HFD(self,result_datas,HFQ_Ratio):
        FT_idx=2
        result_datas[FT_idx]=result_datas[FT_idx][:-1,:] # this is to remove lastday inform
        return result_datas,HFQ_Ratio


    def AV_HFD_Sanity(self,result_datas):
        FT_idx = 2
        assert result_datas[FT_idx].shape==(19,4),result_datas[FT_idx].shape
        return result_datas


    def convert_ratio_to_hash_category(self, ratio):
        assert ratio<=1.0 and ratio>=0.0
        params = np.array([0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        assert all((np.roll(params, -1) - params)[:-1]>0)
        assert 2 ** 4 >= len(params)
        return [int(x) for x in list('{0:04b}'.format((params <= ratio).sum() - 1))]

    def one_hot_base(self,ratio, params):
        assert ratio >= 0.0
        if ratio>1.0: ratio=1.0

        assert all((np.roll(params, -1) - params)[:-1] > 0)
        result = [0 for _ in range(len(params) )]
        try:
            result[(params <= ratio).sum() - 1] = 1
        except Exception as e:
            print(ratio)
            print((params <= ratio).sum())
            assert False
        return result

    def one_hot_V1(self, ratio):
        return self.one_hot_base(ratio, [0, 0.0001, 0.2, 0.4, 0.6, 0.8, 1])

    def one_hot_V2(self, ratio):
        return self.one_hot_base(ratio, [0, 0.0001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35, 0.4, 0.6, 0.8, 1])

    def LV_Together_base(self, result_datas, HFQ_Ratio, One_hot_fun):
        #result_datas [0] shape (days, numb_data)
        #HFQ_Ratio shape (days,, 1)
        FT_idx = 0
        # "NPrice", "Percent", "Volume", "Ratio"
        # init self.Dict_idxs_LV_TypesD
        if not hasattr(self, "LV_order_TypesD") or not hasattr(self, "Dict_idxs_LV_TypesD"):
            self.LV_order_TypesD = ['NPrice', 'Ratio', 'Percent', 'Flag_Tradable']
            assert set(self.FT_TypesD_Flat[FT_idx]) == set(self.LV_order_TypesD)
            np_FT_TypesD_Flat_0 = np.array(self.FT_TypesD_Flat[FT_idx])
            self.Dict_idxs_LV_TypesD = OrderedDict()
            for v in self.LV_order_TypesD:
                self.Dict_idxs_LV_TypesD[v] = np.where(np_FT_TypesD_Flat_0 == v)[0].tolist()

        data_price=result_datas[FT_idx][:,self.Dict_idxs_LV_TypesD['NPrice']]
        data_price_shape=data_price.shape
        Hprice = self.iHFQTool.get_hfqprice_from_Nprice(data_price, HFQ_Ratio)
        MMScaler = MinMaxScaler(feature_range=(0, 1))
        ra = MMScaler.fit_transform(Hprice.reshape(-1, 1)).reshape(data_price_shape)

        data_percent=result_datas[FT_idx][:,self.Dict_idxs_LV_TypesD['Ratio'] + self.Dict_idxs_LV_TypesD['Percent']]
        data_percent_shape=data_percent.shape
        data_percent=data_percent.reshape((-1,))
        percents = []
        for item in data_percent:
            percents.extend(One_hot_fun(item))
        rb = np.array(percents).reshape((data_percent_shape[0],-1))

        adj_result = np.concatenate([ra, rb, result_datas[FT_idx][:,self.Dict_idxs_LV_TypesD['Flag_Tradable']]],axis=1)
        result_datas[FT_idx] = adj_result

        return result_datas, HFQ_Ratio, self.Dict_idxs_LV_TypesD

    def LV_Together(self,result_datas,HFQ_Ratio):
        return self.LV_Together_base(result_datas,HFQ_Ratio, self.one_hot_V1)

    def LV_TogetherV2(self,result_datas,HFQ_Ratio):
        return self.LV_Together_base(result_datas,HFQ_Ratio, self.one_hot_V2)

    def SV_Together(self,result_datas,HFQ_Ratio):
        #result_datas [1] shape (days,  25_5M______241_1M,2)
        #HFQ_Ratio shape (days, 1)

        FT_idx=1
        Nprice = result_datas[FT_idx][:, :, [0]]
        Nprice_shape = Nprice.shape

        #Nprice_shape  (20,241,1) or (1,241,1) to *HFQ (20,1) or (1,1) reshape the Nprice to (20,241) or (1,241)

        Hprice = self.iHFQTool.get_hfqprice_from_Nprice(Nprice.reshape(Nprice_shape[0],Nprice_shape[1]), HFQ_Ratio)

        MMScaler = MinMaxScaler(feature_range=(0, 1))
        ra = MMScaler.fit_transform(Hprice.reshape(-1, 1)).reshape(Nprice_shape)

        original_data = result_datas[FT_idx][:,:, [1]]
        original_data_shape=original_data.shape
        adj_original_data=original_data[original_data > 10]  # has 1 in volume so set it at least 10
        original_data_min = 10 if len(adj_original_data)==0 else min(adj_original_data)  #10 is to avoid log (0) and log result too large
        MMScaler = MinMaxScaler(feature_range=(0, 1))
        rb = MMScaler.fit_transform(np.log(original_data+ original_data_min / 2).reshape(-1, 1)).reshape(original_data_shape)
        adj_result = np.concatenate([ra, rb], axis=-1)
        result_datas[FT_idx]=adj_result

        return result_datas,HFQ_Ratio,{}
