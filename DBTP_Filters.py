from DBTP_Base import DBTP_Base
from DBI_Base import DBI_init,StockList,hfq_toolbox
from sklearn.preprocessing import minmax_scale
import numpy as np

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
