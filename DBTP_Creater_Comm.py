import numpy as np
class Memory_For_DBTP_Creater:
    def __init__(self,npMemTitlesM,iDBIs,max_len):
        self.npMemTitlesM=npMemTitlesM
        self.iDBIs=iDBIs
        self.max_len=max_len
        self.Reset()

    def Reset(self):
        self.memory = [[] for _ in range(len(self.npMemTitlesM))]
        self.daysI = []

    def get_np_item(self, TitleM, ShapeM):
        idx_found = np.where(self.npMemTitlesM == TitleM)
        assert len(idx_found[0]) == 1
        data = self.memory[idx_found[0][0]]
        assert self.Is_Ready(),"The get_data_item only be called either fnwp is exsist or the mx len reached in record"
        if len(ShapeM) == 1:
            assert len(data[0]) == ShapeM[0]
        elif len(ShapeM) == 2:
            assert len(data[0]) == ShapeM[0]
            assert len(data[0][0]) == ShapeM[1]
        else:
            assert False, "Not Support shape length more than 2 {0}".format(ShapeM)
        return np.array(data)

    def Is_Ready(self):
        return len(self.memory[0]) == self.max_len

    def Is_Last_Day(self, DayI):
        return self.daysI[-1] == DayI

    def Get_First_Day(self):
        return self.daysI[0]

    def Add(self, Stock, DayI):
        iDBIDatas = []
        for iDBI in self.iDBIs:
            flag,mess=iDBI.Generate_DBI_day( Stock, DayI)
            if not flag:
                self.Reset()
                return flag,mess
            Lresult = iDBI.load_DBI(Stock, DayI)
            iDBIDatas.extend(Lresult)
        assert len(iDBIDatas) == len(self.npMemTitlesM),"len(iDBIDatas)={0} len(self.np_TTL_DBITitlesM)={1}".\
            format(len(iDBIDatas),len(self.npMemTitlesM))

        flag_pop=self.Is_Ready()
        for idx, iDBIData in enumerate(iDBIDatas):
            if flag_pop:
                self.memory[idx].pop(0)
            self.memory[idx].append(iDBIData)
        if flag_pop:
            self.daysI.pop(0)
        self.daysI.append(DayI)
        return True, "Success Generate DBI or already Exists"
