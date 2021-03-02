class DBI_Generator_Base:
    ShapesM,Input_Params,TitilesD,TypesD = [],[],[],[]
    def Result_Check_Shape(self, result_item):
        if len(self.ShapesM) == 1:
            assert len(result_item) == self.ShapesM[0],"{0} {1} {2}".format(len(result_item), self.ShapesM, result_item)
        else:
            to_check = result_item
            for shape_item in self.ShapesM:
                assert len(to_check) == shape_item, "{0} {1} {2}".format(len(to_check), shape_item, to_check)
                to_check = to_check[0]
    def Gen(self, Inputs):
        return []
    def Get_TitleM(self):
        return self.__class__.__name__
    def Get_ShapesM(self):
        return self.ShapesM
    def Get_Input_Params(self):
        return self.Input_Params
    def Get_TitlesD(self):
        return self.TitilesD
    def Get_TypesD(self):
        return self.TypesD
