import os,sys
import patoolib
import shutil
import config as sc

class qz_extract_data:
    def __init__(self):
        self.rar_base_dir=sc.qz_rar_dir
        self.des_base_dir=sc.base_dir_qz_1
        self.des_base_dir_2 = sc.base_dir_qz_2

        self.convert_date_s = lambda date_s: "{0}-{1}-{2}".format(date_s[0:4], date_s[4:6], date_s[6:8])

    #----------------interface extract data--------------------------------------
    def extract_year_rar_data(self, year, from_str=""):  # year "2013" "2014  "2016 only six month"
        self.extract_year_rar_data_fun(year, self.des_base_dir,from_str=from_str)

    def extract_year_rar_data_2(self, year, from_str=""):  # year "2015"
        self.extract_year_rar_data_fun(year, self.des_base_dir_2,from_str=from_str)

    def extract_year_rar_data_fun(self, year, des_base_dir,from_str=""):
        #from_str is like from_str="201411SH.rar"  this is to resume the break operation
        # year="2013"
        #des_base_dir is "/home/rdchujf/Stk_qz" or "/home/rdchujf/Stk_qz_2"
        working_rar_dir = os.path.join(self.rar_base_dir, year)
        working_des_dir = os.path.join(des_base_dir, year)
        if not os.path.exists(working_des_dir): os.mkdir(working_des_dir)

        # rar_file_list=os.listdir(working_rar_dir)
        rar_file_list = [fn for fn in os.listdir(working_rar_dir) if fn.endswith("rar") and fn>=from_str]


        for fn in rar_file_list:
            src_fnwp = os.path.join(working_rar_dir, fn)
            print("extract {0} to {1}".format(src_fnwp, working_des_dir))
            patoolib.extract_archive(src_fnwp, outdir=working_des_dir)

    def generate_cmd_extract_day_7z_data_fun(self,year, des_base_dir,from_dir ):
        #7z x  -o/home/rdchujf/Downloads /home/rdchujf/Downloads/20180330.7z
        working_rar_dir = os.path.join(self.rar_base_dir, year)
        working_des_dir = os.path.join(des_base_dir, year)
        if not os.path.exists(working_des_dir): os.mkdir(working_des_dir)

        list_dir = [dir for dir in os.listdir(working_rar_dir)  if dir >= from_dir]
        error_log=[]
        cmd=[]
        for dir in list_dir:
            sub_dir=os.path.join(working_rar_dir,dir)
            list_7z_file=[fn for fn in os.listdir(sub_dir) if fn.endswith("7z") ]
            for fn in list_7z_file:
                src_fnwp = os.path.join(sub_dir, fn)
                if not os.path.exists(src_fnwp):
                    print("{0} not exists".format(src_fnwp))
                    error_log.append("{0} not exists".format(src_fnwp))
                else:
                    #print "extract {0} to {1}".format(src_fnwp, working_des_dir)
                #print "extract {0} to {1}".format(src_fnwp, working_des_dir)
                    #patoolib.extract_archive(src_fnwp, outdir=working_des_dir)
                    cmd.append("7z x  -o{0} {1}".format(working_des_dir,src_fnwp))
        print(";".join(cmd))

    def generate_cmd_extract_day_7z_data(self,year,from_dir=""):  #2016 second half :
        self.generate_cmd_extract_day_7z_data_fun(year, self.des_base_dir,from_dir )

    def generate_cmd_extract_day_7z_data_2(self,year,from_dir="" ):  #2017 till 20170428 :
        self.generate_cmd_extract_day_7z_data_fun(year, self.des_base_dir_2,from_dir)

    def generate_cmd_extract_day_7z_data_3(self,year,from_dir="2017-05" ):  #2017 >= 20170502 :
        working_rar_dir = os.path.join(self.rar_base_dir, year)
        working_des_dir = os.path.join(self.des_base_dir_2, year)
        if not os.path.exists(working_des_dir): os.mkdir(working_des_dir)

        list_dir = [dir for dir in os.listdir(working_rar_dir)  if dir >= from_dir]
        error_log=[]
        cmd=[]
        for dir in list_dir:
            sub_dir=os.path.join(working_rar_dir,dir)
            list_7z_file=[fn for fn in os.listdir(sub_dir) if fn.endswith("7z") ]
            for fn in list_7z_file:
                src_fnwp = os.path.join(sub_dir, fn)
                if not os.path.exists(src_fnwp):
                    print("{0} not exists".format(src_fnwp))
                    error_log.append("{0} not exists".format(src_fnwp))
                else:
                    #print "extract {0} to {1}".format(src_fnwp, working_des_dir)
                #print "extract {0} to {1}".format(src_fnwp, working_des_dir)
                    #patoolib.extract_archive(src_fnwp, outdir=working_des_dir)
                    fn_date_s=fn[0:8]
                    fn_date_with_dash=self.convert_date_s(fn_date_s)
                    sub_working_dir=os.path.join(working_des_dir,fn_date_with_dash)
                    if not os.path.exists(sub_working_dir): os.mkdir(sub_working_dir)
                    cmd.append("7z x  -o{0} {1}".format(sub_working_dir,src_fnwp))
        print(";".join(cmd))

    # ----------------interface correct after extract data--------------------------------------
    def solve_folder_under_folder(self):
        #"/home/rdchujf/Stk_qz_2/2017/2017-05-25/2017-05-25" shouldbe /home/rdchujf/Stk_qz_2/2017/2017-05-25/
        working_base_folder="/home/rdchujf/Stk_qz_2/2017"

        folder_list=[name for name in os.listdir(working_base_folder)  ]
        folder_list.sort()

        for folder in folder_list:
            test_folder=os.path.join(working_base_folder,folder,folder)
            if os.path.exists(test_folder):
                print("handling {0}".format(test_folder))
                old_working_folder=os.path.join(working_base_folder,folder)
                new_working_folder=os.path.join(working_base_folder,"{0}_tmp".format(folder))
                os.rename(old_working_folder,new_working_folder)
                src_folder=os.path.join(new_working_folder,folder)
                des_folder=old_working_folder
                shutil.move(src_folder,des_folder)
                shutil.rmtree(new_working_folder)

# this is for data after 20180101
class extract_20180101_qz_data:
    def __init__(self, src_dir, des_dir):
        #self.src_dir ="/mnt/backup_6G/data_per_trade_from20180101"
        #self.des_dir ="/mnt/backup_6G/Stk_qz_3"
        self.src_dir=src_dir
        self.des_dir=des_dir

        if not os.path.exists(self.des_dir): os.mkdir(self.des_dir)
        months_2018 = list(range(201801, 201813, 1))
        months_2019 = list(range(201901, 201908, 1))
        self.months=months_2018 + months_2019


    def extract_1month(self, month_to_extract):

        src_working_dir=os.path.join(self.src_dir, str(month_to_extract))
        lfn=[fn for fn in os.listdir(src_working_dir) if fn.endswith("7z")]
        lfn.sort()

        des_working_dir=os.path.join(self.des_dir, str(month_to_extract))
        if os.path.exists(des_working_dir): os.mkdir(des_working_dir)

        for fn in lfn:
            src_working_fnwp=os.path.join(src_working_dir, fn)
            patoolib.extract_archive(src_working_fnwp, outdir=des_working_dir)

    def extract(self):
        for month_to_extract in self.months:
            des_working_dir = os.path.join(self.des_dir, str(month_to_extract))
            if os.path.exists(des_working_dir):
                print(des_working_dir, "exists skip")
                continue
            self.extract_1month(month_to_extract)




class main():
    def __init__(self, argv):
        if argv[0]=="extract_20180101_qz_data":
            src_dir ="/mnt/backup_6G/data_per_trade_from20180101"
            des_dir ="/mnt/backup_6G/Stk_qz_3"
            extract_20180101_qz_data(src_dir,des_dir)
        else:
            print(("not support {0}".format(argv)))


if __name__ == '__main__':
    main(sys.argv[1:])
