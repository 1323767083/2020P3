from ftplib import FTP
#from DB_Base import DB_Base
import os

from ftplib import FTP
from DB_Base import DB_Base
import os

'''
#using following way in V70 local jupiter run the program
import sys
sys.path.append("D:\\user\\Hu\\workspace_gp\\2020P3")
import DB_FTP
i=DB_FTP.FTP_base()
i.local_get_qz_data(20210303)
i.local_get_HFQ_index(20210303)
'''

class FTP_base:
    data_server = "218.244.131.134"
    account_qz = "jiongfan_hu_stk_tbtqx"
    account_HFQ_index = "jiongfan_hu_stk_day_fqwithhs"
    password = 'jinshuyuan.net9vgf1ufi'

    local_dir="C:\\Users\\lenovo"

    def get_data(self, DateI, data_server, account, password, desbasedir, fn):
        desdir = os.path.join(desbasedir, str(DateI // 100))
        if not os.path.exists(desdir): os.mkdir(desdir)
        des_fnwp = os.path.join(desbasedir, fn)
        if os.path.exists(des_fnwp):
            print("Already Exisits {0}".format(des_fnwp))
            return True
        ftp = FTP(data_server)
        ftp.login(user=account, passwd=password)
        data = []
        ftp.dir(data.append)
        if len([True for item in data if item.split(" ")[-1] == fn]) != 1:
            print("Error in find the file {0}".format(fn))
            for item in data:
                print(item)
            return False
        total_size = ftp.size(fn)
        print("Start Downloading {0} with size {1}K".format(fn, total_size // 1000))
        self.downloaded_size = 0
        self.print_threahold = 0.1
        with open(os.path.join(desdir, fn), 'wb') as fh_local:
            def file_write_with_bar(data):
                fh_local.write(data)
                self.downloaded_size += len(data)
                finish_percent = self.downloaded_size / total_size
                if finish_percent > self.print_threahold:
                    print("Finish {0:.2f}".format(finish_percent))
                    self.print_threahold += 0.1
            ftp.retrbinary('RETR ' + fn, file_write_with_bar, 8196)  # Enter the filename to download
        ftp.quit()  # Terminate the FTP connection
        print("Downloaded to {0}".format(os.path.join(desdir, fn)))
        return True

    def local_get_qz_data(self, DateI):
        fn = "{0}.7z".format(DateI)
        return self.get_data(DateI, self.data_server, self.account_qz, self.password, self.local_dir, fn)

    def local_get_HFQ_index(self, DateI):
        fn = "{0}.rar".format(DateI)
        if not self.get_data(DateI, self.data_server, self.account_HFQ_index, self.password,self.local_dir, fn):
            return False
        return True

class Get_Data_After_closing(DB_Base, FTP_base):
    def __init__(self):
        DB_Base.__init__(self)

    def get_qz_data(self, DateI):
        fn = "{0}.7z".format(DateI)
        return self.get_data(DateI, self.data_server,self.account_qz, self.password, self.Dir_raw_normal_addon, fn)

    def get_HFQ_index(self, DateI):
        fn = "{0}.rar".format(DateI)
        if not self.get_data(DateI, self.data_server,self.account_HFQ_index, self.password, self.Dir_raw_Index_base_addon, fn):
            return False
