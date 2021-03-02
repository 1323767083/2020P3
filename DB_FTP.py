from ftplib import FTP
#from DB_Base import DB_Base
import progressbar
import os

class Get_Data_After_closing:
    data_server = "218.244.131.134"
    account_qz = "jiongfan_hu_stk_tbtqx"
    account_HFQ_index = "jiongfan_hu_stk_day_fqwithhs"
    password = 'jinshuyuan.net9vgf1ufi'
    def __init__(self):
        #DB_Base.__init__(self)

        #self.Dir_raw_normal_addon
        #self.Dir_raw_HFQ_base_addon
        #self.Dir_raw_Index_base_addon

        self.Dir_raw_normal_addon="C:\\Users\\lenovo"
        #self.Dir_raw_HFQ_base_addon
        self.Dir_raw_Index_base_addon="C:\\Users\\lenovo"



    def get_data(self, DateI, account, password, desbasedir, fn):
        monthI=DateI//100
        desdir=os.path.join(desbasedir, str(monthI))
        if not os.path.exists(desdir): os.mkdir(desdir)
        #srcfn="{0}.rar".format(DateI)

        ftp = FTP(self.data_server)
        ftp.login(user=account, passwd=password)

        data = []
        ftp.dir(data.append)
        if len([True for item in data if item.split(" ")[-1]==fn])!=1:
            print ("Error in find the file {0}".format(fn))
            for item in data:
                print(item)
            return False

        size = ftp.size(fn)


        print ("Start Downloading {0}".format(fn))
        self.pbar = progressbar.ProgressBar(widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()], maxval=size)
        self.pbar.start()
        self.downloaded_size=0
        with open(os.path.join(desdir,fn), 'wb') as fh_local:
            def file_write_with_bar(data):
                fh_local.write(data)
                self.downloaded_size+=1024
                self.pbar.update(self.downloaded_size)
            ftp.retrbinary('RETR ' + fn, file_write_with_bar, 1024)  # Enter the filename to download
        ftp.quit() # Terminate the FTP connection
        print ("Downloaded to {0}".format(os.path.join(desdir,fn)))
        return True

    def get_qz_data(self, DateI):
        fn = "{0}.7z".format(DateI)
        return self.get_data(DateI, self.account_qz, self.password, self.Dir_raw_normal_addon, fn)

    def get_HFQ_index(self, DateI):
        fn = "{0}.rar".format(DateI)
        if not self.get_data(DateI, self.account_HFQ_index, self.password, self.Dir_raw_Index_base_addon, fn):
            return False
