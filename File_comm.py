import os,time
from pyinotify import WatchManager, Notifier, ProcessEvent

#IN_DELETE, IN_CREATE, IN_MODIFY

class EventHandler_debug(ProcessEvent):
    def process_IN_ACCESS(self, event):
        print "ACCESS event:", event.pathname

    def process_IN_ATTRIB(self, event):
        print "ATTRIB event:", event.pathname

    def process_IN_CLOSE_NOWRITE(self, event):
        print "CLOSE_NOWRITE event:", event.pathname

    def process_IN_CLOSE_WRITE(self, event):
        print "CLOSE_WRITE event:", event.pathname

    def process_IN_CREATE(self, event):
        print "CREATE event:", event.pathname

    def process_IN_DELETE(self, event):
        print "DELETE event:", event.pathname

    def process_IN_MODIFY(self, event):
        print "MODIFY event:", event.pathname

    def process_IN_OPEN(self, event):
        print "OPEN event:", event.pathname

'''
import T_config_common as sc
def fun_to_debug():
    import json
    from collections import OrderedDict
    fnwp = "/home/rdchujf/n_workspace/RL/try2/config.json"
    param = json.load(open(fnwp, "rb"), object_pairs_hook=OrderedDict)

    lgc = sc.config(param)
    import T_AC_brain2
    import imp
    imp.reload(T_AC_brain2)
    T_AC_brain2.T_AC_brain2_init_gc(lgc)
    i = T_AC_brain2.One_Brain2("One_brain_build_component")
    #i.save_server_models("/home/rdchujf/a.h5")
'''
def FSMonitor_debug(fun_to_debug,path="/home/rdchujf/a.h5"):
    wm = WatchManager()
    mask = 4095 #16|8#24 # according help IN_CLOSE_NOWRITE=16  mask = IN_CLOSE_NOWRITE
    i_eh=EventHandler_debug()
    notifier = Notifier(wm, i_eh)

    wd=wm.add_watch(path, mask, auto_add=True, rec=True)
    print wd
    print'now starting monitor %s' % (path)

    fun_to_debug()
    count=0
    while True:
        try:
            notifier.process_events()
            print "here"
            if notifier.check_events():
                count += 1
                print "count", count
                notifier.read_events()
                print "there1"
            else:
                print "there2"


        except  KeyboardInterrupt:
            #notifier.stop()
            break

    wm.rm_watch(wd[path])
    print'now stoping monitor %s' % (path)

    wm.add_watch("/home/rdchujf/a1.h5", mask, auto_add=True, rec=True)
    os.rename("/home/rdchujf/a.h5", "/home/rdchujf/a1.h5")

    count =0
    while True:
        try:
            notifier.process_events()
            if notifier.check_events():
                count += 1
                print "count", count
                notifier.read_events()
        except  KeyboardInterrupt:
            notifier.stop()
            break

'''
output
4095
{'/home/rdchujf/a.h5': 1}
now starting monitor /home/rdchujf/a.h5
2018-10-12 22:00:33.657731: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 860M, pci bus id: 0000:01:00.0)
here
count 1
there1
OPEN event: /home/rdchujf/a.h5
CLOSE_WRITE event: /home/rdchujf/a.h5
MODIFY event: /home/rdchujf/a.h5
OPEN event: /home/rdchujf/a.h5
MODIFY event: /home/rdchujf/a.h5
CLOSE_WRITE event: /home/rdchujf/a.h5
here
^Cnow stoping monitor /home/rdchujf/a.h5
2018-10-12 22:00:39.590361: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 860M, pci bus id: 0000:01:00.0)
count 1
^C
'''


class EventHandler_1(ProcessEvent):
    def __init__(self):
        ProcessEvent.__init__(self)
        self.reset()

    def reset(self):
        self.count_close_write=0
        self.count_modify=0
        self.count_open=0

    def process_IN_CLOSE_WRITE(self, event):
        self.count_close_write+=1
        #print "CLOSE_WRITE event:", event.pathname

    def process_IN_MODIFY(self, event):
        self.count_modify+=1
        #print "MODIFY event:", event.pathname

    def process_IN_OPEN(self, event):
        self.count_open +=1
        #print "OPEN event:", event.pathname

    def check_valid(self):
        if self.count_open==2 and self.count_modify==2 and self.count_close_write==2:
            return True
        else:
            return False


class check_model_save_finish_write:
    def __init__(self, time_out=600):
        self.wm = WatchManager()
        self.mask = 4095  # 16|8#24 # according help IN_CLOSE_NOWRITE=16  mask = IN_CLOSE_NOWRITE
        self.i_eh = EventHandler_1()
        self.notifier = Notifier(self.wm, self.i_eh)
        self.time_out=time_out #600 second means 10 minu


    def start_monitor(self, path):
        self.wd = self.wm.add_watch(path, self.mask, auto_add=True, rec=True)

    def stop_monitor(self,path):
        assert path in self.wd
        self.wm.rm_watch(self.wd[path])
        self.i_eh.reset()

    def wait_till_finish(self):
        starttime = time.time()
        while True:
            self.notifier.process_events()
            if self.i_eh.check_valid():
                #print self.i_eh.count_open, self.i_eh.count_modify, self.i_eh.count_close_write
                return True
            if self.notifier.check_events():
                self.notifier.read_events()
            if time.time() - starttime > self.time_out:
                return False


