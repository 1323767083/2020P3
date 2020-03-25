import os, time


class lconfig:
    def __init__(self):
        self.system_working_dir=""
        self.command_pipe_seed=""
def init_gc(lgc):
    global lc
    lc=lconfig()
    for key in lc.__dict__.keys():
        lc.__dict__[key] = lgc.__dict__[key]


def pipe_recv_cmd(pipe, timeout):
    if timeout!=0:  #blocking
        count = 0
        while not pipe.poll() and count<timeout*10:
            time.sleep(0.1)
            count+=1
        if count>=timeout*10:
            return None
        else:
            return pipe.recv()
    else:       # non blocking
        if pipe.poll():
            return pipe.recv()
        else:
            return None
def pipe_send_cmd_recv_resp(pipe, cmd, time_out_in_second):
    pipe.send(cmd)
    count = 0
    while not pipe.poll() and count < time_out_in_second*10:
        time.sleep(0.1)
        count += 1
    if count >= time_out_in_second*10:
        return None
    else:
        #print "pipe_send_cmd_recv_resp wait for {0} second".format(count*0.1)
        return pipe.recv()
class name_pipe_cmd:
    def __init__(self, np_fnwp_seed):
        pipe_dir=os.path.join(lc.system_working_dir, "name_pipe")
        if not os.path.exists(pipe_dir):
            os.mkdir(pipe_dir)
        self.np_fnwp=os.path.join(pipe_dir,"{0}_{1}".format(np_fnwp_seed,lc.command_pipe_seed))
        if os.path.exists(self.np_fnwp):
            os.remove(self.np_fnwp)
        os.mkfifo(self.np_fnwp)

    def check_input_immediate_return(self):
        try:
            pipe = os.open(self.np_fnwp, os.O_RDONLY | os.O_NONBLOCK)
            input_command = os.read(pipe, 100)
            os.close(pipe)
            if len(input_command) != 0:
                command_list = input_command.split(" ")
                return command_list
        except OSError as err:
            if err.errno == 11:
                return None
            else:
                raise err
