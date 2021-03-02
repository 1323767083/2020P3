data:
stock list 500's 10m, 5m,1m,

model
1. enhance stride 3 deep net
2. stride 2

program
remove multiplex action
remove or check av close price
remove reward ==0 record before training
FTP program
add on data generation
GPU config for trader
handle actual account and trader account's 误差


Daily Data Updater:
1. Download Data
2. register Data in DB
3. generate DBI  DBTP

Potential issue:
1.the reset day has not action, is it a issue or not a issue for actual situation
2. State 究竟又啥功能， 在executor 里还要保留吗
   a. 就是 force sell 这个总需要
   b. error 退出phase， NB 时间长了 就要reset， Holding 时间长了就要force sell 这两个好像也需要
       reset 是不是占了一天？  reset 在training 里 是占了一个round 但是在现实中 reset 应该不占一天， 这个在 eval 里好像有问题



三月二号
1. 回归测试 
2. 500 stocklist ， change create log name， make 10m data
3. trader 编写调试模拟器
4. FTP 


