#Observation
    Tensorboard advent_high and state_value almost same shape with different direction
#Guess
    adjust_r in trainer maybe very small, make the advent_high dominated by state_value
    1. not sure why dominate advent high not advent
    2. and also not know why the state value be relative low at biginning, or let's say where the inital state value data comes from
# first round thinking of try
    1. add buffer_r and train_r in tensorboard
    2. add limit of square error to 1?
    3. try clip value other than 0.2
    4. the denes net may not heavy enough?
# further thinking
    1. not need add train_r and train_r = advent +state value
    2. all the record go to server are success transaction( Sell success)
# second round thinking of try
    1. maybe try increase the dense layer ?
    2. add add limit of square error to 1?
    
