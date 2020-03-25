# Goal
    Implement the holding terminate status, avoiding following situation
    LHP = 5 as example
    
record_id | s | r | s_ | train record inform | s LHPP2 LHP code  | s_ LHPP2 LHP code
--- | ---|  ---| ---| ---| --- | --- |
0 | s0 | r0 | s0_ | s0 s4_ r0-r4 | 1 0 0 0 0 0  | 0 1 0 0 0 0 |
1 | s1 | r1 | s1_ | s1 s4_ r1-r4 | 0 1 0 0 0 0  | 0 0 1 0 0 0 |
2 | s2 | r2 | s2_ | s2 s4_ r2-r4 | 0 0 1 0 0 0  | 0 0 0 1 0 0 |
3 | s3 | r3 | s3_ | s3 s4_ r3-r4 | 0 0 0 1 0 0  | 0 0 0 0 1 0 |
4 | s4 | r4 | s4_ | s4 s4_ r4-r5 | 0 0 0 0 1 0  | 0 0 0 0 0 1 |
5 | s5 | r5 | s5_ | s5 s5_ r5    | 0 0 0 0 0 1  | 1 0 0 0 0 0 |

The issue is s4_ state value never trained in this transaction, which should be 0

try two senario
1. train optimize not add s_ 's state value, as it should be 0 anyway
2. train optimize keep add s_ 's state value, and add one record in TD buffer for s_ and the state value should be 0

