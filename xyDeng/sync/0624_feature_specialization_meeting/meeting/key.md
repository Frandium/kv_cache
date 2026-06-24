Question: 
是否能够把均匀分布的features稳定均匀分到experts上

Verdict:
1. 对于减去均值后的hidden states进行聚类，将聚类中心作为gating初始化，在背景数据单一情况下能够稳定发生feature-level specialized分发，并且训练1600步的结果现实训练后仍能够保持均匀分发。但是对背景数据增加干扰，聚类结果会发生偏差，最终导致初始化方法没有办法稳定均匀分发。

Model:

h=c + ri + \epsiloni

features均匀存在，在空间中不是均匀分布的。
只有通过聚类找到不同features的位置，初始化的时候才能发生specialzed分发，并且由于top-1分发具有会不断提升初始分发的置信度，因此会导致ending lock-in and collapse。

对于random情形下不行的解释：参见文档：
random情况下的中心

对于聚类centering初始化能行的解释：
参见文档。

Evidence:
Boundary:
Ask mentor:
next step. 