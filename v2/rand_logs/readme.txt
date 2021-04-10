All experiments for the purpose of adding some randomness are logged in this directory.

testmng: 
	testing fixed_mng func in MNGOBox which is a version of MANGO with fixed mask size of 16.

fixedcutout:
	testing CutoutF as a transformation instead of Cutout which is close to testmng func but it also has Cutout randomness in choosing among 4 masks of size 16.

	-purpose of running testmng and fixedcutout experiments is to see how randomness vs wiseness performs. (the more accurate comparison would be to have testmng choose mask of size 16 in any where inside image but there is going to be 256 masks to compare their probabilities which has a huge cost to try)
cutout2:
	running cutout not as a transform but instead applied like MANGO on resnet18 with default settings.

	-this experiment proved the noticable difference between running as a transform or not (~95 acc when a transform vs ~86 when not) on resnet18 with default settings.


