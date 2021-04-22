All experiments for the purpose of adding some randomness are logged in this directory.

testmng: 
	testing fixed_mng func in MNGOBox which is a version of MANGO with fixed mask size of 16.

fixedcutout:
	testing CutoutF as a transformation instead of Cutout which is close to testmng func but it also has Cutout randomness in choosing among 4 masks of size 16.

	-purpose of running testmng and fixedcutout experiments is to see how randomness vs wiseness performs. (the more accurate comparison would be to have testmng choose mask of size 16 in any where inside image but there is going to be 256 masks to compare their probabilities which has a huge cost to try)
cutout2:
	running cutout not as a transform but instead applied like MANGO on resnet18 with default settings.

	-this experiment proved the noticable difference between running as a transform or not (~95 acc when a transform vs ~86 when not) on resnet18 with default settings.

mngcut and mngcut2-astrans:
	running mngcut function once like MANGO and once as a transform on resnet18 with default settings.

******Analysis:
	using cutout (or other functions) as transform is resulting better than applying them once on a dataset and train and test on the result. My reasoning is that apparently using as a transform is applying the functions once in each epoch which means model will be traiined on n_epochs versions of each training data point in the end so we'll have a better model resulting in higher test accuracy in the end.
	**comparing original Cutout with original MANGO on wideresnet shows MANGO is learning the training data faster and more than Cutout (which totally makes sense considering MANGO training on a fixed masked data whereas Cutout is training on randomly different masked data in each epoch) but MANGO is slower and in the end, less accurate on testing data in comparison to Cutout which again is reasonable as Cutout model was traind on a more diverse augmented data so is better generalized and trained and powerful to predict test data better.

--running OrigMNGCut without data_augmentation so that we can use get_id() function, doesn't seem to help. Other than making the time more efficient, the randomness is to either choose the rmask center on a 16x16 portion or a 8x8 one. whereas using data_augementation means we need to apply MANGO from begining on each data point (regarding the randomness in data_augementation) and choose rmask center on a 16x16 or 8x8 portion which can differ (the portion) on each epoch. thus resulting in more randomness and supposedly more accuracy. 