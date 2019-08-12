cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment1
for i in {21..28}
do
	qsub "$i"_job.sh
done
# for i in {1267246..1267273}
# do
# 	qdel "$i"
# done