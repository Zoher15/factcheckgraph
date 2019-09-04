cd /gpfs/home/z/k/zkachwal/Carbonate/FactCheckGraph/Experiment2/DBPedia
for i in {1..29}
do
	qsub "$i"_job_co.sh
done
# for i in {1267246..1267273}
# do
# 	qdel "$i"
# done