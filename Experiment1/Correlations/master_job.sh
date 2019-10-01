for i in {1..29}
do
	qsub "$i"_job.sh
done
# for i in {1358058..1358077}
# do
# 	qdel "$i"
# done