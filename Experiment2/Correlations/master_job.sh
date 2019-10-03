for i in {1..28}
do
	qsub "$i"_job.sh
done
# for i in {1357580..1357606}
# do
# 	qdel "$i"
# done