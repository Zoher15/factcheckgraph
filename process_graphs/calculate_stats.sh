#PBS -k o
#PBS -l nodes=1:ppn=1,vmem=64gb,walltime=0:10:00
#PBS -M zoher.kachwala@gmail.com
#PBS -m abe
#PBS -N calculate_stats
#PBS -j oe
source /N/u/zkachwal/Carbonate/miniconda3/etc/profile.d/conda.sh
conda activate
cd /gpfs/home/z/k/zkachwal/Carbonate/factcheckgraph/process_graphs/
# time python find_intersect.py -fcg fred -kg dbpedia
# time python calculate_stats.py -gc fred -gt tfcg
# time python calculate_stats.py -gc fred -gt ffcg
# time python calculate_stats.py -gc fred -gt ufcg
# time python calculate_stats.py -gc co-occur -gt tfcg_co
# time python calculate_stats.py -gc co-occur -gt ffcg_co
# time python calculate_stats.py -gc co-occur -gt ufcg_co
# time python calculate_stats.py -gc backbone_df -gt tfcg_bbdf
# time python calculate_stats.py -gc backbone_df -gt ffcg_bbdf
# time python calculate_stats.py -gc backbone_df -gt ufcg_bbdf
# time python calculate_stats.py -gc backbone_dc -gt tfcg_bbdc
# time python calculate_stats.py -gc backbone_dc -gt ffcg_bbdc
# time python calculate_stats.py -gc backbone_dc -gt ufcg_bbdc
# time python calculate_stats.py -gc largest_ccf -gt tfcg_lgccf
# time python calculate_stats.py -gc largest_ccf -gt ffcg_lgccf
# time python calculate_stats.py -gc largest_ccf -gt ufcg_lgccf
# time python calculate_stats.py -gc largest_ccc -gt tfcg_lgccc
# time python calculate_stats.py -gc largest_ccc -gt ffcg_lgccc
# time python calculate_stats.py -gc largest_ccc -gt ufcg_lgccc
time python calculate_stats.py -gc old_fred -gt tfcg_old
time python calculate_stats.py -gc old_fred -gt ffcg_old
time python calculate_stats.py -gc old_fred -gt ufcg_old
