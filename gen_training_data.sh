#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=64G
#SBATCH --time=90:00:00
#SBATCH --output=output-%j.txt

echo "Hello world! I am job number ${SLURM_JOBID}, running on these nodes: ${SLURM_JOB_NODELIST}."

. /etc/profile
module load lang/miniconda3/23.1.0
module load lang/python/3.9.18
module load lib/cuda/9.0.176
conda init bash
source ~/.bashrc
conda activate py39




echo "Building all axioms"
time java -Xmx14g -classpath /home/jarrad.singley/data/workspace/sigmanlp/lib/*:/home/jarrad.singley/data/workspace/sigmanlp/build/classes com.articulate.nlp.GenSimpTestData -a allAxioms 
echo "Building ground relations"
time java -Xmx14g -classpath /home/jarrad.singley/data/workspace/sigmanlp/lib/*:/home/jarrad.singley/data/workspace/sigmanlp/build/classes com.articulate.nlp.GenSimpTestData -g groundRelations
echo "Building outKindaSmall"
time java -Xmx14g -classpath /home/jarrad.singley/data/workspace/sigmanlp/lib/*:/home/jarrad.singley/data/workspace/sigmanlp/build/classes com.articulate.nlp.GenSimpTestData -s outKindaSmall 
echo "\n\nCompleted building training data"
