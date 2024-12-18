# Activate the virtual envoriment
source ./virt_env/bin/activate 

# Run the code
python finetuning-testing/BERT_finetuning.py "$@"

# Close the virtual envoriment
deactivate 