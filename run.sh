# Activate the virtual envoriment
source ./virt_env/bin/activate 

# Run the code
python model_training/BERT_finetuning.py "$@"

# Close the virtual envoriment
deactivate 