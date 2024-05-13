### Generate your dataset

If you want to reproduce experiments with the BERT-Base embeddings of the imdb dataset, you need to run the following script 

python make_bert_embeddings.py

By default, the train and the test sizes are set to 100 and 0 by default. 

1) First, the program will create a folder './imdb_dataset_debug' containing the file 'data.pkl'.

This file is built through the function create_imdb_dataset, which can be found in the file utils_data.py

This function selects a subset of the imdb dataset, with short context lengths, balancing the labels (positive sentiment: +1, negative sentiment: -1), and returns a dataset in the form of dictionary

{'X_train': X_train, 'X_train_tok': X_train_tok, 'y_train': y_train, 'X_test': X_test, 'X_test_tok': X_test_tok, 'y_test': y_test}

where the suffix 'tok' stands for 'tokenized'.

2) Second, the program will generate the BERT-Base embeddings of this small dataset. The argument set to 0 as default, means that we look at the first embeddings of BERT. This number can go up to 12, as the number of layers in this encoder model.

This is done through the function generate_embeddings, always in the utils_data.py file. This function returns another dictionary, that is later processed directly in the original make_bert_embeddings.py, producing the final dictionary some_embeddings, only with the training samples, as these will be used both as training and validation later. 

This dictionary is finally saved in './imdb_dataset_debug/embeddings.pkl'


### Compute the sensitivity for RF and RAF

The script to compute the sensitivity is called sensitivity.py
It uses all the functions in the file utils.py, that contains numpy and torch versions of the mapping discussed in the paper, including some utils for the optimization, which is carried with torch.optim

You can run the script with

for dataset in "imdb" "synthetic"; do
  for map in "drf" "rf" "raf" "raf_relu"; do
    python3 ./sensitivity_GD_torch.py --i 0 --map "$map" --dataset "$dataset" > outputs/0.txt
  done
done

if you want to cover all possible settings, including the one with Gaussian synthetic data, not presented in the paper. You can use the argument i to parallelize the script, and remember to adapt the save_dir / folder_name objects in the script, to correctly load your dataset from before and to correctly save your results in the right place. The script saves in txt files the values of d, n, L, S and also the l2 norm of the mapping, which can be used for sanity checks.


### Simulate the generalization on context modification


The script to compute the error on top of token modification is called gen.py
It uses all the functions in the file utils.py, that contains numpy and torch versions of the mapping discussed in the paper, including some utils for the optimization, which is carried with torch.optim

You can run the script with

for dataset in "imdb" "synthetic"; do
  for map in "rf" "raf" "raf_relu"; do
    for training in "fine_tuning" "total"; do
      for optim in "feat_align" "features" "loss"; do
        srun python3 ./gen.py --i 0 --training "$training" --dataset "$dataset" --optim "$optim" --map "$map"> outputs/gen_0.txt
      done
    done
  done
done

where in the code the notation "total" stands for "re-training" in the paper. Feel free to consider only the arguments that are interesting for your reproducibility purposes. Remember to adapt the save_dir / folder_name objects in the script, to correctly load your dataset from before and to correctly save your results in the right place. The script saves in txt files the values of d, n, and N, and the 4 values t0, t1, D0, D1. This four values represent the output of the model:

D represents a sample with the perturbation Delta on top of the original sample t.
0 means that the output is computed before the second training (fine-tuning or re-training).
1 means that the output is computed after the second training (fine-tuning or re-training).

Notice that t1 will be equal to y = -y_\Delta, following the notation of the paper. The possible arguments for --optim are "features" and "feat_align", for the two losses in (24), and "loss" for the loss in (16).


### Reproduce Figure 1

In the file bert.ipynb you can find all the code to plot the average attention scores between the embeddings of BERT-Base of the two sentences "I love her much" and "I love her smile".

In the file llama2.ipynb, you can find the code to reproduce the response of llama2-7b-chat on the prompts in the table in Figure 1. Notice that you need to download the model, and put it in the parent directory.
