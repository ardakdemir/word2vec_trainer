import sys
import os 
import fasttext

if __name__ == "__main__":
    args = sys.argv
    input_file = args[1]
    dim = int(args[3])
    save_path = args[2]
    model = fasttext.train_unsupervised(input_file,dim=dim) 
    model.save_model(save_path)
