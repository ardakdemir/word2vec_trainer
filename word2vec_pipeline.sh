#/usr/local/bin/nosh
##$ -cwd
##$ -l os7,s_vmem=100G,mem_req=100G
##$ -N word2_vec_trainer
w2v_types=(w2v fasttext)
langs=(czech hungarian finnish  turkish)
word_vec_prefs=(cs hu fi tr)

dim=$1
#lang=$1
#dim=$2
#w2v_type=${3}
for lang in "${word_vec_prefs[@]}"
do
    for w2v_type in "${w2v_types[@]}"
    do
        echo "Training a ${dim} dimensional ${w2v_type} vector for ${lang} language"
        model_input_name=$lang'_wiki.txt'

        if [ ${w2v_type} == 'w2v' ]
        then
            model_name=${w2v_type}'_'${lang}'_'${dim}
        else
            model_name=${w2v_type}'_'${lang}'_'${dim}'.bin'
        fi
        wiki_root=https://dumps.wikimedia.org/${lang}wiki/latest/
        wiki_dump=${lang}wiki-latest-pages-articles.xml.bz2
        save_folder="../"${lang}
        if [ ! -d "${save_folder}" ]
        then
            echo "Creating folder "${save_folder}
            mkdir ${save_folder}
        fi
        if [ ! -f "${wiki_dump}" ]
        then
            echo "Downloading wiki dump for "${lang}
            wget ${wiki_root}${wiki_dump}
        fi 
        if [ ! -f "${model_input_name}" ]
        then
            echo "Preprocessing wiki dump for ${lang}"
            python process_wiki.py ${wiki_dump} ${model_input_name}
        fi
        save_path=${save_folder}'/'${model_name} 
        if [ ! -f "${save_path}" ]
        then
            if [ ${w2v_type} == 'w2v' ]
            then
                echo "Training: "${w2v_type}' type model for  '${lang}' language with  '${dim}' dimensions'
                python word2vec_model.py ${model_input_name} ${save_path} ${dim}
            else 
                python train_fasttext.py ${model_input_name} ${save_folder}'/'${model_name} ${dim}
            fi
        else
            echo "Skipping ${model_name} as it already exists"
        fi
    done
done
#python gensim_txt.py ${model_name} $lang'_'${dim}".tsv"
