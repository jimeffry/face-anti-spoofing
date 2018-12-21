#!/bin/bash
#generate image list
python image_preprocess.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/id_5000_org  --out-file annot.txt --cmd-type gen_filepath_2dir
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/id_256/id/  --out-file ./output/id_256.txt --cmd-type gen_filepath_1dir

### compare 2dirs
#python gen_file_test.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_6244_v2/ --dir2-in /home/lxy/Downloads/DataSet/Face_reg/prison_result/server_0_6244/ \
 #       --save-dir /home/lxy/Downloads/DataSet/Face_reg/prison_result/prison_server_not/ --cmd-type compare_dir