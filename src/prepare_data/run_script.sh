#!/bin/bash
#generate image list
#python image_preprocess.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/id_5000_org  --out-file ./output/data.txt --cmd-type gen_filepath_2dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_hand_fg --out-file ./output/hand_fg.txt  --base-label 2 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_hand_bg --out-file ./output/hand_bg.txt  --base-label 0 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_print_fg --out-file ./output/print_fg.txt  --base-label 2 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_print_bg --out-file ./output/print_bg.txt  --base-label 0 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_mobile_fg --out-file ./output/mobile_fg.txt  --base-label 1 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_mobile_bg --out-file ./output/mobile_bg.txt  --base-label 0 --cmd-type gen_filepath_1dir
### merge
#python image_preprocess.py  --file-in ./output/hand_fg.txt  --file2-in ./output/hand_bg.txt --out-file ./output/hand.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/print_fg.txt  --file2-in ./output/print_bg.txt --out-file ./output/print.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/print.txt  --file2-in ./output/hand.txt --out-file ./output/print_hand.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/print_hand.txt  --file2-in ./output/mobile_fg.txt --out-file ./output/data.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/mobile_bg.txt  --file2-in ./output/mobile_fg.txt --out-file ./output/mobile.txt --cmd-type merge
python image_preprocess.py  --file-in ./output/mobile.txt  --file2-in ./output/print_hand.txt --out-file ./output/data2.txt  --base-label 0 --cmd-type merge2change
