#!/bin/bash
#generate image list
#python image_preprocess.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/id_5000_org  --out-file ./output/fg.txt --base-label 0 --cmd-type gen_filepath_2dir
#python image_preprocess.py --img-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_3936 --out-file ./output/bg.txt --base-label 0 --cmd-type gen_filepath_2dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_hand_fg --out-file ./output/hand_fg.txt  --base-label 2 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_hand_bg --out-file ./output/hand_bg.txt  --base-label 0 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_print_fg --out-file ./output/print_fg.txt  --base-label 2 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_print_bg --out-file ./output/print_bg.txt  --base-label 0 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_mobile_fg --out-file ./output/mobile_fg.txt  --base-label 1 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/a_mobile_bg --out-file ./output/mobile_bg.txt  --base-label 0 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/mobile --out-file ./output/mobile_1.txt  --base-label 1 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/TV --out-file ./output/TV.txt  --base-label 2 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/ConTro --out-file ./output/control.txt  --base-label 3 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw --out-file ./output/bg_1.txt --base-label 0 --cmd-type gen_filepath_2dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/Peach --out-file ./output/apple.txt  --base-label 1 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/Apple --out-file ./output/peach.txt  --base-label 2 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/Walnut --out-file ./output/walnut.txt  --base-label 3 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/cellphone --out-file ./output/cellphone.txt  --base-label 1 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/monitor --out-file ./output/monitor.txt  --base-label 2 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/remote_control --out-file ./output/remote_control.txt  --base-label 3 --cmd-type gen_filepath_1dir
#python image_preprocess.py --img-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/bg_2 --out-file ./output/bg_2.txt  --base-label 0 --cmd-type gen_filepath_1dir
### merge
python image_preprocess.py  --file-in ./output/train_celeb.csv   --out-file ./output/train.txt --cmd-type gen_trainlabel
#python image_preprocess.py  --file-in /home/lxy/Downloads/train_celeb.txt  --file2-in /home/lxy/Downloads/identity_CelebA.txt --out-file ./output/train_celeb.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/train.txt  --file2-in ./output/list_eval_partition.txt --save-dir ./output --cmd-type split
#python image_preprocess.py  --file-in ./output/print_hand.txt  --file2-in ./output/mobile_fg.txt --out-file ./output/data.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/mobile_bg.txt  --file2-in ./output/mobile_fg.txt --out-file ./output/mobile.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/mobile.txt  --file2-in ./output/print_hand.txt --out-file ./output/data2.txt  --base-label 0 --cmd-type merge2change
#python image_preprocess.py  --file-in ./output/mobile_1.txt  --file2-in ./output/TV.txt --out-file ./output/mobile_tv.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/mobile_tv.txt  --file2-in ./output/control.txt --out-file ./output/mobile_tv_ct.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/mobile_tv_ct.txt  --file2-in ./output/bg_1.txt --out-file ./output/data3.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/apple.txt  --file2-in ./output/peach.txt --out-file ./output/apple_peach.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/apple_peach.txt  --file2-in ./output/walnut.txt --out-file ./output/app_pea_wal.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/app_pea_wal.txt  --file2-in ./output/bg_1.txt --out-file ./output/data2.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/cellphone.txt  --file2-in ./output/monitor.txt --out-file ./output/phone_monitor.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/phone_monitor.txt  --file2-in ./output/remote_control.txt --out-file ./output/phone_moni_rc.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/phone_moni_rc.txt  --file2-in ./output/bg_2.txt --out-file ./output/data4.txt --cmd-type merge
#python image_preprocess.py  --file-in ./output/data3.txt  --file2-in ./output/data4.txt --out-file ./output/data3_4.txt --cmd-type merge