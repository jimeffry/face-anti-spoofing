#! /bin/bash
#python demo.py --img-path1 test1.jpg  --gpu 0 --load-epoch 370 --cmd-type imgtest
#python demo.py --img-path1 p1.png --gpu 0 --load-epoch 5-100 --cmd-type imgtest
#python demo.py --img-path1 /home/lxy/Downloads/DataSet/fruits-360/Test/Walnut/12_100.jpg  --load-epoch 20 --cmd-type imgtest
#python demo.py --img-path1 th3.jpeg --load-epoch 100 --cmd-type imgtest
#python demo.py --img-path1 train1.jpg --gpu 0 --load-epoch 5-50 --cmd-type imgtest
#python demo.py --img-path1 th.jpeg  --gpu 0 --load-epoch 370  --file-in /home/lxy/Downloads/DataSet/videos/profile_video.wmv --cmd-type videotest

##test filelist
python demo.py --file-in ../prepare_data/output/test.txt --out-file ./output/record.txt --base-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw/ \
        --load-epoch 750 --cmd-type filetest