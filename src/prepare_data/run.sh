#!/bin/bash
#python convert_data_to_tfrecord.py --VOC-dir /home/lxy/Downloads/DataSet/VOC_dataset/VOCdevkit/VOC2012  --xml-dir Annotations --image-dir JPEGImages \
 #           --save-name train --dataset-name VOC2012  
#widerface
#python convert_data_to_tfrecord.py    --image-dir /home/lxy/Downloads/DataSet/Face_reg/id_5000_org \
 #           --save-name train --dataset-name Prison --anno-file ./output/data.txt

#python convert_data_to_tfrecord.py --image-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw --save-name train --dataset-name Prison \
 #       --anno-file ./output/data.txt
#python convert_data_to_tfrecord.py --image-dir /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw --save-name train --dataset-name Mobile \
 #       --anno-file ./output/data2.txt
python convert_data_to_tfrecord.py --image-dir  /home/lxy/Develop/Center_Loss/git_prj/BaiduImageSpider/img_dw --save-name bg --dataset-name FaceAnti \
        --anno-file ./output/data3_4.txt #app_pea_wal.txt

###convert multi dataset
#python convert_data_to_tfrecord.py --image-dir /home/lxy/Downloads/DataSet/Face_reg/id_5000_org --save-name fg --dataset-name Prison \
 #       --anno-file ./output/fg.txt
#python convert_data_to_tfrecord.py --image-dir /home/lxy/Downloads/DataSet/Face_reg/prison_img_test/prison_3936 --save-name bg --dataset-name Prison \
 #       --anno-file ./output/bg.txt
