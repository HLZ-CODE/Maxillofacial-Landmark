# Enhanced Landmark Detection in Oral and Maxillofacial CT and CBCT Scans: A Multi-Stage Global-Local Integration Approach
\usepackage{graphicx}

\begin{figure*}
    \centering
    \includegraphics[width=1\linewidth]{Fig1.pdf}
    \caption{Workflow of the proposed method. The pipeline consists of three stages. In the first stage, the complete CT or CBCT images are input into an alignment network to focus the images on the craniofacial region. In the second stage, global landmark detection is performed, while the third stage refines the landmarks on local patches.}
    \label{fig:fig1}
\end{figure*}

\begin{figure*}
    \centering
    \includegraphics[width=1\linewidth]{Fig2.pdf}
    \caption{Illustration of our coarse-to-fine landmark detection framework. The global stage reduces the resolution of the high-resolution craniofacial images for coarse landmark detection. In the local stage, fine-grained landmark detection is performed by cropping the regions centered on each coarse landmark.}
    \label{fig:fig2}
\end{figure*}

## 1. Highlights
* An accurate three-stage framework for landmark detection from 3D medical images
* A novel global-local information integration strategy for coarse-to-fine detection
* A multi-scale global information extraction module to obtain global feature
* A landmark attention module to capture global information for different landmark
* A novel multi-modality dataset consisting of CT and CBCT scans for evaluation


## 2. Preparation
### 2.1 Requirements
- python >=3.7
- pytorch >=1.10.0
- Cuda 10 or higher
- numpy
- pandas
- scipy
- nrrd
- time

## 3. Train and Test
### 3.1 Network Training 

* Training
```
python3 main.py
```

* Test
```
python3 main.py --resume ./SavePath/main/UNet3D/model_best.ckpt --test_flag 0 # Validation
python3 main.py --resume ./SavePath/main/UNet3D/model_best.ckpt --test_flag 0 # Test
```

## 4. Landmark detection Visualization
\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{Fig7.pdf}
    \caption{Visualization of landmarks on CT and CBCT dataset. The green dots denote ground truth landmarks, the red dots denote predicted landmarks.}
    \label{fig:fig7}
\end{figure}

## 5. Contact

Institution: College of Computer Science, Sichuan University

email: 2943998319@qq.com

## 6. Citation

[1] Mamta Juneja, Poojita Garg, Ravinder Kaur, Palak Manocha, Prateek, Shivam Batra, Pradeep Singh, Shaswat Singh, and Prashant Jindal. A review on cephalometric landmark detection techniques. Biomedical Signal Processing and Control, 66:102486, 2021.

[2] Fakhre Alam, Sami Ur Rahman, Sehat Ullah, and Kamal Gulati. Medical image registration in image guided surgery: Issues, challenges and research opportunities. Biocybernetics and Biomedical Engineering, 38(1):71–89, 2018.

[3] Ivana Isgum, Marius Staring, Annemarieke Rutten, Mathias Prokop, Max A. Viergever, and Bram van Ginneken. Multi-atlas-based segmentation with local decision fusion—application to cardiac and aortic segmentation in ct scans. IEEE Transactions on Medical Imaging, 28(7):1000–1010, 2009.

[4] Ozan Oktay, Wenjia Bai, Ricardo Guerrero, Martin Rajchl, Antonio De Marvao, Declan P O'Regan, Stuart A Cook, Mattias P Heinrich, Ben Glocker, and Daniel Rueckert. Stratified decision forests for accurate anatomical landmark localization in cardiac images. IEEE transactions on medical imaging, 36(1):332–342, 2016.

[5] Razvan Ioan Ionasec, Bogdan Georgescu, Eva Gassner, Sebastian Vogt, Oliver Kutter, Michael Scheuering, Nassir Navab, and Dorin Comaniciu. Dynamic model-driven quantitative and visual evaluation of the aortic valve from 4d ct. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2008: 11th International Conference, New York, NY, USA, September 6-10, 2008, Proceedings, Part I 11, pages 686–694. Springer, 2008.

[6] Dong Han, Yaozong Gao, Guorong Wu, Pew-Thian Yap, and Dinggang Shen. Robust anatomical landmark detection for mr brain image registration. In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2014: 17th International Conference, Boston, MA, USA, September 14-18, 2014, Proceedings, Part I 17, pages 186–193. Springer, 2014.

[7] Walid Abdullah Al, Ho Yub Jung, Il Dong Yun, Yeonggul Jang, HyungBok Park, and Hyuk-Jae Chang. Automatic aortic valve landmark localization in coronary ct angiography using colonial walk. PloS one, 13(7):e0200317, 2018.

[8] Minkyung Lee, Minyoung Chung, and Yeong-Gil Shin. Cephalometric landmark detection via global and local encoders and patch-wise attentions. Neurocomputing, 470:182–189, 2022.

[9] Runnan Chen, Yuexin Ma, Nenglun Chen, Lingjie Liu, Zhiming Cui, Yanhong Lin, and Wenping Wang. Structure-aware long short-term memory network for 3d cephalometric landmark detection. IEEE Transactions on Medical Imaging, 41(7):1791–1801, 2022.

[10] Yankai Jiang, Yiming Li, Xinyue Wang, Yubo Tao, Jun Lin, and Hai Lin. Cephalformer: incorporating global structure constraint into visual features for general cephalometric landmark detection. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 227–237. Springer, 2022.

[11]  ̈Ozg ̈un C ̧ ic ̧ek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3d u-net: learning dense volumetric segmentation from sparse annotation. In Medical Image Computing and ComputerAssisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19, pages 424–432. Springer, 2016.

[12] Runnan Chen, Yuexin Ma, Nenglun Chen, Daniel Lee, and Wenping Wang. Cephalometric landmark detection by attentive feature pyramid fusion and regression-voting. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part III 22, pages 873–881. Springer, 2019.

[13] Hansang Lee, Minseok Park, and Junmo Kim. Cephalometric landmark detection in dental x-ray images using convolutional neural networks. In Medical imaging 2017: Computer-aided diagnosis, volume 10134, pages 494–499. SPIE, 2017.

[14] Sung Min Lee, Hwa Pyung Kim, Kiwan Jeon, Sang-Hwy Lee, and Jin Keun Seo. Automatic 3d cephalometric annotation system using shadowed 2d image-based machine learning. Physics in Medicine & Biology, 64(5):055002, 2019.

[15] Julia MH Noothout, Bob D De Vos, Jelmer M Wolterink, Elbrich M Postma, Paul AM Smeets, Richard AP Takx, Tim Leiner, Max A Viergever, and Ivana Iˇsgum. Deep learning-based regression and classification for automatic landmark localization in medical images.IEEE transactions on medical imaging, 39(12):4011–4022, 2020.

[16] Simone Palazzo, Giovanni Bellitto, Luca Prezzavento, Francesco Rundo, Ulas Bagci, Daniela Giordano, Rosalia Leonardi, and Concetto Spampinato. Deep multi-stage model for automated landmarking of craniomaxillofacial ct scans. In 2020 25th International Conference on Pattern Recognition (ICPR), pages 9982–9987. IEEE, 2021.

[17] Gang Lu, Yuanxiu Zhang, Youyong Kong, Chen Zhang, Jean-Louis Coatrieux, and Huazhong Shu. Landmark localization for cephalometric analysis using multiscale image patch-based graph convolutional networks. IEEE Journal of Biomedical and Health Informatics, 26(7):3015– 3024, 2022.

[18] Christian Payer, Darko ˇStern, Horst Bischof, and Martin Urschler. Integrating spatial configuration into heatmap regression based cnns for landmark localization. Medical image analysis, 54:207–219, 2019.

[19] Qijie Zhao, Junhao Zhu, Junjun Zhu, Anwen Zhou, and Hui Shao. Bone anatomical landmark localization with cascaded spatial configuration network. Measurement Science and Technology, 33(6):065401, 2022.

[20] Antonia Stern, Lalith Sharan, Gabriele Romano, Sven Koehler, Matthias Karck, Raffaele De Simone, Ivo Wolf, and Sandy Engelhardt. Heatmapbased 2d landmark detection with a varying number of landmarks. InBildverarbeitung f ̈ur die Medizin 2021: Proceedings, German Workshop on Medical Image Computing, Regensburg, March 7-9, 2021, pages 22– 27. Springer, 2021.

[21] Guang-Quan Zhou, Juzheng Miao, Xin Yang, Rui Li, En-Ze Huo, Wenlong Shi, Yuhao Huang, Jikuan Qian, Chaoyu Chen, and Dong Ni. Learn fine-grained adaptive loss for multiple anatomical landmark detection in medical images. IEEE Journal of Biomedical and Health Informatics, 25(10):3854–3864, 2021.

[22] Xiang Li, Songcen Lv, Jiusi Zhang, Minglei Li, Juan J RodriguezAndina, Yong Qin, Shen Yin, and Hao Luo. Fdgr-net: Feature decouple and gated recalibration network for medical image landmark detection.Expert Systems with Applications, 238:121746, 2024.

[23] Gang Lu, Huazhong Shu, Han Bao, Youyong Kong, Chen Zhang, Bin Yan, Yuanxiu Zhang, and Jean-Louis Coatrieux. Cmf-net: craniomaxillofacial landmark localization on cbct images using geometric constraint and transformer. Physics in Medicine & Biology, 68(9):095020, 2023.

[24] Xiaoyang Chen, Chunfeng Lian, Hannah H Deng, Tianshu Kuang, Hung-Ying Lin, Deqiang Xiao, Jaime Gateno, Dinggang Shen, James J Xia, and Pew-Thian Yap. Fast and accurate craniomaxillofacial landmark detection via 3d faster r-cnn. IEEE transactions on medical imaging, 40(12):3867–3878, 2021.

[25] Yankun Lang, Chunfeng Lian, Deqiang Xiao, Hannah Deng, KimHan Thung, Peng Yuan, Jaime Gateno, Tianshu Kuang, David M Alfi, Li Wang, et al. Localization of craniomaxillofacial landmarks on cbct images using 3d mask r-cnn and local dependency learning. IEEE transactions on medical imaging, 41(10):2856–2866, 2022.

[26] Tao He, Guikun Xu, Li Cui, Wei Tang, Jie Long, and Jixiang Guo. Anchor ball regression model for large-scale 3d skull landmark detection.Neurocomputing, 567:127051, 2024.

[27] Zhusi Zhong, Jie Li, Zhenxi Zhang, Zhicheng Jiao, and Xinbo Gao. An attention-guided deep regression model for landmark detection in cephalograms. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part VI 22, pages 540–548. Springer, 2019.

[28] Soh Nishimoto, Takuya Saito, Hisako Ishise, Toshihiro Fujiwara, Kenichiro Kawai, and Maso Kakibuchi. Three-dimensional cranio-facial landmark detection in ct slices from a publicly available database, using multi-phased regression networks on a personal computer. MedRxiv, pages 2021–03, 2021.

[29] Minmin Zeng, Zhenlei Yan, Shuai Liu, Yanheng Zhou, and Lixin Qiu. Cascaded convolutional networks for automatic cephalometric landmark detection. Medical Image Analysis, 68:101904, 2021.

[30] Qin Liu, Han Deng, Chunfeng Lian, Xiaoyang Chen, Deqiang Xiao, Lei Ma, Xu Chen, Tianshu Kuang, Jaime Gateno, Pew-Thian Yap, et al. Skullengine: a multi-stage cnn framework for collaborative cbct image segmentation and landmark detection. In Machine Learning in Medical Imaging: 12th International Workshop, MLMI 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, September 27, 2021, Proceedings 12, pages 606–614. Springer, 2021.

[31] Gauthier Dot, Thomas Schouman, Shaole Chang, Fr ́ed ́eric Rafflenbeul, Adeline Kerbrat, Philippe Rouch, and Laurent Gajny. Automatic 3-dimensional cephalometric landmarking via deep learning. Journal of dental research, 101(11):1380–1387, 2022.

[32] Xianglong Wang, Eric Rigall, Qianmin Chen, Shu Zhang, and Junyu Dong. Efficient and stable cephalometric landmark localization using two-stage heatmaps' regression. IEEE Transactions on Instrumentation and Measurement, 71:1–16, 2022.

[33] Florin C Ghesu, Bogdan Georgescu, Tommaso Mansi, Dominik Neumann, Joachim Hornegger, and Dorin Comaniciu. An artificial agent for anatomical landmark detection in medical images. In Medical Image Computing and Computer-Assisted Intervention-MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part III 19, pages 229–237. Springer, 2016.

[34] Florin-Cristian Ghesu, Bogdan Georgescu, Yefeng Zheng, Sasa Grbic, Andreas Maier, Joachim Hornegger, and Dorin Comaniciu. Multi-scale deep reinforcement learning for real-time 3d-landmark detection in ct scans. IEEE transactions on pattern analysis and machine intelligence, 41(1):176–189, 2017.

[35] Athanasios Vlontzos, Amir Alansary, Konstantinos Kamnitsas, Daniel Rueckert, and Bernhard Kainz. Multiple landmark detection using multi-agent reinforcement learning. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13–17, 2019, Proceedings, Part IV 22, pages 262–270. Springer, 2019.

[36] Ionut Cosmin Duta, Li Liu, Fan Zhu, and Ling Shao. Pyramidal convolution: Rethinking convolutional neural networks for visual recognition.arXiv preprint arXiv:2006.11538, 2020.

[37] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV), pages 3–19, 2018.

[38] Tao He, Jie Yao, Weidong Tian, Zhang Yi, Wei Tang, and Jixiang Guo. Cephalometric landmark detection by considering translational invariance in the two-stage framework. Neurocomputing, 464:15–26, 2021.

[39] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi. V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 2016 fourth international conference on 3D vision (3DV), pages 565–571. Ieee, 2016.

[40] Yankun Lang, Chunfeng Lian, Deqiang Xiao, Hannah Deng, Peng Yuan, Jaime Gateno, Steve GF Shen, David M Alfi, Pew-Thian Yap, James J Xia, et al. Automatic localization of landmarks in craniomaxillofacial cbct images using a local attention-based graph convolution network. InMedical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part IV 23, pages 817–826. Springer, 2020.