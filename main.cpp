#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
using namespace dlib;
using namespace std;
/* 函数声明 */
/* 人脸库训练 */
int candidates_train(const char *facesFile,std::vector<matrix<float,0,1>>&candidates_descriptors,std::vector<string>&candidates);
/* 输出人脸位置 返回识别结果 */
string face_location(const char *imgFile,std::vector<int>&locates, std::vector<matrix<float,0,1>>&candidates_descriptors,std::vector<string>&candidates);
/* 判断是否为活体 */
bool liveness_detection(const char *DeepFile,std::vector<int>&locates);  
const int IMG_HEIGHT =  720;
const int IMG_WIDTH =  1280;
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;
template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;
template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;
template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;
using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
    alevel0<
    alevel1<
    alevel2<
    alevel3<
    alevel4<
    max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
    input_rgb_image_sized<150>
    >>>>>>>>>>>>;
frontal_face_detector detector = get_frontal_face_detector(); // 人脸正脸检测器
shape_predictor sp; //人脸关键点检测器
anet_type net;  // 人脸识别模型
int main()
{
    const char *imgFile = "/home/zhoujie/data/allface/0002_IR_allleft.jpg";
    const char *facesFile = "/home/zhoujie/cProject/dlib_test/candidate-face/";
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
    std::vector<matrix<float,0,1>> candidates_descriptors;
    std::vector<string> candidates;
    /* 人脸库训练 */
    candidates_train(facesFile,candidates_descriptors,candidates);
    std::vector<int> locates;
    /* 输出人脸位置 返回识别结果 */
    string who = face_location(imgFile, locates, candidates_descriptors,candidates); 
    cout << "识别结果：" << endl;
    cout << "This is " << who << endl;
    //深度图与红外图是水平翻转的
    locates[0] = IMG_WIDTH - locates[0] -locates[2]; 
    // printf("%d,%d,%d,%d\n", locates[0],locates[1],locates[2],locates[3]);
    const char *DeepFile = "/home/zhoujie/data/allface/0002_raw_allleft.raw";
    bool IS_FACE;
    /* 判断是否为活体 */
    IS_FACE = liveness_detection( DeepFile, locates);
    // printf("RESULT : %d\n", IS_FACE);
}
/* 人脸库训练 */
int candidates_train(const char *facesFile,std::vector<matrix<float,0,1>>&candidates_descriptors,std::vector<string>&candidates)
{
    DIR *dir;
    struct dirent *ptr; 
    char base[30]; 
    const char *pick=".jpg"; //需要的子串;
    char IRfile[100];
    char *name;
    int face_num = 0;
    std::vector<matrix<rgb_pixel>> faces;
    if ((dir=opendir(facesFile)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }
    while ((ptr=readdir(dir)) != NULL)
    {
        strcpy(base, ptr->d_name);
        if(strstr(base,pick)) 
        {
            printf("training image:%s\n",base); 
            strcpy(IRfile, facesFile);
            strcat(IRfile, base);
            name = strtok(base, "_");
            // printf("%s\n",name); 
            string candidate = name;
            cout << "candidate: " << candidate << endl;
            candidates.push_back(candidate);
            matrix<rgb_pixel> img;
            load_image(img, IRfile);
            std::vector<rectangle> dets = detector(img);
            full_object_detection shape = sp(img, dets[0]);
            matrix<rgb_pixel> face_chip;
            extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
            faces.push_back(move(face_chip));
            face_num += 1;
        }  
    }
    candidates_descriptors = net(faces);
    printf("训练结果：\n共训练 %d 张人脸\n", face_num);
    closedir(dir);
    return 0;
}
/* 函数 输出人脸位置 返回识别结果 */
string face_location(const char* imgFile,std::vector<int>&locates, std::vector<matrix<float,0,1>>&candidates_descriptors,std::vector<string>&candidates)
{   
    cout << "processing image " << imgFile << endl;
    matrix<rgb_pixel> img;
    load_image(img, imgFile);
    std::vector<rectangle> dets = detector(img);
    // cout << "Number of faces detected: " << dets.size() << endl;
    locates.push_back(dets[0].left());
    locates.push_back(dets[0].top());
    locates.push_back(dets[0].right() - dets[0].left());
    locates.push_back(dets[0].bottom() - dets[0].top());
    full_object_detection shape = sp(img, dets[0]);
    std::vector<matrix<rgb_pixel>> faces;
    matrix<rgb_pixel> face_chip;
    extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip);
    faces.push_back(move(face_chip));
    std::vector<matrix<float,0,1>> face_descriptors = net(faces);
    float distance;
    float best_distance = length(face_descriptors[0]-candidates_descriptors[0]);
    // printf("k = 0 ,best_distance = %f\n",best_distance);
    size_t candidates_num = candidates_descriptors.size();
    int candidates_num_int = static_cast<int>(candidates_num);
    // printf("candidates_num_int : %d\n", candidates_num_int);
    int best_k = 0;
    for (int k = 1; k < candidates_num_int; k++)
    {
        distance = length(face_descriptors[0]-candidates_descriptors[k]);
        // printf("k = %d ,distance = %f\n",k,distance);
        if (distance < best_distance) 
        {
            best_distance = distance;
            best_k = k;
        }
    }
    string who;
    if (best_distance < 0.6) {
        who = candidates[best_k];
    }
    else{
        who = "Unknow";
    }
    return who;
}
/* 函数判断是否为活体 */
bool liveness_detection(const char *DeepFile,std::vector<int>&locates)
{
    const int ITER = 5000; // 随机取点次数
    const float PLANE_OR_NOT = 0.2; // 判断是否为平面的分界线
    const int SIGMA = 1;
    typedef unsigned short UNIT16;
    // 从.raw读取二进制16位数据到MatDATA
    UNIT16 MatDATA[IMG_HEIGHT*IMG_WIDTH];
    FILE *fp = NULL;
    fp = fopen( DeepFile, "rb" );
    size_t sizeRead = fread(MatDATA,sizeof(UNIT16),IMG_HEIGHT*IMG_WIDTH,fp);
    if (sizeRead != IMG_HEIGHT*IMG_WIDTH) {
        printf("error!\n");
    }    
    fclose(fp);
    int n = 0;
    int i,j;
    int COL = locates[0],ROW = locates[1],FACE_WIDTH = locates[2],FACE_HEIGHT = locates[3]; //位置信息
    // txt :157 66 172 198 , 取行66：66+198,列取157：157+172
    int faceno0_num = FACE_HEIGHT*FACE_WIDTH -1; 
    int FaceDATA[3][100000];
    n = 0;
    for(i = 1;i< FACE_HEIGHT+1;i++)
        {
            for(j= 1;j< FACE_WIDTH+1;j++) 
            { 
                if (MatDATA[IMG_WIDTH*(ROW+i-2)+COL+j-2] == 0)
                {
                    faceno0_num -= 1; // 非零深度点个数为 faceno0_num+1
                    continue;
                }
                FaceDATA[1][n] = i;
                FaceDATA[0][n] = j; 
                FaceDATA[2][n] = MatDATA[IMG_WIDTH*(ROW+i-2)+COL+j-2];
                n += 1;
            } 
        } 
    // int test = 0;  
    // printf("%d,%d,%d,%d\n",test,FaceDATA[0][test],FaceDATA[1][test],FaceDATA[2][test]);    
    int pretotal = 0;  // 符合拟合模型的数据的个数
    int x[3],y[3],z[3];  // 随机取三个点 
    srand((unsigned)time(NULL));
    float a,b,c;  // 拟合平面方程 z=ax+by+c
    // float besta,bestb,bestc;  // 最佳参数
    int rand_num[3];
    float check,distance;
    int total = 0;
    for(i = 0; i < ITER; i++)
    {
        do{
            rand_num[0] = std::rand()%faceno0_num; 
            rand_num[1] = std::rand()%faceno0_num; 
            rand_num[2] = std::rand()%faceno0_num; 
        }while(rand_num[0] == rand_num[1] || rand_num[0] == rand_num[2] || rand_num[1] == rand_num[2]);
        for(n = 0; n < 3; n++ )
        {
            x[n] = FaceDATA[0][rand_num[n]];
            y[n] = FaceDATA[1][rand_num[n]];
            z[n] = FaceDATA[2][rand_num[n]];
            // printf("%d,%d,%d,%d\n", x[n],y[n],z[n],n);
        }
        check = (x[0]-x[1])*(y[0]-y[2]) - (x[0]-x[2])*(y[0]-y[1]);
        if ( check == 0)  // 防止提示浮点数例外 (核心已转储)
        {
            i -= 1;
            continue;
        }
        a = ( (z[0]-z[1])*(y[0]-y[2]) - (z[0]-z[2])*(y[0]-y[1]) )*1.0/( (x[0]-x[1])*(y[0]-y[2]) - (x[0]-x[2])*(y[0]-y[1]) );
        if (y[0] == y[2])  // 防止提示浮点数例外 (核心已转储)
        {
            i -= 1;
            continue;
        }
        b = ((z[0] - z[2]) - a * (x[0] - x[2]))*1.0/(y[0]-y[2]);
        c = z[0]- a * x[0] - b * y[0];
        // printf("%f,%f,%f\n",a,b,c);
        total = 0;
        for(n = 0; n < faceno0_num +1 ; n++ )
        {
            distance = fabs(a*FaceDATA[0][n] + b*FaceDATA[1][n] - 1*FaceDATA[2][n] + c*1);
            if (distance < SIGMA)
            {
                total +=1;
            }
        }
        // printf("%d,%f,%d\n",i,distance,total);
        if (total > pretotal)  // 找到符合拟合平面数据最多的拟合平面
        {
            pretotal=total;
            // besta = a;
            // bestb = b;
            // bestc = c;
        }
    }
    float pretotal_ary = pretotal *1.0/ faceno0_num ;
    printf("活体检测结果：\npretotal_ary=%f,",pretotal_ary);
    bool IS_FACE;
    if (pretotal_ary < PLANE_OR_NOT)
    {
        IS_FACE =  true;
        printf("是人脸\n");
    }
    else
    {
        IS_FACE = false;
        printf("不是人脸\n");
    }
    // printf("%d\n", IS_FACE);
    return  IS_FACE;
}
