#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window

const int SMOOTHING_RADIUS = 30; //30 In frames. The larger the more stable the video, but less reactive to sudden panning

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }

    double x;
    double y;
    double a; // angle
};

int main(int argc, char **argv)
{
    if(argc < 2) {
        cout << "./VideoStab [video.avi]" << endl;
        return 0;
    }

    // For further analysis
    ofstream out_transform("prev_to_cur_transformation.txt");
    ofstream out_trajectory("trajectory.txt");
    ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
    ofstream out_new_transform("new_prev_to_cur_transformation.txt");

    VideoCapture cap(argv[1]);
    assert(cap.isOpened());

    VideoWriter wri("msample.mp4",
		cap.get(CV_CAP_PROP_FOURCC),
		cap.get(CV_CAP_PROP_FPS),
		Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH)*2+10,    // Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT)));

    Mat cur, cur_grey;
    Mat prev, prev_grey;

    cap >> prev;
    cvtColor(prev, prev_grey, COLOR_BGR2GRAY);

    // Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
    vector <TransformParam> prev_to_cur_transform; // previous to current

    int k=1;
    int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    Mat last_T;
    
    //int count = 1;
    while(true) {
	//if(count++>100)break;
        cap >> cur;

        if(cur.data == NULL) {
            break;
        }

        cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

        // vector from prev to cur
        vector <Point2f> prev_corner, cur_corner;
        vector <Point2f> prev_corner2, cur_corner2;
        vector <uchar> status;
        vector <float> err;
	
        //todo: try other Detectors 
        goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
        calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);

        // weed out bad matches
        for(size_t i=0; i < status.size(); i++) {
            if(status[i]) {
                prev_corner2.push_back(prev_corner[i]);
                cur_corner2.push_back(cur_corner[i]);
            }
        }
        
        // draw matches
        vector <KeyPoint> kp1, kp2;
        vector <DMatch> matches1to2;
        kp1.clear();kp2.clear();matches1to2.clear();
        for(size_t i=0; i < cur_corner2.size(); i++){
            matches1to2.push_back(DMatch(i,i,1.f));
            kp1.push_back(KeyPoint(prev_corner2[i], 1.f));
            kp2.push_back(KeyPoint(cur_corner2[i], 1.f));
        }
        Mat imatches;
        drawMatches(prev, kp1, cur, kp2, matches1to2, imatches);
        //imshow("matches", imatches);
        //waitKey(0);
        char match_str[300];
        sprintf(match_str, "matches/%08d.jpg", k);
        imwrite(match_str, imatches);


        // translation + rotation only
        Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing

        // in rare cases no transform is found. We'll just use the last known good transform.
        if(T.data == NULL) {
            last_T.copyTo(T);
        }

        T.copyTo(last_T);

        // decompose T
        double dx = T.at<double>(0,2);
        double dy = T.at<double>(1,2);
        double da = atan2(T.at<double>(1,0), T.at<double>(0,0)); //rotation

        prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

        out_transform << k << " " << dx << " " << dy << " " << da << endl;

        cur.copyTo(prev);
        cur_grey.copyTo(prev_grey);

        cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
        k++;
    }

    // Step 2 - Accumulate the transformations to get the image trajectory

    // Accumulated frame to frame transform
    double a = 0;
    double x = 0;
    double y = 0;

    vector <Trajectory> trajectory; // trajectory at all frames

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        trajectory.push_back(Trajectory(x,y,a));

        out_trajectory << (i+1) << " " << x << " " << y << " " << a << endl;
    }

    // Step 3 - Smooth out the trajectory using an averaging window
    vector <Trajectory> smoothed_trajectory; // trajectory at all frames

    for(size_t i=0; i < trajectory.size(); i++) {
        double sum_x = 0;
        double sum_y = 0;
        double sum_a = 0;
        int count = 0;

        for(int j=-SMOOTHING_RADIUS; j <= SMOOTHING_RADIUS; j++) {
            if(i+j >= 0 && i+j < trajectory.size()) {
                sum_x += trajectory[i+j].x;
                sum_y += trajectory[i+j].y;
                sum_a += trajectory[i+j].a;

                count++;
            }
        }

        double avg_a = sum_a / count;
        double avg_x = sum_x / count;
        double avg_y = sum_y / count;

        smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));

        out_smoothed_trajectory << (i+1) << " " << avg_x << " " << avg_y << " " << avg_a << endl;
    }

    // Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
    vector <TransformParam> new_prev_to_cur_transform;
    vector <TransformParam> new_cur2_to_cur_transform;

    // Accumulated frame to frame transform
    a = 0;
    x = 0;
    y = 0;

    for(size_t i=0; i < prev_to_cur_transform.size(); i++) {
        x += prev_to_cur_transform[i].dx;
        y += prev_to_cur_transform[i].dy;
        a += prev_to_cur_transform[i].da;

        // target - current
        double diff_x = smoothed_trajectory[i].x - x;
        double diff_y = smoothed_trajectory[i].y - y;
        double diff_a = smoothed_trajectory[i].a - a;

        double dx = prev_to_cur_transform[i].dx + diff_x;
        double dy = prev_to_cur_transform[i].dy + diff_y;
        double da = prev_to_cur_transform[i].da + diff_a;
        //todo: right?
        new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
        //new_prev_to_cur_transform.push_back(TransformParam(diff_x, diff_y, diff_a));

        out_new_transform << (i+1) << " " << dx << " " << dy << " " << da << endl;
    }

    new_cur2_to_cur_transform.push_back(TransformParam(0,0,0));
    for(size_t i=1; i < prev_to_cur_transform.size(); i++) {
        double diff_x = smoothed_trajectory[i].x - smoothed_trajectory[i-1].x;
        double diff_y = smoothed_trajectory[i].y - smoothed_trajectory[i-1].y;
        double diff_a = smoothed_trajectory[i].a - smoothed_trajectory[i-1].a;

        new_cur2_to_cur_transform.push_back(TransformParam(diff_x, diff_y, diff_a));

    }
     

    // Step 5 - Apply the new transformation to the video
    cap.set(CV_CAP_PROP_POS_FRAMES, 0);
    Mat T(2,3,CV_64F);//transform matrix
    Mat T_fix(2,3,CV_64F); //pre to curmpl


    int vert_border = 1 * prev.rows / prev.cols; // get the aspect ratio correct

    k=0;
    
    
    //new_pre.copyTo(cur(Range::all(), Range::all()));
    while(k < max_frames-1) { // don't process the very last frame, no valid transform
        cout << "Frame: " << k << "/" << max_frames << "drawing" << endl;
        cap >> cur;

        if(cur.data == NULL) {
            break;
        }

        T.at<double>(0,0) = cos(new_prev_to_cur_transform[k].da);
        T.at<double>(0,1) = -sin(new_prev_to_cur_transform[k].da);
        T.at<double>(1,0) = sin(new_prev_to_cur_transform[k].da);
        T.at<double>(1,1) = cos(new_prev_to_cur_transform[k].da);

        T.at<double>(0,2) = new_prev_to_cur_transform[k].dx;
        T.at<double>(1,2) = new_prev_to_cur_transform[k].dy;

        T_fix.at<double>(0,0) = cos(new_cur2_to_cur_transform[k].da);
        T_fix.at<double>(0,1) = -sin(new_cur2_to_cur_transform[k].da);
        T_fix.at<double>(1,0) = sin(new_cur2_to_cur_transform[k].da);
        T_fix.at<double>(1,1) = cos(new_cur2_to_cur_transform[k].da);

        T_fix.at<double>(0,2) = new_cur2_to_cur_transform[k].dx;
        T_fix.at<double>(1,2) = new_cur2_to_cur_transform[k].dy;

        Mat cur2;
        Mat new_pre = Mat::zeros(cur.rows, cur.cols, cur.type());;
        Mat pre_cur;        

        warpAffine(cur, cur2, T, cur.size());
         
        ////////////////////fix////////////
        /*
        //warpAffine(new_pre, pre_cur, T_fix, new_pre.size());

        
        for(int i=0; i<cur2.rows-1; ++i){
            for(int j=0; j<cur2.cols-1; ++j){
                Vec3b color = cur2.at<Vec3b>(Point(j,i));
                if(color[0]==0 && color[1]==0 && color[2]==0)
                {    
                //Vec3b new_color = pre_cur.at<Vec3b>(Point(j,i));
                Vec3b new_color = cur.at<Vec3b>(Point(j,i));
                cur2.at<Vec3b>(Point(j,i)) = new_color;
                //pre_cur.copyTo(cur(Range(i-1,i), Range(j-1,j)));
                }   
            }
        }
        */

        // Resize cur2 back to cur size, for better side by side comparison
        //resize(cur2, cur2, cur.size());

        // Now draw the original and stablised side by side for coolness
        Mat canvas = Mat::zeros(cur.rows, cur.cols*2+10, cur.type());

        cur.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
        cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));
        cur2.copyTo(new_pre(Range::all(), Range(0, cur2.cols)));     


	wri << canvas;
	//wri << cur2;   //single      
	char str[256];
        sprintf(str, "images/%08d.jpg", k);
        imwrite(str, canvas);

        //waitKey(20);

        k++;
    }
    ///cvReleaseVideoWriter(wri);
    return 0;
}

