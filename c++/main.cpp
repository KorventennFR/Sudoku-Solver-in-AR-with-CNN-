#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <dirent.h>

cv::Mat getGrid(cv::Mat img);
std::vector<cv::Point> getGridEdges(cv::Mat gridImg);
std::vector<cv::Mat> getAllBoxes(cv::Mat img);
void fusion(std::vector<cv::Vec2f> *lines, cv::Mat &gridImg);
std::vector<cv::Point> getCorners(std::vector<cv::Vec2f> lines, cv::Mat gridImg);
int detectionAndCreate(std::string path, int num);

int main(int argc, char** argv)
{

    cv::Mat sudoku, grey_sudoku;
    sudoku = cv::imread("C:/Users/KuroK/Documents/Travail/Projet_Recherche/SudokuC++/images/image9.jpg", cv::IMREAD_GRAYSCALE);
    if(!sudoku.data) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat grid = cv::Mat(sudoku.size(), CV_8UC1);
    cv::adaptiveThreshold(sudoku, grid, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 79, 2);

    cv::Mat bigBlob = getGrid(grid);

    std::vector<cv::Point> p = getGridEdges(bigBlob);

    cv::Mat unwrapped;
    cv::Mat lambda = cv::Mat::zeros(grid.rows, grid.cols, grid.type() );
    cv::Point2f inputPoints[4];
    cv::Point2f outputPoints[4];


    inputPoints[0] = cv::Point2f(p[0]);
    inputPoints[1] = cv::Point2f(p[1]);
    inputPoints[2] = cv::Point2f(p[2]);
    inputPoints[3] = cv::Point2f(p[3]);

    outputPoints[0] = cv::Point2f( 0,0 );
    outputPoints[1] = cv::Point2f( grid.cols-1,0);
    outputPoints[2] = cv::Point2f( grid.cols-1,grid.rows-1);
    outputPoints[3] = cv::Point2f(0,grid.rows-1);

    lambda = getPerspectiveTransform(inputPoints, outputPoints);
    warpPerspective(sudoku, unwrapped, lambda, unwrapped.size());

    cv::imshow("sudoku", unwrapped);

    cv::waitKey(0);


    // For test Purposes only

    /*
    int state;
    DIR* rep = opendir("C:/Users/KuroK/Documents/Travail/Projet_Recherche/SudokuC++/images/");

    if ( rep != NULL )
    {
        struct dirent *ent;
        int i = 0;
        while ( (ent = readdir(rep) ) != NULL )
        {
            std::string fichier = ent->d_name;
            std::string path = "C:/Users/KuroK/Documents/Travail/Projet_Recherche/SudokuC++/images/" + fichier;
            if (fichier.length() > 4){
                std::string extension = fichier.substr(fichier.length()-4, 4);
                if(extension == ".jpg") {
                    std::cout << fichier << std::endl;
                    state = detectionAndCreate(path, i);
                    i++;

                }
            }
            if(state == EXIT_FAILURE) break;
        }

        closedir(rep);
    }*/

    return 0;
}



int detectionAndCreate(std::string path, int num){
    cv::Mat sudoku, grey_sudoku;
    sudoku = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if(!sudoku.data) // Check for invalid input
    {
        std::cout << "Could not open or find the frame" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat grid = cv::Mat(sudoku.size(), CV_8UC1);
    //cv::GaussianBlur(sudoku, sudoku, cv::Size(3,3), 0);
    cv::adaptiveThreshold(sudoku, grid, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 79, 2);

    cv::Mat bigBlob = getGrid(grid);


    std::vector<cv::Point> p = getGridEdges(bigBlob);

    cv::Mat unwrapped;
    cv::Mat lambda = cv::Mat::zeros(grid.rows, grid.cols, grid.type() );
    cv::Point2f inputPoints[4];
    cv::Point2f outputPoints[4];


    inputPoints[0] = cv::Point2f(p[0]);
    inputPoints[1] = cv::Point2f(p[1]);
    inputPoints[2] = cv::Point2f(p[2]);
    inputPoints[3] = cv::Point2f(p[3]);

    outputPoints[0] = cv::Point2f( 0,0 );
    outputPoints[1] = cv::Point2f( grid.cols-1,0);
    outputPoints[2] = cv::Point2f( grid.cols-1,grid.rows-1);
    outputPoints[3] = cv::Point2f(0,grid.rows-1);

    /*for (auto & inputPoint : inputPoints) {
        cv::circle(grid, inputPoint, 8, cv::Scalar(255,255,255), -1);
    }*/

    lambda = getPerspectiveTransform(inputPoints, outputPoints);
    warpPerspective(sudoku, unwrapped, lambda, unwrapped.size());
    //std::vector<cv::Mat> boxesTab = getAllBoxes(unwrapped);

    //cv::imshow("Unwrapped Grid", unwrapped);
    //cv::adaptiveThreshold(unwrapped, unwrapped, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 79, 2);
    //cv::imshow("box", boxesTab[9]);
    /*

    std::string number = std::to_string(num);
    std::string name = number + ".jpg";
    std::string folderPath = "C:/Users/KuroK/Documents/Travail/Projet_Recherche/SudokuC++/OutputImg/";
    name = folderPath + name;
    cv::imwrite(name, unwrapped);

     */
    //cv::waitKey(0);

    return EXIT_SUCCESS;
}

std::vector<cv::Mat> getAllBoxes(cv::Mat img){
    int x,y,w,h,cpt,xOffsetP,yOffsetP,xOffsetN,yOffsetN;
    std::vector<cv::Mat> ret(81);
    //cv::threshold(img, img, 127, 255, 0);
    cv::adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 79, 2);
    w = (int)(img.cols/9.0);
    h = (int)(img.rows/9.0);
    x = 0;
    y = 0;
    cpt = 0;
    cv::Mat tmp;
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            xOffsetP = 3;
            xOffsetN = -3;
            yOffsetP = 3;
            yOffsetN = -3;
            if(x==0)xOffsetN=0;
            if(y==0)yOffsetN=0;
            if(x>=img.cols-2)xOffsetP=0;
            if(y>=img.rows-2)yOffsetP=0;
            tmp = img(cv::Rect(x+xOffsetN,y+yOffsetN,w+xOffsetP,h+yOffsetP));
            tmp.copyTo(ret[cpt]);
            cpt++;
            x = x+w;
        }
        x = 0;
        y = y+h;
    }

    return ret;
}

cv::Mat getGrid(cv::Mat img){
    int count=0;
    int max=-1;

    cv::Point maxPt;

    for(int y=0;y<img.size().height;y++){
        uchar *row = img.ptr(y);
        for(int x=0;x<img.size().width;x++){
            if(row[x]>=128){
                int area = floodFill(img, cv::Point(x,y), CV_RGB(0,0,64));

                if(area>max){
                    maxPt = cv::Point(x,y);
                    max = area;
                }
            }
        }

    }
    floodFill(img, maxPt, CV_RGB(255,255,255));
    for(int y=0;y<img.size().height;y++)
    {
        uchar *row = img.ptr(y);
        for(int x=0;x<img.size().width;x++)
        {
            if(row[x]==64 && x!=maxPt.x && y!=maxPt.y)
            {
                int area = floodFill(img, cv::Point(x,y), CV_RGB(0,0,0));
            }
        }
    }

     return img;
}

std::vector<cv::Point> getGridEdges(cv::Mat gridImg) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(gridImg, contours, hierarchy, cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    cv::Mat imgCountour = cv::Mat(gridImg.size(), CV_8UC1, cv::Scalar(0,0,0));
    cv::drawContours(imgCountour, contours, -1, cv::Scalar(255, 255, 255), 2);
    cv::Mat imgC = imgCountour.clone();

    std::vector<cv::Vec2f> lines; // will hold the results of the detection
    cv::HoughLines(imgCountour, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
    fusion(&lines, imgCountour);

    /*for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( imgC, pt1, pt2, cv::Scalar(255), 1, cv::LINE_AA);
    }*/

    //cv::imshow("Contours", imgC);
    //cv::waitKey(0);

    return getCorners(lines, gridImg);
}

void fusion(std::vector<cv::Vec2f> *lines, cv::Mat &gridImg) {
    std::vector<cv::Vec2f>::iterator current;
    for(current=lines->begin();current!=lines->end();current++){
        if((*current)[0]==0 && (*current)[1]==-100) continue; //if the line has been already merged, skip it
        float p1 = (*current)[0];
        float theta1 = (*current)[1];

        // We get two points from the line
        cv::Point pt1current, pt2current;
        if(theta1>CV_PI*45/180 && theta1<CV_PI*135/180){
            pt1current.x=0;

            pt1current.y = p1/sin(theta1);

            pt2current.x=gridImg.size().width;
            pt2current.y=-pt2current.x/tan(theta1) + p1/sin(theta1);
        }
        else{
            pt1current.y=0;

            pt1current.x=p1/cos(theta1);

            pt2current.y=gridImg.size().height;
            pt2current.x=-pt2current.y/tan(theta1) + p1/cos(theta1);

        }
        std::vector<cv::Vec2f>::iterator pos;
        for(pos=lines->begin();pos!=lines->end();pos++) {
            if (*current == *pos) continue;
            if(fabs((*pos)[0]-(*current)[0])<20 && fabs((*pos)[1]-(*current)[1])<CV_PI*10/180)
            {
                float p = (*pos)[0];
                float theta = (*pos)[1];
                cv::Point pt1, pt2;
                if((*pos)[1]>CV_PI*45/180 && (*pos)[1]<CV_PI*135/180)
                {
                    pt1.x=0;
                    pt1.y = p/sin(theta);
                    pt2.x=gridImg.size().width;
                    pt2.y=-pt2.x/tan(theta) + p/sin(theta);
                }
                else
                {
                    pt1.y=0;
                    pt1.x=p/cos(theta);
                    pt2.y=gridImg.size().height;
                    pt2.x=-pt2.y/tan(theta) + p/cos(theta);
                }
                if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<64*64) &&
                   ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) + (pt2.y-pt2current.y)*(pt2.y-pt2current.y)<64*64)){
                    // Merge the two
                    (*current)[0] = ((*current)[0]+(*pos)[0])/2;

                    (*current)[1] = ((*current)[1]+(*pos)[1])/2;

                    (*pos)[0]=0;
                    (*pos)[1]=-100;
                }
            }
        }
    }
}

std::vector<cv::Point> getCorners(std::vector<cv::Vec2f> lines, cv::Mat gridImg) {
    // Now detect the lines on extremes
    cv::Vec2f topEdge = cv::Vec2f(10000,10000);    double topYIntercept=1000000, topXIntercept=0;
    cv::Vec2f bottomEdge = cv::Vec2f(-10000,-10000);        double bottomYIntercept=0, bottomXIntercept=0;
    cv::Vec2f leftEdge = cv::Vec2f(10000,10000);    double leftXIntercept=1000000, leftYIntercept=0;
    cv::Vec2f rightEdge = cv::Vec2f(-10000,-10000);        double rightXIntercept=0, rightYIntercept=0;

    for(int i=0;i<lines.size();i++)
    {
        cv::Vec2f curLine = lines[i];

        float p=curLine[0];

        float theta=curLine[1];

        if(p==0 && theta==-100)
            continue;

        double xIntercept, yIntercept;
        xIntercept = p/cos(theta);
        yIntercept = p/(cos(theta)*sin(theta));

        if(theta>CV_PI*80/180 && theta<CV_PI*100/180){
            if(p<topEdge[0])
                topEdge = curLine;

            if(p>bottomEdge[0])
                bottomEdge = curLine;
        }
        else if(theta<CV_PI*10/180 || theta>CV_PI*170/180){
            if(xIntercept>rightXIntercept){
                rightEdge = curLine;
                rightXIntercept = xIntercept;
            }
            else if(xIntercept<=leftXIntercept){
                leftEdge = curLine;
                leftXIntercept = xIntercept;
            }
        }
    }

    cv::Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;

    int height=gridImg.size().height;

    int width=gridImg.size().width;

    if(leftEdge[1]!=0){
        left1.x=0;        left1.y=leftEdge[0]/sin(leftEdge[1]);
        left2.x=width;    left2.y=-left2.x/tan(leftEdge[1]) + left1.y;
    }
    else{
        left1.y=0;        left1.x=leftEdge[0]/cos(leftEdge[1]);
        left2.y=height;    left2.x=left1.x - height*tan(leftEdge[1]);

    }

    if(rightEdge[1]!=0){
        right1.x=0;        right1.y=rightEdge[0]/sin(rightEdge[1]);
        right2.x=width;    right2.y=-right2.x/tan(rightEdge[1]) + right1.y;
    }
    else{
        right1.y=0;        right1.x=rightEdge[0]/cos(rightEdge[1]);
        right2.y=height;    right2.x=right1.x - height*tan(rightEdge[1]);

    }

    bottom1.x=0;    bottom1.y=bottomEdge[0]/sin(bottomEdge[1]);

    bottom2.x=width;bottom2.y=-bottom2.x/tan(bottomEdge[1]) + bottom1.y;

    top1.x=0;        top1.y=topEdge[0]/sin(topEdge[1]);
    top2.x=width;    top2.y=-top2.x/tan(topEdge[1]) + top1.y;

    // Next, we find the intersection of  these four lines
    double leftA = left2.y-left1.y;
    double leftB = left1.x-left2.x;

    double leftC = leftA*left1.x + leftB*left1.y;

    double rightA = right2.y-right1.y;
    double rightB = right1.x-right2.x;

    double rightC = rightA*right1.x + rightB*right1.y;

    double topA = top2.y-top1.y;
    double topB = top1.x-top2.x;

    double topC = topA*top1.x + topB*top1.y;

    double bottomA = bottom2.y-bottom1.y;
    double bottomB = bottom1.x-bottom2.x;

    double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;

    std::vector<cv::Point> ret;

    // Intersection of left and top
    double detTopLeft = leftA*topB - leftB*topA;

    cv::Point ptTopLeft = cv::Point((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);
    ret.push_back(ptTopLeft);

    // Intersection of top and right
    double detTopRight = rightA*topB - rightB*topA;

    cv::Point ptTopRight = cv::Point((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);
    ret.push_back(ptTopRight);

    // Intersection of right and bottom
    double detBottomRight = rightA*bottomB - rightB*bottomA;
    cv::Point ptBottomRight = cv::Point((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);
    ret.push_back(ptBottomRight);

    // Intersection of bottom and left
    double detBottomLeft = leftA*bottomB-leftB*bottomA;
    cv::Point ptBottomLeft = cv::Point((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);
    ret.push_back(ptBottomLeft);

    return  ret;
}

