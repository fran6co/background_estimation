#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include <time.h>
#include "pugixml.hpp"
#include "ReddyEstimator.h"
#include "ReddyClusterer.h"

namespace fs = boost::filesystem;


double AGE(const cv::Mat& image1, const cv::Mat& image2,const cv::Mat& mask){
    cv::Mat temp1, temp2;

    cv::cvtColor(image1,temp1,CV_BGR2GRAY);
    cv::cvtColor(image2,temp2,CV_BGR2GRAY);

    cv::Mat diff (temp1.size(),temp1.type(),cv::Scalar::all(0));

    cv::add(diff,cv::abs(temp1-temp2),diff,mask);

    return cv::mean(diff,mask).val[0];
}

int main(int argc, char ** argv)
{
    fs::path base_path ("/Users/fran6co/Code/segmenter");
    fs::path data_path = fs::system_complete( base_path / fs::path( "data/caviar/" ) );
    fs::path result_path;
    {
        time_t time_stamp = time(NULL);
        std::string filename;
        std::stringstream ss;
        ss << "results-"<< time_stamp << ".csv";
        filename = ss.str();

        result_path= fs::system_complete( base_path / fs::path( "results/" + filename ) );
    }

    std::ofstream results (result_path.string().c_str(), std::fstream::out | std::fstream::trunc);
    fs::directory_iterator end_iter;
    for ( fs::directory_iterator m_itr( data_path ); m_itr != end_iter; ++m_itr ) {
        if (!fs::is_regular_file( m_itr->status() ) || m_itr->path().extension() != ".xml") {
            continue;
        }

        pugi::xml_document doc;
        doc.load_file(m_itr->path().string().c_str());

        std::string dataset_name = doc.child("dataset").attribute("name").as_string();

        std::cout << "Clustering " << dataset_name << std::endl;

        ReddyClusterer clusterer;
        clusterer.N = 8;
        clusterer.pixel_diff = 5;
        clusterer.corr_coef_threshold = 0.8;

        int width, height, channels;

        int i = 0;
        for ( fs::directory_iterator dir_itr( data_path / dataset_name ); dir_itr != end_iter; ++dir_itr) {
            if ( fs::is_regular_file( dir_itr->status() ) && (dir_itr->path().extension() == ".bmp" || dir_itr->path().extension() == ".jpg")) {
                ++i;
                cv::Mat frame = cv::imread(dir_itr->path().string());
                width = frame.cols;
                height = frame.rows;
                channels = frame.channels();

                clusterer.process(frame);

                cv::imshow("video", frame);
                cv::waitKey(1);
            }
        }

        std::cout << "Estimating " << dataset_name << std::endl;

        ReddyEstimator estimator(8, 100, 3);
        estimator.iterations = 1;
        estimator.width = width;
        estimator.height = height;
        estimator.channels = channels;

        cv::Mat background = estimator.estimate(clusterer.clusters());
        cv::imshow("video", background);
        cv::waitKey(0);

        results << "test\tage" << std::endl;
        pugi::xml_node info_frame = doc.child("dataset").child("frame");
        i = 0;
        for ( fs::directory_iterator dir_itr( data_path / dataset_name ); dir_itr != end_iter; ++dir_itr ) {
            if ( fs::is_regular_file( dir_itr->status() ) && (dir_itr->path().extension() == ".bmp" || dir_itr->path().extension() == ".jpg")) {
                ++i;

                cv::Mat frame = cv::imread(dir_itr->path().string());
                cv::Mat mask (frame.size(), CV_8UC1, cv::Scalar::all(255));

                for(pugi::xml_node object = info_frame.child("objectlist").child("object"); object; object = object.next_sibling("object")){
                    pugi::xml_node box = object.child("box");

                    cv::Size size (box.attribute("w").as_int(), box.attribute("h").as_int());
                    cv::Point org (box.attribute("xc").as_int()-size.width/2,box.attribute("yc").as_int()-9-size.height/2);

                    if (org.y < 0) {
                        org.y = 0;
                    }

                    if (size.height > mask.rows - org.y) {
                        size.height = mask.rows - org.y;
                    }

                    cv::Mat boundingMask (mask, cv::Rect(org, size));
                    boundingMask.setTo(cv::Scalar::all(0));
                }

                results << m_itr->path().filename() <<"\t";

                double age = AGE(background,frame.clone(),mask.clone());

                results << age << std::endl;

                info_frame = info_frame.next_sibling("frame");
            }
        }
    }
}