#include <numeric>
#include "matching2D.hpp"
#include "RingBuffer.h"
#include <map>

using namespace std;


// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
    	if(descriptorType.compare("DES_HOG") == 0){
    		normType = cv::NORM_L2;
    	}
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
    	if (descSource.type() != CV_32F)
		{ // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
			descSource.convertTo(descSource, CV_32F);
			descRef.convertTo(descRef, CV_32F);
		}

		// implement FLANN matching
		cout << "FLANN matching";
		matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
    	double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
    	// k nearest neighbors (k=2)
    	std::vector<std::vector<cv::DMatch> > knn_matches;
		// implement k-nearest-neighbor matching
		double t = (double)cv::getTickCount();
		matcher->knnMatch(descSource, descRef, knn_matches, 2); // Finds the k best match for each descriptor in desc1
		// filter matches using descriptor distance ratio test
		for(auto &twomatches: knn_matches){
			float ratio = twomatches[0].distance / twomatches[1].distance;
			if(ratio < 0.8){
				matches.push_back(twomatches[0]);
			}
		}
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		cout << " (KNN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
		cout << " (KNN) with discard percentage =" << (kPtsSource.size() - matches.size())/float(kPtsSource.size()) << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }else if(descriptorType.compare("BRIEF") == 0){
    	extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0){
    	extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0){
    	extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0){
    	extractor = cv::AKAZE::create();
    }
    else{
    	extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
	cv::Ptr<cv::FeatureDetector> detector;


	if (detectorType.compare("FAST") == 0){
		int threshold = 30;
		detector = cv::FastFeatureDetector::create(threshold);

	}else if(detectorType.compare("BRISK") == 0){
		detector = cv::BRISK::create();
	}else if(detectorType.compare("ORB") == 0){
		detector = cv::ORB::create();
	}else if(detectorType.compare("AKAZE") == 0){
		detector = cv::AKAZE::create();
	}else {
		detector = cv::xfeatures2d::SIFT::create();
	}

	double t = (double)cv::getTickCount();
	detector->detect(img, keypoints);
	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << detectorType<< " detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
	// visualize results
	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
		string windowName = detectorType + "  Corner Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){

	double t = (double)cv::getTickCount();

	// Detector parameters
	int blockSize = 4;     // for every pixel, a blockSize × blockSize neighborhood is considered
	int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
	int minResponse = 80; // minimum value for a corner in the 8bit scaled response matrix
	double k = 0.04;       // Harris parameter (see equation for details)

	// Detect Harris corners and normalize output
	cv::Mat dst, dst_norm, dst_norm_scaled;
	dst = cv::Mat::zeros(img.size(), CV_32FC1);
	cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
	cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());


	// locate local maxima in the Harris response matrix
	// and perform a non-maximum suppression (NMS) in a local neighborhood around
	// each maximum. The resulting coordinates shall be stored in a list of keypoints
	// of the type `vector<cv::KeyPoint>`.
	double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression

	for(auto r = 0; r< dst.rows; r++){
		for(auto c = 0; c< dst.cols; c++){
			int response = (int)dst_norm.at<float>(r, c);
			if(response <= minResponse){
				continue;
			}
			cv::KeyPoint newKeyPoint;

			newKeyPoint.pt = cv::Point2f(c, r);
			newKeyPoint.size = apertureSize * 2;
			newKeyPoint.response = response;

			bool bOverlap = false;

			for(auto it=keypoints.begin(); it != keypoints.end(); it++){

				auto overlap = cv::KeyPoint::overlap(newKeyPoint, *it);
				if(overlap <=  maxOverlap) {
					continue;
				}
				bOverlap = true;
				if(response > it->response){
					*it = newKeyPoint;
					break;
				}
			}//end of loop over keypoints vector
			if(!bOverlap){
				keypoints.push_back(newKeyPoint);
			}
		}//end of loop over cols
	}//end of loop over rows

	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
	// visualize results
	if (bVis)
	{
		cv::Mat visImage = img.clone();
		cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		string windowName = "Harris Corner Detector Results";
		cv::namedWindow(windowName, 6);
		imshow(windowName, visImage);
		cv::waitKey(0);
	}
}

static void feature_matching_test(string detectorType, string descriptorType, std::map<std::string, std::vector<float>> &res){

	float matched_num = 0;
	float detetecion_time = 0;
	float extraction_time = 0;
	string detector_extraction_name = detectorType + "_"+ descriptorType;

	/* INIT VARIABLES AND DATA STRUCTURES */

	// data location
	string dataPath = "../";

	// camera
	string imgBasePath = dataPath + "images/";
	string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
	string imgFileType = ".png";
	int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
	int imgEndIndex = 9;   // last file index to load
	int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

	// misc
	int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
	RingBuffer<DataFrame> dataBuffer(dataBufferSize); //class RingBuffer implements the logic of ring buffer, and provided some identical interfaces as std::vector
//    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
	bool bVis = false;            // visualize results

	/* MAIN LOOP OVER ALL IMAGES */

	for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
	{
		/* LOAD IMAGE INTO BUFFER */

		// assemble filenames for current index
		ostringstream imgNumber;
		imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
		string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

		// load image from file and convert to grayscale
		cv::Mat img, imgGray;
		img = cv::imread(imgFullFilename);
		cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

		//// STUDENT ASSIGNMENT
		//// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

		// push image into data frame buffer
		DataFrame frame;
		frame.cameraImg = imgGray;
		dataBuffer.push_back(frame);

		//// EOF STUDENT ASSIGNMENT
		cout << "#1 : LOAD IMAGE INTO BUFFER done, image ***"<< imgIndex + 1<<"***"<< endl;

		/* DETECT IMAGE KEYPOINTS */

		// extract 2D keypoints from current image
		vector<cv::KeyPoint> keypoints; // create empty feature list for current image


		//// STUDENT ASSIGNMENT
		//// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
		//// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
		bVis = false;
		double t = (double)cv::getTickCount();
		if (detectorType.compare("SHITOMASI") == 0)
		{
			detKeypointsShiTomasi(keypoints, imgGray, bVis);
		}
		else if(detectorType.compare("HARRIS") == 0){
			detKeypointsHarris(keypoints, imgGray, bVis);
		}
		else
		{
			detKeypointsModern(keypoints, imgGray, detectorType, bVis);
		}
		t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		detetecion_time += t;
		//// EOF STUDENT ASSIGNMENT

		//// STUDENT ASSIGNMENT
		//// TASK MP.3 -> only keep keypoints on the preceding vehicle

		// only keep keypoints on the preceding vehicle
		bool bFocusOnVehicle = true;
		cv::Rect vehicleRect(535, 180, 180, 150);
		vector<cv::KeyPoint> filtered_keypoints;
		if (bFocusOnVehicle)
		{
			for(auto &kp: keypoints){
				if(vehicleRect.contains(kp.pt)){
					filtered_keypoints.push_back(kp);
				}
			}
		}
		keypoints = std::move(filtered_keypoints);
		cout << detectorType<< " detector with n= " << keypoints.size() << " keypoints, filtered" << endl;

		//// EOF STUDENT ASSIGNMENT

		// push keypoints and descriptor for current frame to end of data buffer
		(dataBuffer.end() - 1)->keypoints = keypoints;
		cout << "#2 : DETECT KEYPOINTS done" << endl;

		/* EXTRACT KEYPOINT DESCRIPTORS */

		//// STUDENT ASSIGNMENT
		//// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
		//// -> BRIEF, ORB, FREAK, AKAZE, SIFT

		cv::Mat descriptors;
		 // BRIEF, ORB, FREAK, AKAZE, SIFT
		t = (double)cv::getTickCount();
		descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
		t = 1000 * ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		extraction_time += t;
		//// EOF STUDENT ASSIGNMENT

		// push descriptors for current frame to end of data buffer
		(dataBuffer.end() - 1)->descriptors = descriptors;

		cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

		if (dataBuffer.size() > 1) // wait until at least two images have been processed
		{

			/* MATCH KEYPOINT DESCRIPTORS */

			vector<cv::DMatch> matches;
			string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
			string descriptorSubType = "DES_BINARY"; // DES_BINARY, DES_HOG
			if(descriptorType.compare("SIFT") == 0) {
				descriptorSubType = "DES_HOG";
			}
			string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

			//// STUDENT ASSIGNMENT
			//// TASK MP.5 -> add FLANN matching in file matching2D.cpp
			//// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

			matchDescriptors(dataBuffer.tail_prev()->keypoints, (dataBuffer.end() - 1)->keypoints,
							 dataBuffer.tail_prev()->descriptors, (dataBuffer.end() - 1)->descriptors,
							 matches, descriptorSubType, matcherType, selectorType);

			//// EOF STUDENT ASSIGNMENT

			// store matches in current data frame
			(dataBuffer.end() - 1)->kptMatches = matches;

			matched_num += float(matches.size());

			cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;
			bVis = false;
		}

	} // eof loop over all images
	int img_num = imgEndIndex - imgStartIndex + 1;
	matched_num /= img_num - 1;
	detetecion_time /= img_num;
	extraction_time /= img_num;
	cout<<detector_extraction_name<<": matched_num="<<matched_num<<",detetecion_time="<<detetecion_time<<",extraction_time="<<extraction_time<<endl;

	res[detector_extraction_name] = {matched_num, detetecion_time, extraction_time};

}
void Performance_test()
{
	std::map<std::string, std::vector<float>> res;
	for(auto detector:{"SHITOMASI", "HARRIS","FAST", "BRISK", "ORB", "AKAZE", "SIFT"}){
		for(auto descriptor: {"BRIEF", "ORB", "FREAK", "SIFT"}){
			if(string(detector).compare("SIFT") == 0 && string(descriptor).compare("ORB") == 0) continue;
			feature_matching_test(detector, descriptor, res);
		}
	}

	for(auto detector:{"AKAZE"}){
		for(auto descriptor: {"AKAZE"}){
			feature_matching_test(detector, descriptor, res);
		}
	}

	for(auto test : res ){
		auto &detector_extraction_name = test.first;
		auto &data = test.second;
		int matched_num = int(data[0]);
		float detetecion_time = data[1];
		float extraction_time = data[2];
		float overall_time = detetecion_time + extraction_time;


		cout<< std::fixed << std::setprecision(1)<<detector_extraction_name<<", "<<overall_time<<", "<<matched_num<<", "<<detetecion_time<<", "<<extraction_time<<endl;
	}

}


