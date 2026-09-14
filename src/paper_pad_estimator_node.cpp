#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32MultiArray.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <yaml-cpp/yaml.h>

namespace {

constexpr double kPi = 3.14159265358979323846;

double radians(double degrees) { return degrees * kPi / 180.0; }

double angleDifference(double first, double second) {
  return std::atan2(std::sin(first - second), std::cos(first - second));
}

struct MarkerModel {
  double size_m = 0.0;
  cv::Vec3d center_pad{0.0, 0.0, 0.0};
};

struct Candidate {
  int id = -1;
  cv::Vec3d marker_translation_camera{0.0, 0.0, 0.0};
  cv::Vec3d pad_translation_camera{0.0, 0.0, 0.0};
  cv::Matx33d pad_rotation_camera = cv::Matx33d::eye();
  tf2::Quaternion quaternion;
  double yaw = 0.0;
  double reprojection_rmse_px = 0.0;
  double quality = 0.0;
};

struct FusionResult {
  cv::Vec3d translation{0.0, 0.0, 0.0};
  tf2::Quaternion quaternion;
  std::vector<std::size_t> inliers;
  std::array<double, 3> translation_variance{{0.0, 0.0, 0.0}};
  double yaw_variance = 0.0;
};

tf2::Quaternion rotationToQuaternion(const cv::Matx33d& rotation) {
  tf2::Matrix3x3 matrix(rotation(0, 0), rotation(0, 1), rotation(0, 2),
                       rotation(1, 0), rotation(1, 1), rotation(1, 2),
                       rotation(2, 0), rotation(2, 1), rotation(2, 2));
  tf2::Quaternion quaternion;
  matrix.getRotation(quaternion);
  quaternion.normalize();
  return quaternion;
}

geometry_msgs::Pose makePose(const cv::Vec3d& translation,
                             const tf2::Quaternion& quaternion) {
  geometry_msgs::Pose pose;
  pose.position.x = translation[0];
  pose.position.y = translation[1];
  pose.position.z = translation[2];
  pose.orientation.x = quaternion.x();
  pose.orientation.y = quaternion.y();
  pose.orientation.z = quaternion.z();
  pose.orientation.w = quaternion.w();
  return pose;
}

int dictionaryId(const std::string& name) {
  static const std::unordered_map<std::string, int> dictionaries = {
      {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
      {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
      {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
      {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
      {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
      {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
      {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
      {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
      {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
      {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
      {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
      {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
      {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
      {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
      {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
      {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
  };
  const auto found = dictionaries.find(name);
  if (found == dictionaries.end()) {
    throw std::runtime_error("unsupported ArUco dictionary: " + name);
  }
  return found->second;
}

}  // namespace

class PaperPadEstimator {
 public:
  PaperPadEstimator() : private_node_("~") {
    std::string layout_file;
    std::string dictionary_name;
    private_node_.param<std::string>("layout_file", layout_file, "");
    private_node_.param<std::string>("dictionary", dictionary_name, "DICT_4X4_100");
    private_node_.param("pad_size_m", pad_size_m_, 0.80);
    private_node_.param("ransac_translation_threshold_m", translation_threshold_m_, 0.12);
    private_node_.param("ransac_yaw_threshold_deg", yaw_threshold_deg_, 12.0);
    private_node_.param("ransac_max_iterations", max_iterations_, 64);
    private_node_.param("ransac_min_inliers", min_inliers_, 1);
    private_node_.param("reprojection_error_floor_px", reprojection_floor_px_, 0.25);
    private_node_.param("max_reprojection_error_px", max_reprojection_error_px_, 5.0);
    private_node_.param("translation_stddev_floor_m", translation_stddev_floor_m_, 0.01);
    private_node_.param("rotation_stddev_floor_deg", rotation_stddev_floor_deg_, 2.0);
    private_node_.param("processing_budget_ms", processing_budget_ms_, 16.6667);
    private_node_.param("processing_width", processing_width_, 720);
    private_node_.param("processing_height", processing_height_, 720);
    private_node_.param("center_crop", center_crop_, true);
    private_node_.param("opencv_threads", opencv_threads_, 4);
    private_node_.param("publish_debug_image", publish_debug_, false);

    if (layout_file.empty()) {
      throw std::runtime_error("~layout_file is required");
    }
    if (pad_size_m_ <= 0.0) {
      throw std::runtime_error("~pad_size_m must be positive");
    }
    if (processing_width_ <= 0 || processing_height_ <= 0) {
      throw std::runtime_error("~processing_width and ~processing_height must be positive");
    }
    loadLayout(layout_file, dictionary_name);
    dictionary_ = cv::aruco::getPredefinedDictionary(dictionaryId(dictionary_name));
    detector_parameters_ = cv::aruco::DetectorParameters::create();
    detector_parameters_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    detector_parameters_->cornerRefinementWinSize = 5;
    detector_parameters_->cornerRefinementMaxIterations = 30;
    detector_parameters_->cornerRefinementMinAccuracy = 0.01;
    cv::setNumThreads(std::max(1, opencv_threads_));

    std::string image_topic;
    std::string camera_info_topic;
    private_node_.param<std::string>("image_topic", image_topic, "/landing/camera/image_raw");
    private_node_.param<std::string>("camera_info_topic", camera_info_topic,
                                    "/landing/camera/camera_info");
    image_subscriber_ = node_.subscribe(image_topic, 1, &PaperPadEstimator::imageCallback, this,
                                        ros::TransportHints().tcpNoDelay());
    camera_info_subscriber_ =
        node_.subscribe(camera_info_topic, 1, &PaperPadEstimator::cameraInfoCallback, this,
                        ros::TransportHints().tcpNoDelay());

    ids_publisher_ = node_.advertise<std_msgs::Int32MultiArray>("/landing/markers/ids", 2);
    poses_publisher_ =
        node_.advertise<geometry_msgs::PoseArray>("/landing/markers/poses_camera", 2);
    target_publisher_ = node_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
        "/landing/target_pose_camera", 2);
    visible_publisher_ = node_.advertise<std_msgs::Bool>("/landing/target_visible", 2);
    inlier_ids_publisher_ =
        node_.advertise<std_msgs::Int32MultiArray>("/landing/estimator/inlier_ids", 2);
    processing_publisher_ =
        node_.advertise<std_msgs::Float32>("/landing/estimator/processing_ms", 2);
    debug_publisher_ = node_.advertise<sensor_msgs::Image>("/landing/debug/image", 1);

    ROS_INFO("paper pad estimator ready: %zu markers, L=%.3fm, %s, processing=%dx%d, "
             "budget=%.3fms",
             markers_.size(), pad_size_m_, dictionary_name.c_str(), processing_width_,
             processing_height_, processing_budget_ms_);
  }

 private:
  void loadLayout(const std::string& path, const std::string& expected_dictionary) {
    const YAML::Node root = YAML::LoadFile(path);
    const double canvas_units = root["canvas_units"].as<double>();
    const std::string layout_dictionary = root["dictionary"].as<std::string>();
    if (layout_dictionary != expected_dictionary) {
      throw std::runtime_error("layout dictionary and estimator dictionary differ");
    }
    if (canvas_units <= 0.0 || !root["markers"] || root["markers"].size() == 0) {
      throw std::runtime_error("layout must contain at least one marker");
    }
    const double scale = pad_size_m_ / canvas_units;
    for (const YAML::Node& marker : root["markers"]) {
      const int id = marker["id"].as<int>();
      const double size = marker["size"].as<double>();
      const double center_x = marker["x"].as<double>() + size / 2.0;
      const double center_y = marker["y"].as<double>() + size / 2.0;
      MarkerModel model;
      model.size_m = size * scale;
      model.center_pad =
          cv::Vec3d((center_x - canvas_units / 2.0) * scale,
                    (canvas_units / 2.0 - center_y) * scale, 0.0);
      if (!markers_.emplace(id, model).second) {
        throw std::runtime_error("duplicate marker ID in layout");
      }
    }
  }

  void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& message) {
    if (message->width == 0 || message->height == 0 || message->K[0] <= 0.0 ||
        message->K[4] <= 0.0) {
      calibrated_ = false;
      return;
    }
    if (static_cast<int>(message->width) < processing_width_ ||
        static_cast<int>(message->height) < processing_height_) {
      ROS_ERROR_THROTTLE(2.0, "CameraInfo is %ux%u, smaller than requested processing crop %dx%d",
                         message->width, message->height, processing_width_, processing_height_);
      calibrated_ = false;
      return;
    }
    crop_x_ = center_crop_ ? (static_cast<int>(message->width) - processing_width_) / 2 : 0;
    crop_y_ = center_crop_ ? (static_cast<int>(message->height) - processing_height_) / 2 : 0;
    camera_matrix_ = (cv::Mat_<double>(3, 3) << message->K[0], message->K[1],
                      message->K[2] - crop_x_,
                      message->K[3], message->K[4], message->K[5], message->K[6],
                      message->K[7], message->K[8]);
    camera_matrix_.at<double>(1, 2) -= crop_y_;
    if (message->D.empty()) {
      distortion_ = cv::Mat::zeros(5, 1, CV_64F);
    } else {
      distortion_ = cv::Mat(message->D, true).reshape(1, static_cast<int>(message->D.size()));
    }
    camera_frame_ = message->header.frame_id;
    calibrated_ = true;
  }

  bool makeGrayView(const sensor_msgs::Image& message, cv::Mat& gray) const {
    cv::Mat full_gray;
    if (message.encoding == sensor_msgs::image_encodings::MONO8) {
      full_gray = cv::Mat(message.height, message.width, CV_8UC1,
                          const_cast<unsigned char*>(message.data.data()), message.step);
    } else {
      int channels = 0;
      int conversion = -1;
      if (message.encoding == sensor_msgs::image_encodings::BGR8) {
        channels = 3;
        conversion = cv::COLOR_BGR2GRAY;
      } else if (message.encoding == sensor_msgs::image_encodings::RGB8) {
        channels = 3;
        conversion = cv::COLOR_RGB2GRAY;
      } else if (message.encoding == sensor_msgs::image_encodings::BGRA8) {
        channels = 4;
        conversion = cv::COLOR_BGRA2GRAY;
      } else if (message.encoding == sensor_msgs::image_encodings::RGBA8) {
        channels = 4;
        conversion = cv::COLOR_RGBA2GRAY;
      } else {
        ROS_ERROR_THROTTLE(2.0, "unsupported image encoding: %s", message.encoding.c_str());
        return false;
      }
      const cv::Mat color(message.height, message.width, CV_MAKETYPE(CV_8U, channels),
                          const_cast<unsigned char*>(message.data.data()), message.step);
      cv::cvtColor(color, full_gray, conversion);
    }
    if (full_gray.cols < processing_width_ || full_gray.rows < processing_height_) {
      ROS_ERROR_THROTTLE(2.0, "image is %dx%d, smaller than processing crop %dx%d",
                         full_gray.cols, full_gray.rows, processing_width_, processing_height_);
      return false;
    }
    const int x = center_crop_ ? (full_gray.cols - processing_width_) / 2 : 0;
    const int y = center_crop_ ? (full_gray.rows - processing_height_) / 2 : 0;
    gray = full_gray(cv::Rect(x, y, processing_width_, processing_height_));
    return true;
  }

  Candidate estimateCandidate(int id, const std::vector<cv::Point2f>& image_corners,
                              const MarkerModel& model) const {
    const float half = static_cast<float>(model.size_m / 2.0);
    const std::vector<cv::Point3f> object_corners = {
        {-half, half, 0.0F}, {half, half, 0.0F},
        {half, -half, 0.0F}, {-half, -half, 0.0F}};
    cv::Mat rotation_vector;
    cv::Mat translation_vector;
    bool solved = cv::solvePnP(object_corners, image_corners, camera_matrix_, distortion_,
                               rotation_vector, translation_vector, false,
                               cv::SOLVEPNP_IPPE_SQUARE);
    if (!solved) {
      throw std::runtime_error("solvePnP failed");
    }
    cv::solvePnPRefineLM(object_corners, image_corners, camera_matrix_, distortion_,
                         rotation_vector, translation_vector);

    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation_vector, rotation_matrix);
    cv::Matx33d rotation;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        rotation(row, column) = rotation_matrix.at<double>(row, column);
      }
    }
    const cv::Vec3d marker_translation(translation_vector.at<double>(0),
                                       translation_vector.at<double>(1),
                                       translation_vector.at<double>(2));
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_corners, rotation_vector, translation_vector, camera_matrix_,
                      distortion_, projected);
    double squared_error = 0.0;
    for (std::size_t index = 0; index < projected.size(); ++index) {
      const cv::Point2f delta = projected[index] - image_corners[index];
      squared_error += delta.dot(delta);
    }
    const double reprojection_rmse = std::sqrt(squared_error / projected.size());
    const double pixel_area = std::abs(cv::contourArea(image_corners));

    Candidate candidate;
    candidate.id = id;
    candidate.marker_translation_camera = marker_translation;
    candidate.pad_rotation_camera = rotation;
    candidate.pad_translation_camera = marker_translation - rotation * model.center_pad;
    candidate.quaternion = rotationToQuaternion(rotation);
    candidate.yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    candidate.reprojection_rmse_px = reprojection_rmse;
    candidate.quality =
        std::sqrt(std::max(1.0, pixel_area)) / std::max(reprojection_floor_px_, reprojection_rmse);
    return candidate;
  }

  FusionResult weightedFusion(const std::vector<Candidate>& candidates,
                              const std::vector<std::size_t>& indexes) const {
    FusionResult result;
    result.inliers = indexes;
    double total_weight = 0.0;
    double sine_sum = 0.0;
    double cosine_sum = 0.0;
    std::array<double, 4> quaternion_sum{{0.0, 0.0, 0.0, 0.0}};
    const tf2::Quaternion reference = candidates[indexes.front()].quaternion;
    for (std::size_t index : indexes) {
      const Candidate& candidate = candidates[index];
      const double weight = candidate.quality;
      total_weight += weight;
      result.translation += candidate.pad_translation_camera * weight;
      sine_sum += weight * std::sin(candidate.yaw);
      cosine_sum += weight * std::cos(candidate.yaw);
      tf2::Quaternion quaternion = candidate.quaternion;
      if (reference.dot(quaternion) < 0.0) {
        quaternion = tf2::Quaternion(-quaternion.x(), -quaternion.y(), -quaternion.z(),
                                     -quaternion.w());
      }
      quaternion_sum[0] += weight * quaternion.x();
      quaternion_sum[1] += weight * quaternion.y();
      quaternion_sum[2] += weight * quaternion.z();
      quaternion_sum[3] += weight * quaternion.w();
    }
    if (total_weight <= 0.0) {
      throw std::runtime_error("zero fusion weight");
    }
    result.translation *= 1.0 / total_weight;
    result.quaternion = tf2::Quaternion(quaternion_sum[0], quaternion_sum[1], quaternion_sum[2],
                                       quaternion_sum[3]);
    result.quaternion.normalize();
    const double mean_yaw = std::atan2(sine_sum, cosine_sum);
    for (std::size_t index : indexes) {
      const Candidate& candidate = candidates[index];
      const double normalized_weight = candidate.quality / total_weight;
      const cv::Vec3d residual = candidate.pad_translation_camera - result.translation;
      for (int axis = 0; axis < 3; ++axis) {
        result.translation_variance[axis] += normalized_weight * residual[axis] * residual[axis];
      }
      const double yaw_error = angleDifference(candidate.yaw, mean_yaw);
      result.yaw_variance += normalized_weight * yaw_error * yaw_error;
    }
    return result;
  }

  FusionResult ransacFusion(const std::vector<Candidate>& candidates,
                            std::uint32_t frame_seed) const {
    if (candidates.size() == 1) {
      return weightedFusion(candidates, {0});
    }
    std::vector<std::size_t> hypotheses(candidates.size());
    std::iota(hypotheses.begin(), hypotheses.end(), 0);
    std::mt19937 generator(frame_seed);
    std::shuffle(hypotheses.begin(), hypotheses.end(), generator);
    const std::size_t iterations =
        std::min<std::size_t>(hypotheses.size(), std::max(1, max_iterations_));
    const double yaw_threshold = radians(yaw_threshold_deg_);
    double best_score = -1.0;
    std::vector<std::size_t> best_inliers;
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
      const Candidate& hypothesis = candidates[hypotheses[iteration]];
      std::vector<std::size_t> inliers;
      double score = 0.0;
      for (std::size_t index = 0; index < candidates.size(); ++index) {
        const Candidate& candidate = candidates[index];
        const double translation_error =
            cv::norm(candidate.pad_translation_camera - hypothesis.pad_translation_camera);
        const double yaw_error = std::abs(angleDifference(candidate.yaw, hypothesis.yaw));
        if (translation_error <= translation_threshold_m_ && yaw_error <= yaw_threshold) {
          inliers.push_back(index);
          score += candidate.quality;
        }
      }
      if (score > best_score ||
          (score == best_score && inliers.size() > best_inliers.size())) {
        best_score = score;
        best_inliers = std::move(inliers);
      }
    }
    if (static_cast<int>(best_inliers.size()) < min_inliers_) {
      throw std::runtime_error("RANSAC found too few inliers");
    }
    return weightedFusion(candidates, best_inliers);
  }

  void publishDebug(const sensor_msgs::Image& message, const cv::Mat& gray,
                    const std::vector<std::vector<cv::Point2f>>& corners,
                    const std::vector<int>& ids) {
    if (!publish_debug_ || debug_publisher_.getNumSubscribers() == 0) {
      return;
    }
    cv::Mat debug;
    cv::cvtColor(gray, debug, cv::COLOR_GRAY2BGR);
    if (!ids.empty()) {
      cv::aruco::drawDetectedMarkers(debug, corners, ids);
    }
    sensor_msgs::Image output;
    output.header = message.header;
    output.height = static_cast<std::uint32_t>(debug.rows);
    output.width = static_cast<std::uint32_t>(debug.cols);
    output.encoding = sensor_msgs::image_encodings::BGR8;
    output.is_bigendian = false;
    output.step = static_cast<std::uint32_t>(debug.cols * debug.elemSize());
    output.data.assign(debug.datastart, debug.dataend);
    debug_publisher_.publish(output);
  }

  void publishTiming(double elapsed_ms) {
    std_msgs::Float32 timing;
    timing.data = static_cast<float>(elapsed_ms);
    processing_publisher_.publish(timing);
    accumulated_ms_ += elapsed_ms;
    ++frame_count_;
    if (elapsed_ms > processing_budget_ms_) {
      ++over_budget_count_;
    }
    if (frame_count_ % 60 == 0) {
      ROS_INFO("paper pad estimator: mean %.3fms, over %.3fms budget %zu/60 frames",
               accumulated_ms_ / 60.0, processing_budget_ms_, over_budget_count_);
      accumulated_ms_ = 0.0;
      over_budget_count_ = 0;
    }
  }

  void imageCallback(const sensor_msgs::ImageConstPtr& message) {
    const auto started = std::chrono::steady_clock::now();
    cv::Mat gray;
    if (!makeGrayView(*message, gray)) {
      return;
    }
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> rejected;
    std::vector<int> ids;
    cv::aruco::detectMarkers(gray, dictionary_, corners, ids, detector_parameters_, rejected);

    std_msgs::Int32MultiArray ids_message;
    ids_message.data.assign(ids.begin(), ids.end());
    ids_publisher_.publish(ids_message);
    bool known_visible = false;
    for (int id : ids) {
      known_visible = known_visible || markers_.count(id) > 0;
    }

    std::vector<Candidate> candidates;
    geometry_msgs::PoseArray marker_poses;
    marker_poses.header = message->header;
    if (!camera_frame_.empty()) {
      marker_poses.header.frame_id = camera_frame_;
    }
    if (calibrated_) {
      for (std::size_t index = 0; index < ids.size(); ++index) {
        const auto model = markers_.find(ids[index]);
        if (model == markers_.end()) {
          continue;
        }
        try {
          Candidate candidate = estimateCandidate(ids[index], corners[index], model->second);
          if (candidate.marker_translation_camera[2] <= 0.0 ||
              candidate.reprojection_rmse_px > max_reprojection_error_px_) {
            continue;
          }
          marker_poses.poses.push_back(
              makePose(candidate.marker_translation_camera, candidate.quaternion));
          candidates.push_back(std::move(candidate));
        } catch (const cv::Exception& exception) {
          ROS_WARN_THROTTLE(2.0, "marker PnP failed: %s", exception.what());
        } catch (const std::exception& exception) {
          ROS_WARN_THROTTLE(2.0, "marker pose rejected: %s", exception.what());
        }
      }
    } else if (known_visible) {
      ROS_WARN_THROTTLE(2.0, "markers visible but CameraInfo has no valid K; pose withheld");
    }
    if (!marker_poses.poses.empty()) {
      poses_publisher_.publish(marker_poses);
    }

    bool valid_pose = false;
    if (!candidates.empty()) {
      try {
        const std::uint32_t seed = message->header.seq ^ message->header.stamp.nsec;
        const FusionResult fusion = ransacFusion(candidates, seed);
        geometry_msgs::PoseWithCovarianceStamped target;
        target.header = marker_poses.header;
        target.pose.pose = makePose(fusion.translation, fusion.quaternion);
        const double translation_floor =
            translation_stddev_floor_m_ * translation_stddev_floor_m_;
        const double rotation_floor =
            radians(rotation_stddev_floor_deg_) * radians(rotation_stddev_floor_deg_);
        target.pose.covariance[0] = std::max(translation_floor, fusion.translation_variance[0]);
        target.pose.covariance[7] = std::max(translation_floor, fusion.translation_variance[1]);
        target.pose.covariance[14] = std::max(translation_floor, fusion.translation_variance[2]);
        target.pose.covariance[21] = rotation_floor;
        target.pose.covariance[28] = rotation_floor;
        target.pose.covariance[35] = std::max(rotation_floor, fusion.yaw_variance);
        target_publisher_.publish(target);
        valid_pose = true;

        std_msgs::Int32MultiArray inlier_ids;
        for (std::size_t index : fusion.inliers) {
          inlier_ids.data.push_back(candidates[index].id);
        }
        inlier_ids_publisher_.publish(inlier_ids);
      } catch (const std::exception& exception) {
        ROS_WARN_THROTTLE(2.0, "paper pad fusion failed: %s", exception.what());
      }
    }
    std_msgs::Bool visible;
    visible.data = valid_pose;
    visible_publisher_.publish(visible);
    publishDebug(*message, gray, corners, ids);
    const auto finished = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(finished - started).count();
    publishTiming(elapsed_ms);
  }

  ros::NodeHandle node_;
  ros::NodeHandle private_node_;
  ros::Subscriber image_subscriber_;
  ros::Subscriber camera_info_subscriber_;
  ros::Publisher ids_publisher_;
  ros::Publisher poses_publisher_;
  ros::Publisher target_publisher_;
  ros::Publisher visible_publisher_;
  ros::Publisher inlier_ids_publisher_;
  ros::Publisher processing_publisher_;
  ros::Publisher debug_publisher_;

  std::unordered_map<int, MarkerModel> markers_;
  cv::Ptr<cv::aruco::Dictionary> dictionary_;
  cv::Ptr<cv::aruco::DetectorParameters> detector_parameters_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_;
  std::string camera_frame_;
  bool calibrated_ = false;
  bool publish_debug_ = false;
  double pad_size_m_ = 0.80;
  double translation_threshold_m_ = 0.12;
  double yaw_threshold_deg_ = 12.0;
  int max_iterations_ = 64;
  int min_inliers_ = 1;
  double reprojection_floor_px_ = 0.25;
  double max_reprojection_error_px_ = 5.0;
  double translation_stddev_floor_m_ = 0.01;
  double rotation_stddev_floor_deg_ = 2.0;
  double processing_budget_ms_ = 16.6667;
  int opencv_threads_ = 4;
  int processing_width_ = 720;
  int processing_height_ = 720;
  int crop_x_ = 0;
  int crop_y_ = 0;
  bool center_crop_ = true;
  std::size_t frame_count_ = 0;
  std::size_t over_budget_count_ = 0;
  double accumulated_ms_ = 0.0;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "paper_pad_estimator");
  try {
    PaperPadEstimator estimator;
    ros::spin();
  } catch (const std::exception& exception) {
    ROS_FATAL("paper pad estimator startup failed: %s", exception.what());
    return 1;
  }
  return 0;
}
