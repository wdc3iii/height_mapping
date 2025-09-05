#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "HeightMapping.hpp"

class HeightMapNode : public rclcpp::Node {
public:
  HeightMapNode() : Node("largegrid_elevation_map_node"),
              tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {

    // --- params ---
    height_mapping::Params P;
    map_frame_    = declare_parameter<std::string>("map_frame", "map");
    base_frame_   = declare_parameter<std::string>("base_frame", "base_link");
    topic_cloud_  = declare_parameter<std::string>("topic_cloud", "/cloud_registered");
    lidar_frame_  = declare_parameter<std::string>("lidar_frame", "");

    P.res         = declare_parameter<double>("resolution", 0.05);
    P.Wb          = declare_parameter<int>("big_width", 400);
    P.Hb          = declare_parameter<int>("big_height", 400);
    P.max_h       = declare_parameter<double>("max_height", 2.0);
    P.z_min       = declare_parameter<double>("z_min", -1.0);
    P.z_max       = declare_parameter<double>("z_max",  2.0);
    P.drop_thresh = declare_parameter<double>("drop_thresh", 0.07);
    P.min_support = declare_parameter<int>("min_support", 4);
    P.shift_thresh= declare_parameter<double>("shift_thresh_m", 0.5);

    P.Wq          = declare_parameter<int>("sub_width", 200);
    P.Hq          = declare_parameter<int>("sub_height", 200);
    P.res_q       = declare_parameter<double>("sub_resolution", P.res);

    publish_rate_ = declare_parameter<double>("publish_rate_hz", 10.0);
    use_voxel_ds_ = declare_parameter<bool>("voxel_downsample", false);
    voxel_leaf_   = declare_parameter<double>("voxel_leaf", 0.05);
    transform_cloud_if_needed_ = declare_parameter<bool>("transform_cloud", true);

    mapper_ = std::make_shared<height_mapping::HeightMap>(P);

    // pubs/subs
    pub_raw_  = create_publisher<sensor_msgs::msg::Image>("height_grid/sub_raw", 1);
    pub_fill_ = create_publisher<sensor_msgs::msg::Image>("height_grid/sub_filled", 1);
    pub_meta_ = create_publisher<nav_msgs::msg::MapMetaData>("height_grid/meta", 1);

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      topic_cloud_, rclcpp::SensorDataQoS(),
      std::bind(&HeightMapNode::cloudCb, this, std::placeholders::_1));

    // timer
    const auto period = std::chrono::duration<double>(1.0 / publish_rate_);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::milliseconds>(period),
        std::bind(&HeightMapNode::onPublish, this));

    RCLCPP_INFO(get_logger(), "hieght_mapping node up: big %dx%d @%.2f, sub %dx%d @%.2f",
                P.Wb, P.Hb, P.res, P.Wq, P.Hq, P.res_q);
  }

private:
  bool getRobotPose(const rclcpp::Time& t, double& x, double& y, double& yaw) {
    try {
      auto T = tf_buffer_.lookupTransform(map_frame_, base_frame_, t, tf2::durationFromSec(0.05));
      x = T.transform.translation.x;
      y = T.transform.translation.y;
      const auto &q = T.transform.rotation;
      tf2::Quaternion qq(q.x, q.y, q.z, q.w);
      tf2::Matrix3x3 R(qq);
      double roll, pitch; R.getRPY(roll, pitch, yaw);
      return true;
    } catch (...) { return false; }
  }

  void cloudCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // TF: ensure origin & recenter
    double rx, ry, rYaw;
    if (!getRobotPose(msg->header.stamp, rx, ry, rYaw)) return;
    mapper_->ensureOrigin(rx, ry);
    mapper_->recenterIfNeeded(rx, ry);

    // Optional voxel DS (still in ROS layer)
    const sensor_msgs::msg::PointCloud2 *cloud_in = msg.get();
    sensor_msgs::msg::PointCloud2 cloud_ds;

    if (use_voxel_ds_) {
        pcl::PCLPointCloud2 pcl_pc2;
        pcl_conversions::toPCL(*msg, pcl_pc2);

        pcl::PCLPointCloud2::Ptr pcl_in(new pcl::PCLPointCloud2(pcl_pc2));  // <- PCL-compatible Ptr
        pcl::PCLPointCloud2 pcl_out;

        pcl::VoxelGrid<pcl::PCLPointCloud2> vg;
        vg.setInputCloud(pcl_in);
        vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
        vg.filter(pcl_out);

        sensor_msgs::msg::PointCloud2 cloud_ds;
        pcl_conversions::fromPCL(pcl_out, cloud_ds);
        cloud_in = &cloud_ds;
    }

    // Transform to map if required (on-the-fly numeric TF)
    geometry_msgs::msg::TransformStamped T_map_src;
    bool do_tf = false;
    if (transform_cloud_if_needed_) {
      const std::string src_frame = cloud_in->header.frame_id.empty() ? lidar_frame_ : cloud_in->header.frame_id;
      if (!src_frame.empty() && src_frame != map_frame_) {
        try {
          T_map_src = tf_buffer_.lookupTransform(map_frame_, src_frame, msg->header.stamp, tf2::durationFromSec(0.05));
          do_tf = true;
        } catch (...) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "TF missing %s->%s", map_frame_.c_str(), src_frame.c_str());
        }
      }
    }

    double r00=1, r01=0, r02=0, tx=0;
    double r10=0, r11=1, r12=0, ty=0;
    double r20=0, r21=0, r22=1, tz=0;
    if (do_tf) {
      const auto &t = T_map_src.transform.translation;
      const auto &q = T_map_src.transform.rotation;
      tf2::Quaternion qq(q.x, q.y, q.z, q.w);
      tf2::Matrix3x3 R(qq);
      r00 = R[0][0]; r01 = R[0][1]; r02 = R[0][2]; tx = t.x;
      r10 = R[1][0]; r11 = R[1][1]; r12 = R[1][2]; ty = t.y;
      r20 = R[2][0]; r21 = R[2][1]; r22 = R[2][2]; tz = t.z;
    }

    // Convert cloud to vector<Point3f> in map frame
    std::vector<height_mapping::Point3f> pts;
    pts.reserve(cloud_in->width * cloud_in->height);

    sensor_msgs::PointCloud2ConstIterator<float> it_x(*cloud_in, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*cloud_in, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*cloud_in, "z");

    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z) {
      double x = *it_x, y = *it_y, z = *it_z;
      if (do_tf) {
        const double xx = r00*x + r01*y + r02*z + tx;
        const double yy = r10*x + r11*y + r12*z + ty;
        const double zz = r20*x + r21*y + r22*z + tz;
        x = xx; y = yy; z = zz;
      }
      pts.push_back(height_mapping::Point3f{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
    }

    mapper_->ingestPoints(pts);
  }

  void onPublish() {
    // Pose at publish time
    double rx, ry, rYaw;
    if (!getRobotPose(now(), rx, ry, rYaw)) return;
    if (!mapper_->haveOrigin()) return;

    cv::Mat sub_raw, sub_filled;
    height_mapping::SubgridMeta meta;
    mapper_->generateSubgrid(rx, ry, rYaw, sub_raw, sub_filled, meta);

    // Publish images
    auto stamp = now();
    auto msg_raw = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", sub_raw).toImageMsg();
    msg_raw->header.stamp = stamp;
    msg_raw->header.frame_id = map_frame_;
    pub_raw_->publish(*msg_raw);

    auto msg_fill = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", sub_filled).toImageMsg();
    msg_fill->header.stamp = stamp;
    msg_fill->header.frame_id = map_frame_;
    pub_fill_->publish(*msg_fill);

    // Publish meta (origin + yaw encoded)
    nav_msgs::msg::MapMetaData mm;
    mm.map_load_time = stamp;
    mm.resolution = meta.resolution;
    mm.width  = meta.width;
    mm.height = meta.height;
    mm.origin.position.x = meta.origin_x;
    mm.origin.position.y = meta.origin_y;
    mm.origin.position.z = 0.0;
    // yaw-only quaternion
    double cy = std::cos(0.5*meta.yaw), sy = std::sin(0.5*meta.yaw);
    mm.origin.orientation.x = 0.0;
    mm.origin.orientation.y = 0.0;
    mm.origin.orientation.z = sy;
    mm.origin.orientation.w = cy;

    pub_meta_->publish(mm);
  }

private:
  // ROS
  std::string map_frame_, base_frame_, topic_cloud_, lidar_frame_;
  double publish_rate_{10.0};
  bool use_voxel_ds_{false}, transform_cloud_if_needed_{true};
  double voxel_leaf_{0.05};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_raw_, pub_fill_;
  rclcpp::Publisher<nav_msgs::msg::MapMetaData>::SharedPtr pub_meta_;
  rclcpp::TimerBase::SharedPtr timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Core mapper
  std::shared_ptr<height_mapping::HeightMap> mapper_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HeightMapNode>());
  rclcpp::shutdown();
  return 0;
}
