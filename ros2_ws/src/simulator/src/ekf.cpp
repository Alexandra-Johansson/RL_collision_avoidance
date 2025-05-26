#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"

using namespace std::chrono_literals;

class PublisherNode : public rclcpp::Node
{
  public:
    PublisherNode()
    : Node("EKF_publisher"), count_(0)
    {
      publisher_ = this->create_publisher<nav_msgs::msg::Path>("simulator_ekf", 10);
      timer_ = this->create_wall_timer(
      100ms, std::bind(&PublisherNode::timer_callback, this));
    }

  private:
    void timer_callback()
    {
      auto message = nav_msgs::msg::Path();
      geometry_msgs::msg::PoseStamped pose = geometry_msgs::msg::PoseStamped();
      message.header.stamp = rclcpp::Clock().now();
      message.header.frame_id = "EKF Path";

      pose.header.stamp = rclcpp::Clock().now();
      pose.header.frame_id = "EKF Pose";
      message.poses.push_back(pose);

      RCLCPP_INFO(this->get_logger(), "Publishing: EKF Path");
      publisher_->publish(message);
    }
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PublisherNode>());
  rclcpp::shutdown();
  return 0;
}