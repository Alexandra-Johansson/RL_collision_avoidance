#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"

using namespace std::chrono_literals;

void getRandomPosition(float min, float max);
void getRandomVelocity();

class ObjectSimNode : public rclcpp::Node
{
  public:
    ObjectSimNode()
    : Node("object_sim"), count_(0)
    {
      publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("object_position", 10);
      timer_ = this->create_wall_timer(
      100ms, std::bind(&ObjectSimNode::timer_callback, this));
      getRandomPosition(2.0, 3.0);
      getRandomVelocity();
    }

  private:
    float update_intervall = 0.1;
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;
    float v_x = 0.0;
    float v_y = 0.0;
    float v_z = 0.0;
    float radius = 0.1;
    void timer_callback()
    {
      auto message = visualization_msgs::msg::Marker();

      message.header.stamp = rclcpp::Clock().now();
      message.header.frame_id = "map";
      message.ns = "object";
      message.id = 0;
      message.type = 2; // 2 = sphere shape
      message.action = 0; // 0 = add
      message.color.r = 255;
      message.color.a = 1.0;
      message.scale.x = radius;
      message.scale.y = radius;
      message.scale.z = radius;

      moveObj();
      updVel();

      message.pose.position.x = x;
      message.pose.position.y = y;
      message.pose.position.z = z;

      RCLCPP_INFO(this->get_logger(), "Publishing: Object Position");
      publisher_->publish(message);
    }

    
    float getRandomFloat(float min, float max) {
      std::random_device rd; // obtain a random number from hardware
      std::mt19937 gen(rd()); // seed the generator
      std::uniform_real_distribution<> distr(min, max); // define the range

      return distr(gen);
    }

    void getRandomPosition(float min, float max){
      float angle = getRandomFloat(0,2*3.14);
      float distance = getRandomFloat(min, max);
      x = distance*cos(angle);
      y = distance*sin(angle);
      z = 1.2;
    }

    void getRandomVelocity(){
      float norm  = sqrt(x*x + y*y);
      v_x = 3*getRandomFloat(-x/norm-0.2,-x/norm+0.2);
      v_y = 3*getRandomFloat(-y/norm-0.2,-y/norm+0.2);
      v_z = 4*getRandomFloat(1,2);
    }

    void moveObj(){
      x += v_x*update_intervall;
      y += v_y*update_intervall;
      z += v_z*update_intervall;
      if (z < radius){
        z = radius;
      }
    }

    void updVel(){
      v_x = v_x*0.95; // Lose about 5% due to resistances.
      v_y = v_y*0.95;

      if (z <= radius){
        v_z = -v_z*0.75;  // Bouncing on the floor loses about 25% of speed.
      }
      v_z -= 9.82*update_intervall;
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObjectSimNode>());
  rclcpp::shutdown();
  return 0;
}