from __future__ import print_function, division
import rospy
from sr_robot_msgs.msg import BiotacAll
from std_msgs.msg import Float64
from IPython import embed


pressure_init = [2050, 2050, 2050, 2050, 2050]
window_size = 10
tactile_win = 22


class ForceFeedback:
    def __init__(self):
        feedback_start = False
        if rospy.has_param('start_tactile_feedback'):
            feedback_start = rospy.get_param('start_tactile_feedback')
        else:
            print("Waiting for forward teleoperation finished")

        if feedback_start:
            self.ff_pub = rospy.Publisher("/sh_rh_ffj3_effort_controller/command", Float64, queue_size=10)
            self.mf_pub = rospy.Publisher("/sh_rh_mfj3_effort_controller/command", Float64, queue_size=10)
            self.rf_pub = rospy.Publisher("/sh_rh_rfj3_effort_controller/command", Float64, queue_size=10)
            self.lf_pub = rospy.Publisher("/sh_rh_lfj3_effort_controller/command", Float64, queue_size=10)
            self.lf_pub = rospy.Publisher("/sh_rh_lfj3_effort_controller/command", Float64, queue_size=10)
            rospy.Subscribe("rh/tactile_filtered", BiotacAll, self.tactile_callback)
            rospy.Subscribe("rh/tactile_zero", Float64, self.zero_callback)

            rospy.on_shutdown(self.release_tendon())

            self.finger_name = ["ff", "mf", "rf", "lf", "th"]
            self.pacs = {"ff": [], "mf": [], "rf": [], "lf": [], "th": []}

            self.force_init = [0, 0, 0, 0, 0]
            if rospy.has_param('contact_info'):
                self.contact_info = rospy.get_param('contact_info')
            else:
                rospy.logerr("please set cotact info")
            self.slipping = False
            self.stable_grasp()
            rospy.spin()

    def tactile_callback(self,data):
        if len(self.pacs["ff"]) < tactile_win - 1:
            wait_data = True

        for i in range(5):
            # t = data.header.time_stamp
            pac = self.pacs[self.finger_name[i]]
            pac.append(data.tactiles[i].pac1)
            if not wait_data:
                pac.pop(0)
            slip_check = [pre > pressure_init[i] for pre in pac]
            if sum(slip_check) >= tactile_win:
                self.slipping = True
                rospy.logwarn("slipping is detected")

    def stable_grasp(self):
        # TODO: lift object and increase force based on slipping detection
        # TODO: lift the object while detect slipping at the same time?
        if self.slipping:
            if self.contact_info[0]:
                ff_force = self.force_init[0] + 20
                self.ff_pub.publish(ff_force)
            if self.contact_info[1]:
                mf_force = self.force_init[1] + 20
            if self.contact_info[2]:
                rf_force = self.force_init[2] + 20
            if self.contact_info[3]:
                lf_force = self.force_init[3] + 20
            if self.contact_info[4]:
                th_force = self.force_init[4] + 20
            # TODO: can I publish all force at the same time??

    # release the tendons of f3 joints
    def release_tendon(self):
        f = 0.0
        self.ff_pub.publish(f)
        self.mf_pub.publish(f)
        self.rf_pub.publish(f)
        self.lf_pub.publish(f)
        rospy.sleep(0.5)

    def zero_callback(self, data):
        for i in range(len(data)):
            self.cantact[i] = data[i]


if __name__ == "__main__":
    rospy.init_node('slip_detector', anonymous=True)
    force_feedback = ForceFeedback()
    while not rospy.is_shutdown():
        force_feedback.stable_grasp()



