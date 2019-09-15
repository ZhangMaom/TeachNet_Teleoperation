from __future__ import print_function
import argparse
import sys
import time
import pickle
import glob
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import csv
import copy
from model.model import *
from utils import seg_hand_depth

import rospy
import moveit_commander
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from shadow_teleop.srv import *
from sr_robot_msgs.srv import RobotTeachMode


parser = argparse.ArgumentParser(description='deepShadowTeleop')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model-path', type=str, default='./weights/new_early_teach_teleop.model',
                   help='pre-trained model path')
# add robot lated args here

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

np.random.seed(int(time.time()))

input_size=100
embedding_size=128
joint_size=22

joint_upper_range = torch.tensor([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                  1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                  1.571, 1.047, 1.222, 0.209, 0.524, 1.571])
joint_lower_range = torch.tensor([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                  -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0])

model = torch.load(args.model_path, map_location='cpu')
model.device_ids = [args.gpu]
print('load model {}'.format(args.model_path))

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    joint_upper_range = joint_upper_range.cuda()
    joint_lower_range = joint_lower_range.cuda()


def test(model, img):
    model.eval()
    torch.set_grad_enabled(False)

    assert(img.shape == (input_size, input_size))
    img = img[np.newaxis, np.newaxis, ...]
    img = torch.Tensor(img)
    if args.cuda:
        img = img.cuda()

    # human part
    embedding_human, joint_human = model(img, is_human=True)
    joint_human = joint_human * (joint_upper_range - joint_lower_range) + joint_lower_range

    return joint_human.cpu().data.numpy()[0]


class Teleoperation():
    def __init__(self):
        self.mgi = moveit_commander.MoveGroupCommander("right_hand")
        self.bridge = CvBridge()
        self.mgi.set_named_target("open")
        self.mgi.go()

        # collision check and manipulate
        self.csl_client = rospy.ServiceProxy('CheckSelfCollision', checkSelfCollision)

        # Zero values in dictionary for tactile sensors (initialized at 0)
        self.force_zero = {"FF": 0, "MF": 0, "RF": 0, "LF": 0, "TH": 0}
        # Initialize values for current tactile values
        self.tactile_values = {"FF": 0, "MF": 0, "RF": 0, "LF": 0, "TH": 0}

        if rospy.has_param('contact_info'):
            self.contact_finger = rospy.get_param('contact_info')
        else:
            rospy.logerr("please set cotact info")
        self.ff_contacted = False
        self.mf_contacted = False
        self.rf_contacted = False
        self.lf_contacted = False
        self.th_contacted = False

        self.zero_tactile_sensors()

    def online_once(self):
        while True:     
            #       /camera/aligned_depth_to_color/image_raw
            img_data = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
            rospy.loginfo("Got an image ^_^")
            try:
                img = self.bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
            except CvBridgeError as e:
                rospy.logerr(e)

            ff_fixed = False
            mf_fixed = False
            rf_fixed = False
            lf_fixed = False
            th_fixed = False
            if not self.contact_finger[0]:
                ff_fixed = True
            if not self.contact_finger[1]:
                mf_fixed = True
            if not self.contact_finger[2]:
                rf_fixed = True
            if not self.contact_finger[3]:
                lf_fixed = True
            if not self.contact_finger[4]:
                th_fixed = True

            while not (ff_fixed and lf_fixed and rf_fixed and mf_fixed and th_fixed):
                if self.tactile_values['FF'] > self.force_zero['FF']:
                    self.ff_contacted = True
                if self.tactile_values['MF'] > self.force_zero['MF']:
                    self.mf_contacted = True
                if self.tactile_values['RF'] > self.force_zero['RF']:
                    self.rf_contacted = True
                if self.tactile_values['LF'] > self.force_zero['LF']:
                    self.lf_contacted = True
                if self.tactile_values['TH'] > self.force_zero['TH']:
                    self.th_contacted = True

                # preproces
                img = seg_hand_depth(img, 500, 1000, 10, 100, 4, 4, 250, True, 300)
                img = img.astype(np.float32)
                img = img / 255. * 2. - 1

                n = cv2.resize(img, (0, 0), fx=2, fy=2)
                n1 = cv2.normalize(n, n, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow("segmented human hand", n1)
                cv2.waitKey(1)

                # get the clipped joints
                goal = self.joint_cal(img, isbio=True)
                start = self.mgi.get_current_joint_values()

                group_variable_values = self.hand_group.get_current_joint_values()
                if (not self.ff_contacted) or (not ff_fixed):
                    ff_values = goal[2:6]
                else:
                    ff_values = group_variable_values[2:6]
                    ff_fixed = True

                if (not self.lf_contacted) and (not lf_fixed):
                    lf_values = goal[6:11]
                else:
                    lf_values = group_variable_values[6:11]
                    lf_fixed = True

                if (not self.mf_contacted) and (not mf_fixed):
                    mf_values = goal[11:15]
                else:
                    mf_values = group_variable_values[11:15]
                    mf_fixed = True

                if (not self.rf_contacted) and (not rf_fixed):
                    rf_values = goal[15:19]
                else:
                    rf_values = group_variable_values[15:19]
                    rf_fixed = True

                if (not self.th_contacted) and (not th_fixed):
                    th_values = goal[19:]
                else:
                    th_values = group_variable_values[19:]
                    th_fixed = True

                updated_variable_values = group_variable_values[
                                          0:2] + ff_values + lf_values + mf_values + rf_values + th_values

                try:
                    shadow_pos = self.csl_client(start, tuple(updated_variable_values))
                    if shadow_pos.result:
                        rospy.loginfo("Move Done!")
                    else:
                       rospy.logwarn("Failed to move!")
                except rospy.ServiceException as exc:
                   rospy.logwarn("Service did not process request: " + str(exc))

                rospy.loginfo("Next one please ---->")

            self.change_controller()
            rospy.sleep(1)
            rospy.set_param('start_tactile_feedback', "True")
            sys.exit()

    def joint_cal(self, img, isbio=False):
        # start = rospy.Time.now().to_sec()

        # run the model
        feature = test(model, img)
        # network_time = rospy.Time.now().to_sec() - start
        # print("network_time is ", network_time)

        joint = [0.0, 0.0]
        joint += feature.tolist()
        if isbio:
            joint[5] = 0.3498509706185152
            joint[10] = 0.3498509706185152
            joint[14] = 0.3498509706185152
            joint[18] = 0.3498509706185152
            joint[23] = 0.3498509706185152

        # joints crop
        joint[2] = self.clip(joint[2], 0.349, -0.349)
        joint[3] = self.clip(joint[3], 1.57, 0)
        joint[4] = self.clip(joint[4], 1.57, 0)
        joint[5] = self.clip(joint[5], 1.57, 0)

        joint[6] = self.clip(joint[6], 0.785, 0)

        joint[7] = self.clip(joint[7], 0.349, -0.349)
        joint[8] = self.clip(joint[8], 1.57, 0)
        joint[9] = self.clip(joint[9], 1.57, 0)
        joint[10] = self.clip(joint[10], 1.57, 0)

        joint[11] = self.clip(joint[11], 0.349, -0.349)
        joint[12] = self.clip(joint[12], 1.57, 0)
        joint[13] = self.clip(joint[13], 1.57, 0)
        joint[14] = self.clip(joint[14], 1.57, 0)

        joint[15] = self.clip(joint[15], 0.349, -0.349)
        joint[16] = self.clip(joint[16], 1.57, 0)
        joint[17] = self.clip(joint[17], 1.57, 0)
        joint[18] = self.clip(joint[18], 1.57, 0)

        joint[19] = self.clip(joint[19], 1.047, -1.047)
        joint[20] = self.clip(joint[20], 1.222, 0)
        joint[21] = self.clip(joint[21], 0.209, -0.209)
        joint[22] = self.clip(joint[22], 0.524, -0.524)
        joint[23] = self.clip(joint[23], 1.57, 0)

        return joint

    def clip(self, x, maxv=None, minv=None):
        if maxv is not None and x > maxv:
            x = maxv
        if minv is not None and x < minv:
            x = minv
        return x

    def read_tactile_values(self,msg):
        self.tactile_values['FF'] = msg.tactiles[0].pdc
        self.tactile_values['MF'] = msg.tactiles[1].pdc
        self.tactile_values['RF'] = msg.tactiles[2].pdc
        self.tactile_values['LF'] = msg.tactiles[3].pdc
        self.tactile_values['TH'] = msg.tactiles[4].pdc

    def zero_tactile_sensors(self):
        for x in xrange(1, 1000):
            # Read current state of tactile sensors to zero them
            if self.tactile_values['FF'] > self.force_zero['FF']:
                self.force_zero['FF'] = self.tactile_values['FF']
            if self.tactile_values['MF'] > self.force_zero['MF']:
                self.force_zero['MF'] = self.tactile_values['MF']
            if self.tactile_values['RF'] > self.force_zero['RF']:
                self.force_zero['RF'] = self.tactile_values['RF']
            if self.tactile_values['LF'] > self.force_zero['LF']:
                self.force_zero['LF'] = self.tactile_values['LF']
            if self.tactile_values['TH'] > self.force_zero['TH']:
                self.force_zero['TH'] = self.tactile_values['TH']

        self.force_zero['FF'] = self.force_zero['FF'] + 5
        self.force_zero['MF'] = self.force_zero['MF'] + 5
        self.force_zero['RF'] = self.force_zero['RF'] + 5
        self.force_zero['LF'] = self.force_zero['LF'] + 5
        self.force_zero['TH'] = self.force_zero['TH'] + 5

        rospy.loginfo("\n\nOK, ready for the grasp")

    # detect contact then stop the robot and change the controllers
    @ staticmethod
    def change_controller():
        rospy.wait_for_service('/teach_mode')
        try:
            teach_mode = rospy.ServiceProxy('/teach_mode', RobotTeachMode)
            result = teach_mode(1, "right_hand")
            if not result:
                print("run force control now")
            else:
                rospy.logerr("failed to start force control")
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)


def main():
    rospy.init_node('human_teleop_shadow')
    tele = Teleoperation()
    while not rospy.is_shutdown():
        tele.online_once()


if __name__ == "__main__":
    main()
