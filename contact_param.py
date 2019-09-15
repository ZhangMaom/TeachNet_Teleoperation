import rospy
import sys

info = [sys.argv[1], sys.argv[1], sys.argv[1], sys.argv[1], sys.argv[1]]

rospy.set_param('contact_info', info)
