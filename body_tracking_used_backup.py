########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
from easymocap.mytools import read_camera
from os.path import join
from pythonosc.dispatcher import Dispatcher
from typing import List, Any

dispatcher = Dispatcher()

def set_filter(address: str, *args: List[Any]) -> None:
    # We expect two float arguments
    if not len(args) == 2 or type(args[0]) is not float or type(args[1]) is not float:
        return

    # Check that address starts with filter
    if not address[:-1] == "/filter":  # Cut off the last character
        print("coucou")
        return

    value1 = args[0]
    value2 = args[1]
    filterno = address[-1]
    print(f"Setting filter {filterno} values: {value1}, {value2}")


dispatcher.map("/filter*", set_filter)  # Map wildcard address to set_filter function

# Set up server and client for testing
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient

# server = BlockingOSCUDPServer(("172.25.32.95", 12000), dispatcher) #172.25.32.95
# client = SimpleUDPClient("172.25.32.5", 12000) #172.25.32.5

server = BlockingOSCUDPServer(("169.254.108.82", 12000), dispatcher)  # 172.25.32.95
client = SimpleUDPClient("169.254.108.81", 12000)  # 172.25.32.5

msg = 0
# Send message and receive exactly one message (blocking)
client.send_message("/filter1", [1., 2.]) #
#server.handle_request()

client.send_message("/filter8", [6., -2.])
#server.handle_request()


def get_cameras_params():
    views = ['10028261', '10028262', '10028263', '10028260']#, '03'
    cam_path = "/home/user/acq_bench/Data_Processing_Tools/EasyMocap/apps/calibration/extri_data"
    cameras = read_camera(join(cam_path, 'intri.yml'), join(cam_path, 'extri.yml'), views)
    cameras = {key: cameras[key] for key in views}
    return cameras

if __name__ == "__main__":
    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    #init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    # To open the chosen camera
    #serial_to_id_corresp = {'10028263': '02', '10028261': '01',  '10028260': '00'} #'10028262': '03',
    serials = [10028261, 10028262, 10028263, 10028260]
    serial = '10028262'
    init_params.set_from_serial_number(int(serial))

    """cameras_params = get_cameras_params()
    extrinsics = cameras_params[serial]['RT']
    intrinsics = cameras_params[serial]['K']
    print(extrinsics.shape)
    print(intrinsics.shape)

    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T  # inverse of rot matrix is transpose
    R_inv_C = np.matmul(R_inv, C)
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    cam_proj_mat_homo = np.concatenate([extrinsics, [np.array([0, 0, 0, 1])]])
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)#[0:3]
    cam_proj_mat_inv_4f = sl.Matrix4f()
    for i in range(4):
        for j in range(4):
            cam_proj_mat_inv_4f[i, j]=cam_proj_mat_inv[i, j]
    transf = sl.Transform()
    transf.init_matrix(cam_proj_mat_inv_4f)


    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[0]
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)"""

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    #change the camera frame to world frame
    # positional_tracking_parameters.set_initial_world_transform(transf)


    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    #obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_ACCURATE
    obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 30

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
                 , display_resolution.height / camera_info.camera_resolution.height]
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()
    runtime = sl.RuntimeParameters()
    #runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    # runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA
    #positional_tracking_parameters.set_initial_world_transform(transf)
    while viewer.is_available():
        # Grab an image
        #if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)
            object_list = bodies.object_list
            client.send_message("N_Agent", len(object_list))
            print("N_Agent", len(object_list))
            all_confs = []
            for i in range(len(object_list)):
                print(len(object_list))
                print("agent id :" + str(object_list[i].id%(len(object_list))))
                all_confs.append(np.mean(object_list[i].keypoint_confidence))
                for j in range(18):
                    data = np.concatenate([object_list[i].keypoint_2d[j],np.array([object_list[i].keypoint[j][-1]]), np.array([object_list[i].keypoint_confidence[j]])], axis=-1)
                    # print(data.shape)
                    client.send_message("Agent_"+str(object_list[i].id%(len(object_list)))+"_Joint_"+str(j)+"_View_0",data.tolist())
                #print("object label : "+str(repr(object_list[i].label)))
                #print("agent id : " + str(repr(object_list[i].unique_object_id)))

                #print("keypoints 3D : "+str(object_list[i].keypoint))

            # Update GL view
            viewer.update_view(image, bodies)
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
            image_left_ocv_resize = cv2.resize(image_left_ocv, (640,360))
            cv2.imshow("ZED | 2D View", image_left_ocv_resize)
            cv2.waitKey(10)

    viewer.exit()

    image.free(sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
