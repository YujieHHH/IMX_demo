import time

import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import multiprocessing
from multiprocessing import Pool, Process, Manager, Queue
from pynput import keyboard
from pythonosc.dispatcher import Dispatcher


dispatcher = Dispatcher()
from typing import List, Any


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


process_list = []
viewer_zed = []
num_id = None
stop_signal = False


# id_0 = []
# id_1 = []
# skeleton_det_0 = []
# skeleton_det_1 = []
# len_0 = None
# len_1 = None

def my_press(key):
    global stop_signal
    # stop_signal = True
    print("Calling keyboard Listener.....")
    # if key == keyboard.Key.esc:
    if key == keyboard.Key.esc:
        # raise MyException(key)
        stop_signal = True
        return False


def find_index_by_id(lst, target_id):
    # print(type(lst))
    # lst = list(lst)
    # print(type(lst))
    # print(len(lst))
    # print(type(target_id))
    # print('test fonction ', lst[0].id)
    # print('test target_id', target_id)
    for index, id in enumerate(lst):
        # print('fonc find_index_by_id')
        if id == int(target_id):
            return index
    return -1


def find_person_not_in(lst, target_id):
    for i, obj in enumerate(lst):
        if target_id == obj.id:
            return False  ##not absent
    return True


def cam1_to_cam0(width, kpt):
    for i, joint in enumerate(kpt):
        joint[0] = width - joint[0]
    return kpt


def if_cam_0_absent_detected(cam0_dict):
    if any(value is None for value in cam0_dict.values()):
        return True
    else:
        return False


def if_cam_1_absent_detected(cam1_dict):
    if any(value is None for value in cam1_dict.values()):
        return True
    else:
        return False


def grab_run(index, serials_zed, open_zed, finished_proc, meta_data, skeleton_output, len_0, len_1, num_id, person_cam0,
             person_cam1, person_position, person_cam_0_conf, person_cam_1_conf,
             skeleton_2dkpt_0, skeleton_2dkpt_1, skeleton_iddet_0, skeleton_iddet_1, client,
             pos_x_0_temp, pos_x_1_temp, id_0_temp, id_1_temp, key_temp):
    init_params = sl.InitParameters()
    init_params.camera_fps = 30
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER  # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.set_from_serial_number(serials_zed[index])

    print("opening zed {} ".format(index))
    zed = sl.Camera()
    err = zed.open(init_params)
    time.sleep(2)
    # if err != sl.ERROR_CODE.SUCCESS:
    #     exit(1)
    # time.sleep(1)
    if err != sl.ERROR_CODE.SUCCESS:  # if not success
        print("camera opening error : " + str(repr(err)))
        print("Index: {}".format(index))
        zed.reboot(serials_zed[index])
        err = zed.open(init_params)  # if failed, reboot camera
        zed.close()
        exit(1)
    print("Zed {} is opened...".format(index))

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True  # Smooth skeleton move
    obj_param.enable_tracking = True  # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 30

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    image_width = camera_info.camera_resolution.width
    image_height = camera_info.camera_resolution.height
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
        , display_resolution.height / camera_info.camera_resolution.height]

    # Create OpenGL viewer
    # if index == 0:
    viewer = gl.GLViewer()
    viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking, obj_param.body_format)

    # Create ZED objects filled in the main loop
    # bodies = sl.Objects()
    image = sl.Mat()
    depth = sl.Mat()
    runtime = sl.RuntimeParameters()

    temp_sk_1 = []
    pos_x_0 = []
    pos_x_1 = []

    # print('11111111111111111')

    while viewer.is_available():

        open_zed[index] = True
        # wait all zed opened
        while not all(open_zed):
            continue
        print("all cameras are opened")

        while not meta_data['yes_no']:
            continue
        image.free(sl.MEM.CPU)

        bodies = sl.Objects()
        image = sl.Mat()

        # if index == 0:
        #     viewer = gl.GLViewer()
        #     viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking, obj_param.body_format)


        ################################### Initialization ########################################
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)
            # if index == 0:
            # viewer.update_view(image, bodies)
            if index == 0:
                skeleton_det_0 = bodies.object_list
                len_0.value = len(bodies.object_list)
            if index == 1:
                skeleton_det_1 = bodies.object_list  # local list
                len_1.value = len(skeleton_det_1)
            while len_0.value != 2 or len_1.value != 2:
                if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                    zed.retrieve_objects(bodies, obj_runtime_param)
                    if index == 0:
                        skeleton_det_0 = bodies.object_list
                        len_0.value = len(bodies.object_list)
                    if index == 1:
                        skeleton_det_1 = bodies.object_list  # local list
                        len_1.value = len(skeleton_det_1)
                print('wait for the second person detected......')
            if index == 0:
                skeleton_2dkpt_0[:] = []
                skeleton_iddet_0[:] = []
                pos_x_0.clear()

                skeleton_det_0 = bodies.object_list  # local list
                len_0.value = len(bodies.object_list)  # global value
                for obj in skeleton_det_0:
                    skeleton_2dkpt_0.append(obj.keypoint_2d)
                    skeleton_iddet_0.append(obj.id)
                    # skeleton_conf_0.append(obj.confidence)
                print('index0', len_0.value)
                for i in range(len_0.value):
                    pos_x_0.append(skeleton_det_0[i].head_position[0])  # head x position
                pos_x_0.sort()  # head from left to right
                for i in range(len_0.value):
                    for skeleton in skeleton_det_0:
                        if pos_x_0[i] == skeleton.head_position[0]:
                            person_cam0[str(i)] = skeleton.id  #### a list contains the id from left to right
                            person_cam_0_conf[str(i)] = skeleton.confidence
                finished_proc[index] = True
            if index == 1:
                skeleton_2dkpt_1[:] = []
                skeleton_iddet_1[:] = []
                pos_x_1.clear()

                skeleton_det_1 = bodies.object_list  # local list
                len_1.value = len(skeleton_det_1)
                for obj in skeleton_det_1:
                    skeleton_2dkpt_1.append(obj.keypoint_2d)
                    skeleton_iddet_1.append(obj.id)
                print('index1', len_1.value)
                for i in range(len_1.value):
                    pos_x_1.append(skeleton_det_1[i].head_position[0])
                pos_x_1.sort(reverse=True)  # head from right to left
                for i in range(len_1.value):
                    for skeleton in skeleton_det_1:
                        # print('index1, and sk_id = ', skeleton.id)
                        if pos_x_1[i] == skeleton.head_position[0]:
                            person_cam1[str(i)] = skeleton.id  #### a list contains the id from left to right
                            # print('test person cam1', person_cam1)
                            person_cam_1_conf[str(i)] = skeleton.confidence
                finished_proc[index] = True

        while not all(finished_proc):
            continue
        finished_proc[index] = False


        print('len 0 and len 1', len(skeleton_iddet_0), len(skeleton_iddet_1))
        if len_0.value != len_1.value:
            print('ERROR')
            exit(1)
        if len_0.value == len_1.value:
            num_id.value = len_0.value
        #########################" Compare the confidence and take the higher one as person_position"#######################
        if index == 0:
            for i in range(num_id.value):
                if person_cam_0_conf[str(i)] >= person_cam_1_conf[str(i)]:

                    # print('flag 0 skeleton_det_0')
                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                    person_position[str(i)] = skeleton_2dkpt_0[my_index_0]
                    # print('conf0 >= conf1', person_position)
                else:
                    # print('flag1 skeleton_det_1')
                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                    person_position[str(i)] = cam1_to_cam0(image_width, skeleton_2dkpt_1[my_index_1])
                    # print('conf0 < conf1', person_position)
            finished_proc[index] = True

        if index == 1:
            finished_proc[index] = True
        # print('id0 and id1', id_0, id_1)
        # time.sleep(10)

        while not all(finished_proc):
            continue

        print('Initialization finished............;')
        # time.sleep(10)
        while viewer.is_available() and all(open_zed) and meta_data['yes_no'] == True and meta_data['stop'] == False:
            meta_data['use_back_cam'] = False
            # print('my ids', id_0, id_1)
            skeleton_output[:] = []

            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                key_temp[:] = []
                finished_proc[index] = False
                # while any(finished_proc):
                #     continue
                # print('I take a photo..........')
                zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                # Retrieve objects
                zed.retrieve_objects(bodies, obj_runtime_param)

                ##########" visualization #################"
                # if index == 0:
                viewer.update_view(image, bodies)
                # image_left_ocv = image.get_data()
                # cv_viewer.render_2D(image_left_ocv, image_scale, bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
                # cv2.imshow("ZED | 2D View", image_left_ocv)
                # cv2.waitKey(10)
                ###############################################"
                if index == 0:
                    pos_x_0_temp[:] = []
                    id_0_temp[:] = []
                    skeleton_2dkpt_0[:] = []
                    skeleton_iddet_0[:] = []
                    skeleton_det_0 = bodies.object_list
                    len_0.value = len(bodies.object_list)
                    for obj in skeleton_det_0:
                        skeleton_2dkpt_0.append(obj.keypoint_2d)
                        skeleton_iddet_0.append(obj.id)
                    finished_proc[index] = True

                if index == 1:
                    pos_x_1_temp[:] = []
                    id_1_temp[:] = []
                    skeleton_2dkpt_1[:] = []
                    skeleton_iddet_1[:] = []
                    skeleton_det_1 = bodies.object_list
                    len_1.value = len(bodies.object_list)
                    for obj in skeleton_det_1:
                        skeleton_2dkpt_1.append(obj.keypoint_2d)
                        skeleton_iddet_1.append(obj.id)
                    finished_proc[index] = True

                while not all(finished_proc):
                    # print('1')
                    continue
                finished_proc[index] = False
                # print('all kid processes are finished')
                #### take cam0 as the main cam, compare the result########
                if (len_0.value == 0) and (len_1.value == 0):
                    print('case all zero')
                    pass
                else:
                    # if (len_0.value < num_id.value) and (len_1.value < num_id.value):
                    if (len_0.value == 1) and (len_1.value == 1):
                        # print('case 0')
                        if index == 0:  ### cam 0 lost
                            for i in range(num_id.value):
                                absent = find_person_not_in(skeleton_det_0, person_cam0[str(i)])
                                if not absent:
                                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                    person_cam_0_conf[str(i)] = skeleton_det_0[my_index_0].confidence
                                if absent:
                                    person_cam0[str(i)] = None
                                    person_cam_0_conf[str(i)] = -999
                            finished_proc[index] = True
                        if index == 1:  ### cam 0 lost
                            for i in range(num_id.value):
                                absent = find_person_not_in(skeleton_det_1, person_cam1[str(i)])
                                if not absent:
                                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                    person_cam_1_conf[str(i)] = skeleton_det_1[my_index_1].confidence
                                if absent:
                                    person_cam1[str(i)] = None
                                    person_cam_1_conf[str(i)] = -999
                            finished_proc[index] = True
                        while not all(finished_proc):
                            # print('2')
                            continue
                        finished_proc[index] = False
                        ################# Then compare the confidence and take the better one ##################
                        if index == 0:
                            for i in range(num_id.value):
                                # print('index 0 arrived....')
                                # print('conf 0 and 1 ', person_cam_0_conf, person_cam_1_conf)
                                if person_cam_0_conf[str(i)] >= person_cam_1_conf[str(i)]:
                                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                    person_position[str(i)] = skeleton_2dkpt_0[my_index_0]

                                else:
                                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                    person_position[str(i)] = cam1_to_cam0(image_width, skeleton_2dkpt_1[my_index_1])
                            finished_proc[index] = True
                        if index == 1:
                            finished_proc[index] = True

                        while not all(finished_proc):
                            # print('3')
                            continue
                        # print('case 0 done')

                    ############################## case 1: 2 cams detects all ############################################################"
                    if (len_0.value == num_id.value) and (len_1.value == num_id.value):  ########### There is no id drop
                        # print('case 1')
                        # print('person cam 0', person_cam0)
                        # print('person cam 1', person_cam1)
                        # print('person cam 1', person_cam1)
                        cam_0_with_absent = if_cam_0_absent_detected(person_cam0)
                        cam_1_with_absent = if_cam_1_absent_detected(person_cam1)
                        ####################" if there is no person absent ###############################"
                        if (cam_0_with_absent == True) and (
                                cam_1_with_absent == True):  ##if last time each cam lost person
                            if index == 0:
                                for i in range(num_id.value):
                                    absent = find_person_not_in(skeleton_det_0, person_cam0[str(i)])
                                    if not absent:
                                        my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                        person_cam_0_conf[str(i)] = skeleton_det_0[my_index_0].confidence
                                    if absent:  #### if only lost 1 person
                                        person_cam0[str(i)] = skeleton_det_0[-1].id
                                        person_cam_0_conf[str(i)] = skeleton_det_0[-1].confidence
                                finished_proc[index] = True
                            if index == 1:
                                for i in range(num_id.value):
                                    absent = find_person_not_in(skeleton_det_1, person_cam1[str(i)])
                                    if not absent:
                                        my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                        person_cam_1_conf[str(i)] = skeleton_det_1[my_index_1].confidence
                                    if absent:  #### if only lost 1 person
                                        person_cam1[str(i)] = skeleton_det_1[-1].id
                                        person_cam_1_conf[str(i)] = skeleton_det_1[-1].confidence
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('4')
                                continue
                            finished_proc[index] = False
                            ################# Then compare the confidence and take the better one ##################
                            if index == 0:
                                for i in range(num_id.value):
                                    if person_cam_0_conf[str(i)] >= person_cam_1_conf[str(i)]:
                                        my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                        person_position[str(i)] = skeleton_2dkpt_0[my_index_0]
                                    else:
                                        my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                        person_position[str(i)] = cam1_to_cam0(image_width,
                                                                               skeleton_2dkpt_1[my_index_1])
                                finished_proc[index] = True
                            if index == 1:
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('5')
                                continue

                        if (cam_0_with_absent == False) and (cam_1_with_absent == False):
                            # print('if there is no person absent')
                            ############## update confidence ############"
                            if index == 0:
                                for i in range(num_id.value):
                                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                    person_cam_0_conf[str(i)] = skeleton_det_0[my_index_0].confidence
                                finished_proc[index] = True
                            if index == 1:
                                for i in range(num_id.value):
                                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                    person_cam_1_conf[str(i)] = skeleton_det_1[my_index_1].confidence
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('6')
                                continue
                            finished_proc[index] = False
                            ################ choose the better one as person_position ####################
                            if index == 0:
                                # print('conf 0 and 1 ', person_cam_0_conf, person_cam_1_conf)
                                for i in range(num_id.value):
                                    if person_cam_0_conf[str(i)] >= person_cam_1_conf[str(i)]:
                                        # print('conf0>conf1')
                                        my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                        person_position[str(i)] = skeleton_2dkpt_0[my_index_0]
                                        # print(person_position)
                                    else:
                                        # print('conf0<conf1')
                                        my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                        person_position[str(i)] = skeleton_2dkpt_1[my_index_1]
                                        # print('before', person_position)
                                        person_position[str(i)] = cam1_to_cam0(image_width,
                                                                               skeleton_2dkpt_1[my_index_1])
                                        # print('after',person_position)
                                finished_proc[index] = True
                            if index == 1:
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('7')
                                continue
                            # while not all(finished_proc) == True:
                            #     continue

                        ################## if cam0 has lost 1 person ###########################
                        if (cam_0_with_absent == True) and (cam_1_with_absent == False):  #### if last time cam0 has an absent
                            # print('if last time cam0 has an absent')
                            ################### find the absent one and give it a new id+confidence ###################
                            """if index == 0:
                                for i in range(num_id.value):
                                    absent = find_person_not_in(skeleton_det_0, person_cam0[str(i)])
                                    if not absent:
                                        my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                        person_cam_0_conf[str(i)] = skeleton_det_0[my_index_0].confidence
                                    if absent: #### if only lost 1 person
                                        person_cam0[str(i)] = skeleton_det_0[-1].id
                                        person_cam_0_conf[str(i)] = skeleton_det_0[-1].confidence
                                finished_proc[index] = True

                            else:
                                finished_proc[index] = True
                                pass"""
                            ######## This part can solve the pb of losing 2+ people ########
                            if index == 1:  ### last time cam 1 has no lost
                                for i in range(num_id.value):
                                    pos_x_1_temp.append(skeleton_det_1[i].head_position[0])
                                pos_x_1_temp.sort(reverse=True)  # head from left to right
                                for i in range(num_id.value):
                                    for skeleton in skeleton_det_1:
                                        if pos_x_1_temp[i] == skeleton.head_position[0]:
                                            id_1_temp.append(skeleton.id)
                                keys = person_cam1.keys()
                                # print('key test', keys)
                                # print('id 1 test', id_1_temp)

                                for i in range(num_id.value):
                                    for k in keys:
                                        if person_cam1[k] == id_1_temp[i]:
                                            key_temp.append(k)  #### get a list contains person id from left to right
                                # print('mye key_temp from index 1', key_temp)
                                finished_proc[index] = True
                            if index == 0:
                                finished_proc[index] = True

                            while not all(finished_proc):
                                # print('8')
                                continue
                            finished_proc[index] = False
                            ########## find the lack person's new id ######################
                            if index == 0:  ## cam 0 is the one lost person
                                for i in range(num_id.value):
                                    pos_x_0_temp.append(skeleton_det_0[i].head_position[0])
                                pos_x_0_temp.sort()
                                for i in range(num_id.value):
                                    for skeleton in skeleton_det_0:
                                        if pos_x_0_temp[i] == skeleton.head_position[0]:
                                            id_0_temp.append(skeleton.id)
                                # print('mye key_temp from index 0', key_temp)
                                # print('index 0 id 0 temp', id_0_temp)
                                # print('index 0 person before', person_cam0)
                                for i, k in enumerate(key_temp):
                                    person_cam0[k] = id_0_temp[i]  ##### fill up the person_cam dict with the new id
                                # print('index 0 person after', person_cam0)
                                finished_proc[index] = True
                            if index == 1:
                                finished_proc[index] = True

                            while not all(finished_proc):
                                # print('9')
                                continue
                            finished_proc[index] = False
                            ############ update the conf ###################
                            if index == 0:
                                for i in range(num_id.value):
                                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                    person_cam_0_conf[str(i)] = skeleton_det_0[my_index_0].confidence
                                finished_proc[index] = True
                            if index == 1:
                                for i in range(num_id.value):
                                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                    person_cam_1_conf[str(i)] = skeleton_det_1[my_index_1].confidence
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('10')
                                continue
                            finished_proc[index] = False

                            ################# Then compare the confidence and take the better one ##################
                            if index == 0:
                                for i in range(num_id.value):
                                    if person_cam_0_conf[str(i)] >= person_cam_1_conf[str(i)]:
                                        my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                        person_position[str(i)] = skeleton_2dkpt_0[my_index_0]
                                    else:
                                        my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                        person_position[str(i)] = cam1_to_cam0(image_width,
                                                                               skeleton_2dkpt_1[my_index_1])
                                finished_proc[index] = True
                            if index == 1:
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('11')
                                continue

                        ################## if cam1 has lost 1 person ###########################
                        if (cam_1_with_absent == True) and (cam_0_with_absent == False):  #### if last time cam0 has an absent
                            # print('if last time cam1 has an absent')
                            ################### find the absent one and give it a new id+confidence ###################
                            """if index == 1:
                                for i in range(num_id.value):
                                    # print('id dect',skeleton_det_1[0].id )
                                    absent = find_person_not_in(skeleton_det_1, person_cam1[str(i)])
                                    # print('person cam 1 before',person_cam1 )
                                    if not absent:
                                        my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                        person_cam_1_conf[str(i)] = skeleton_det_1[my_index_1].confidence
                                    if absent:
                                        person_cam1[str(i)] = skeleton_det_1[-1].id
                                        # print('person cam 1 after', person_cam1)
                                        person_cam_1_conf[str(i)] = skeleton_det_1[-1].confidence
                                finished_proc[index] = True
                                else:
                                    finished_proc[index] = True
                                    pass
                                """
                            ######## This part can solve the pb of losing 2+ people ########
                            if index == 0:  ### last time cam 0 has no lost
                                for i in range(num_id.value):
                                    pos_x_0_temp.append(skeleton_det_0[i].head_position[0])
                                pos_x_0_temp.sort()  # head from left to right
                                for i in range(num_id.value):
                                    for skeleton in skeleton_det_0:
                                        if pos_x_0_temp[i] == skeleton.head_position[0]:
                                            id_0_temp.append(skeleton.id)
                                keys = person_cam0.keys()
                                # key_temp[:] = []
                                for i in range(num_id.value):
                                    for k in keys:
                                        if person_cam0[k] == id_0_temp[i]:
                                            key_temp.append(k)  #### get a list contains person id from left to right
                                # print('my key_temp from index 0', key_temp)
                                finished_proc[index] = True
                            if index == 1:
                                finished_proc[index] = True

                            while not all(finished_proc):
                                # print('12')
                                continue
                            finished_proc[index] = False
                            ########## find the lack person's new id ######################
                            if index == 1:  ## cam 1 is the one lost person
                                for i in range(num_id.value):
                                    pos_x_1_temp.append(skeleton_det_1[i].head_position[0])
                                pos_x_1_temp.sort(reverse=True)
                                for i in range(num_id.value):
                                    for skeleton in skeleton_det_1:
                                        if pos_x_1_temp[i] == skeleton.head_position[0]:
                                            id_1_temp.append(skeleton.id)
                                # print('my key_temp from index 1', key_temp)
                                # print('index 1 id_1_temp', id_1_temp)
                                # print('index 1 person before', person_cam1)
                                for i, k in enumerate(key_temp):
                                    person_cam1[k] = id_1_temp[i]  ##### fill up the person_cam dict with the new id
                                # print('index 1 person after', person_cam1)
                                finished_proc[index] = True
                            if index == 0:
                                finished_proc[index] = True

                            while not all(finished_proc):
                                # print('13')
                                continue
                            finished_proc[index] = False
                            ############ update the conf ###################
                            if index == 0:
                                for i in range(num_id.value):
                                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                    person_cam_0_conf[str(i)] = skeleton_det_0[my_index_0].confidence
                                finished_proc[index] = True
                            if index == 1:
                                for i in range(num_id.value):
                                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                    person_cam_1_conf[str(i)] = skeleton_det_1[my_index_1].confidence
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('14')
                                continue
                            finished_proc[index] = False

                            ################# Then compare the confidence and take the better one ##################
                            if index == 0:
                                for i in range(num_id.value):
                                    if person_cam_0_conf[str(i)] >= person_cam_1_conf[str(i)]:
                                        my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                        person_position[str(i)] = skeleton_2dkpt_0[my_index_0]
                                    else:
                                        my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                        person_position[str(i)] = cam1_to_cam0(image_width,
                                                                               skeleton_2dkpt_1[my_index_1])
                                finished_proc[index] = True
                            if index == 1:
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('15')
                                continue

                        finished_proc[index] = False
                        # print('case 1 done')

                    ############################## case 2: only cam 0 detects all #######################################################################"
                    if (len_0.value == num_id.value) and (len_1.value < num_id.value):  ############# There is an id drop at cam1
                        # print('case 2')
                        cam_0_with_absent_temp = if_cam_0_absent_detected(person_cam0)
                        cam_1_with_absent_temp = if_cam_1_absent_detected(person_cam1)
                        ########### if case 3 is after case 0 ###########################
                        if (cam_0_with_absent_temp == True) and (
                                cam_1_with_absent_temp == True):  ##if last time each cam lost person
                            ############ cam 0 detects both #############
                            ############ update person_cam0 by adding the newest id ############
                            if index == 0:
                                for i in range(num_id.value):
                                    absent = find_person_not_in(skeleton_det_0, person_cam0[str(i)])
                                    if absent:  #### if only lost 1 person
                                        person_cam0[str(i)] = skeleton_det_0[-1].id
                                finished_proc[index] = True
                            if index == 1:
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('16')
                                continue
                            finished_proc[index] = False
                        ################### update the conf for cam0 ############################""
                        if index == 0:
                            # print('cam {} detects all participants'.format(0))
                            # person_cam0 = person_cam0
                            for i in range(num_id.value):
                                my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                person_cam_0_conf[str(i)] = skeleton_det_0[my_index_0].confidence
                            finished_proc[index] = True
                        ################"## update the conf for cam1, the absent one has a conf -999 ########################
                        if index == 1:
                            for i in range(num_id.value):
                                absent = find_person_not_in(skeleton_det_1, person_cam1[str(i)])
                                if not absent:
                                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                    person_cam_1_conf[str(i)] = skeleton_det_1[my_index_1].confidence
                                if absent:
                                    person_cam1[str(i)] = None
                                    person_cam_1_conf[str(i)] = -999
                            finished_proc[index] = True
                        while not all(finished_proc):
                            # print('17')
                            continue
                        finished_proc[index] = False
                        ################# Then compare the confidence and take the better one ##################
                        if index == 0:
                            # print('conf 0 and 1 ', person_cam_0_conf, person_cam_1_conf)
                            for i in range(num_id.value):
                                if person_cam_0_conf[str(i)] >= person_cam_1_conf[str(i)]:
                                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                    person_position[str(i)] = skeleton_2dkpt_0[my_index_0]
                                else:
                                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                    person_position[str(i)] = cam1_to_cam0(image_width, skeleton_2dkpt_1[my_index_1])
                            finished_proc[index] = True
                        if index == 1:
                            finished_proc[index] = True
                        while not all(finished_proc):
                            # print('18')
                            continue
                        # print('case 2 done')
                        finished_proc[index] = False

                    ############################## case 3: only cam 1 detects all #######################################################################"
                    if (len_0.value < num_id.value) and (len_1.value == num_id.value):  ############# There is qn id drop qt cam0
                        # print('case 3')
                        cam_0_with_absent_temp = if_cam_0_absent_detected(person_cam0)
                        cam_1_with_absent_temp = if_cam_1_absent_detected(person_cam1)
                        ########### if case 3 is after case 0 ###########################
                        if (cam_0_with_absent_temp == True) and (
                                cam_1_with_absent_temp == True):  ##if last time each cam lost person
                            ############ cam 1 detects both #############
                            ############ update person_cam1 by adding the newest id ############
                            # print("update person cam 1.......")
                            if index == 1:
                                for i in range(num_id.value):
                                    absent = find_person_not_in(skeleton_det_1, person_cam1[str(i)])
                                    if absent:  #### if only lost 1 person
                                        person_cam1[str(i)] = skeleton_det_1[-1].id
                                finished_proc[index] = True
                            if index == 0:
                                finished_proc[index] = True
                            while not all(finished_proc):
                                # print('19')
                                continue
                            finished_proc[index] = False
                        ################### update the conf for cam1 ############################""
                        if index == 1:
                            meta_data['use_back_cam'] = True
                            # print('cam {} detects all participants'.format(1))
                            for i in range(num_id.value):
                                my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                person_cam_1_conf[str(i)] = skeleton_det_1[my_index_1].confidence
                            finished_proc[index] = True
                        ################"## update the conf for cam0, the absent one has a conf -999 ########################
                        if index == 0:
                            for i in range(num_id.value):
                                absent = find_person_not_in(skeleton_det_0, person_cam0[str(i)])
                                if not absent:
                                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                    person_cam_0_conf[str(i)] = skeleton_det_0[my_index_0].confidence
                                if absent:
                                    person_cam0[str(i)] = None
                                    person_cam_0_conf[str(i)] = -999
                            finished_proc[index] = True
                        while not all(finished_proc):
                            # print('20')
                            continue
                        finished_proc[index] = False
                        ################# Then compare the confidence and take the better one ##################
                        if index == 0:
                            for i in range(num_id.value):
                                # print('conf 0 and 1 ', person_cam_0_conf, person_cam_1_conf)
                                if person_cam_0_conf[str(i)] >= person_cam_1_conf[str(i)]:
                                    my_index_0 = find_index_by_id(skeleton_iddet_0, person_cam0[str(i)])
                                    person_position[str(i)] = skeleton_2dkpt_0[my_index_0]
                                else:
                                    my_index_1 = find_index_by_id(skeleton_iddet_1, person_cam1[str(i)])
                                    person_position[str(i)] = cam1_to_cam0(image_width, skeleton_2dkpt_1[my_index_1])
                            finished_proc[index] = True
                        if index == 1:
                            finished_proc[index] = True
                        while not all(finished_proc):
                            # print('21')
                            continue
                        # print('case 3 done')
                        finished_proc[index] = False
                # while not all(finished_proc) :
                #     print('22')
                #     continue

                # client.send_message("N_Agent", num_id.value)
                if index == 0:
                    # print('send msg')
                    keys = person_position.keys()
                    client.send_message("N_Agent", num_id.value)
                    for i in range(num_id.value):
                        # print('N_Agent : ',num_id.value)
                        # print("agent id :" + str(keys[i]))
                        # print('person position',person_position)
                        for j in range(18):
                            data = np.concatenate(
                                [person_position[str(i)][j], np.array([0]), np.array([person_cam_0_conf[str(i)]])],
                                axis=-1)
                            # print(data.shape)
                            client.send_message("Agent_" + str(keys[i]) + "_Joint_" + str(j) + "_View_0", data.tolist())
                    finished_proc[index] = True
                if index == 1:
                    finished_proc[index] = True

                while not all(finished_proc):
                    # print('23')
                    continue
                # print('msg sent')
            key_temp[:] = []
        finished_proc[index] = False
        # while not all(finished_proc) :
        #     print('24')
        #     continue
    # if index == 0:
    viewer.exit()
    image.free(sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()
    return 0


def main():
    dispatcher.map("/filter*", set_filter)  # Map wildcard address to set_filter function

    # Set up server and client for testing
    from pythonosc.osc_server import BlockingOSCUDPServer
    from pythonosc.udp_client import SimpleUDPClient

    # server = BlockingOSCUDPServer(("172.25.32.95", 12000), dispatcher)  # 172.25.32.95
    # client = SimpleUDPClient("172.25.32.5", 12000)  # 172.25.32.5

    server = BlockingOSCUDPServer(("169.254.108.82", 12000), dispatcher)  # 172.25.32.95
    client = SimpleUDPClient("169.254.108.81", 12000)  # 172.25.32.5 #169.254.108.81

    msg = 0
    # Send message and receive exactly one message (blocking)
    client.send_message("/filter1", [1., 2.])  #
    # server.handle_request()

    client.send_message("/filter8", [6., -2.])
    # server.handle_request()

    cameras = sl.Camera.get_device_list()  # return a list
    serials_zed = [i.serial_number for i in cameras]  # return the list of serials number
    serials_zed.sort(reverse=True)
    # serials_zed.sort()
    print(serials_zed)
    print("Serials zed={}".format(serials_zed))
    manager = multiprocessing.Manager()
    len_0 = manager.Value('i', 0, lock=False)
    len_1 = manager.Value('i', 0, lock=False)
    num_id = manager.Value('i', 0, lock=False)
    l = [False for i in range(len(serials_zed))]
    open_zed = manager.list((l))  # list proxy of the state of 4 cameras:0/1
    finished_proc = manager.list((l))
    m = {"yes_no": False, "use_back_cam": False, "stop": False}
    meta_data = manager.dict(m)
    # p_cam0 = {"a": None, "b":None, "c":None, "a_conf":None, "b_conf":None, "c_conf": None}
    # person_cam0 = manager.dict(p_cam0)
    # p_cam1 = {"a": None, "b": None, "c": None, "a_conf": None, "b_conf": None, "c_conf": None}
    # person_cam1 = manager.dict(p_cam1)
    person_cam0 = manager.dict()
    person_cam1 = manager.dict()
    person_cam_0_conf = manager.dict()
    person_cam_1_conf = manager.dict()
    # p_position = {"a_position":np.zeros((18,2)), "b_position":np.zeros((18,2)), "c_position":np.zeros((18,2))}
    person_position = manager.dict()

    id_output = manager.list()
    skeleton_output = manager.list()
    # skeleton_det_0 = manager.list()
    # skeleton_det_1 = manager.list()
    skeleton_2dkpt_0 = manager.list()
    skeleton_2dkpt_1 = manager.list()
    skeleton_iddet_0 = manager.list()
    skeleton_iddet_1 = manager.list()
    skeleton_conf_0 = manager.list()
    skeleton_conf_1 = manager.list()
    id_0 = manager.list()
    id_1 = manager.list()
    length = manager.list()
    pos_x_0_temp = manager.list()
    pos_x_1_temp = manager.list()
    id_0_temp = manager.list()
    id_1_temp = manager.list()
    key_temp = manager.list()

    for index in range(len(cameras)):
        process_list.append(
            Process(target=grab_run, args=(
            index, serials_zed, open_zed, finished_proc, meta_data, skeleton_output, len_0, len_1, num_id, person_cam0,
            person_cam1, person_position, person_cam_0_conf, person_cam_1_conf,
            skeleton_2dkpt_0, skeleton_2dkpt_1, skeleton_iddet_0, skeleton_iddet_1, client,
            pos_x_0_temp, pos_x_1_temp, id_0_temp, id_1_temp, key_temp,)))  # recording_flir, opened_flir,
        # print("The Process ID of Parent Process is :{}".format(multiprocessing.current_process().pid))
        process_list[index].start()
    while True:
        while not all(open_zed):
            continue
        if_start = 'n'
        while if_start != 'y' and if_start != 'Y':
            if_start = input("Ready to start? [y/n] :")
        print("if_start : " + str(if_start))
        if if_start != 'y':  # if not start, next example
            print("Process denied...")
        if if_start == 'y':  # if start
            print("Process wil begin in 3 seconds")
            time.sleep(5)
            # recording_zed[0] = True  #Recoding start
            # start recording_zed
            meta_data['yes_no'] = True
            meta_data['stop'] = False

        global stop_signal
        stop_signal = False
        L = keyboard.Listener(on_press=my_press)
        L.start()
        while not stop_signal:
            continue
        meta_data['stop'] = True
        meta_data['yes_no'] = False
        L.stop()

    return 0


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    __spec__ = None
    main()