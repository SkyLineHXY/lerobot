import math
import time
from dataclasses import dataclass, field
from typing import Dict

from piper_sdk import C_PiperInterface_V2


def _to_float(x) -> float:
    # 如果是 Tensor，取 item；否则直接转 float
    if "torch" in str(type(x)) and hasattr(x, "item"):
        return float(x.item())
    return float(x)


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]
    # --- 复位/安全相关参数 ---
    reset_hz: float = 100.0            # 平滑复位时的下发频率
    reset_duration_s: float = 4.0      # 平滑复位的期望总时长
    max_joint_step_rad: float = 0.01   # 单步允许的最大关节角度增量（限速，保证 MIT 模式下不剧烈）
    open_gripper_on_init: bool = True  # 复位时是否张开夹爪
    gripper_open_range: float = 0.07   # 夹爪张开开度（米，范围 0~0.08）
    home_position: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


class PiperMotorsBus:
    """
        对Piper SDK的二次封装
    """
    def __init__(self,
                 config: PiperMotorsBusConfig):
        self.piper = C_PiperInterface_V2(config.can_name)
        self.piper.ConnectPort()
        self.motors = config.motors
        # 录制数据集时改成0
        self.init_joint_position = list(config.home_position) # [6 joints + 1 gripper]
        self.safe_disable_position = [0.0, 0.0, 0.0, 0.0, 0.52, 0.0, 0.0]
        self.pose_factor = 1000 # 单位 0.001mm
        self.joint_factor = 57324.840764 # 1000*180/3.14， rad -> 度（单位0.001度）
        # 复位/安全参数
        self.reset_hz = config.reset_hz
        self.reset_duration_s = config.reset_duration_s
        self.max_joint_step_rad = config.max_joint_step_rad
        self.open_gripper_on_init = config.open_gripper_on_init
        self.gripper_open_range = config.gripper_open_range

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> list[int]:
        return [idx for idx, _ in self.motors.values()]


    def connect(self, enable:bool) -> bool:
        '''
            使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
        '''
        enable_flag = False
        loop_flag = False
        # 设置超时时间（秒）
        timeout = 5
        # 记录进入循环前的时间
        start_time = time.time()
        while not (loop_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_list = []
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status)
            enable_list.append(self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status)
            if (enable):
                enable_flag = all(enable_list)
                self.piper.EnableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
            else:
                # move to safe disconnect position
                enable_flag = any(enable_list)
                self.piper.DisableArm(7)
                self.piper.GripperCtrl(0, 1000, 0x02, 0)
            print(f"使能状态: {enable_flag}")
            print("--------------------")
            if (enable_flag == enable):
                loop_flag = True
                enable_flag = True
            else:
                loop_flag = False
                enable_flag = False
            # 检查是否超过超时时间
            if elapsed_time > timeout:
                print("超时....")
                enable_flag = False
                loop_flag = True
                break
            time.sleep(0.5)
        resp = enable_flag
        print(f"Returning response: {resp}")
        return resp

    def set_calibration(self):
        return

    def revert_calibration(self):
        return

    def apply_calibration(self):
        """
            平滑移动到初始（home）位置。

            MIT 模式下若直接下发远端目标，机械臂会从未知位置猛地冲向原点，
            这里改为「读取当前位置 -> 小步长缓入缓出插值」的平滑复位，降低冲击。
            若 open_gripper_on_init 为 True，则在复位过程中同步把夹爪张开。
        """
        target = list(self.init_joint_position)
        if self.open_gripper_on_init:
            target[6] = self.gripper_open_range
        self.move_to_joint_smoothly(target)

    def _read_joint_list(self) -> list:
        """读取当前 6 关节 + 夹爪 的有序列表（与 write 顺序一致，单位：弧度 / 米）"""
        joint_state = self.piper.GetArmJointMsgs().joint_state
        gripper_state = self.piper.GetArmGripperMsgs().gripper_state
        return [
            joint_state.joint_1 / self.joint_factor,
            joint_state.joint_2 / self.joint_factor,
            joint_state.joint_3 / self.joint_factor,
            joint_state.joint_4 / self.joint_factor,
            joint_state.joint_5 / self.joint_factor,
            joint_state.joint_6 / self.joint_factor,
            gripper_state.grippers_angle / 1000000,
        ]

    def move_to_joint_smoothly(self, target_joint: list, duration_s=None, hz=None, max_joint_step_rad=None):
        """
            从当前关节位置平滑插值运动到目标位置。

            步数同时受「总时长」和「单步最大角度增量」两个约束（取较大者）：
            - 总时长保证正常情况下动作柔和；
            - 单步限速保证即使距离很远/时长设得很短，每一拍的运动量也被限定，避免剧烈冲击。
            采用 smoothstep 缓入缓出，进一步减小起停时的抖动。
        """
        hz = self.reset_hz if hz is None else hz
        duration_s = self.reset_duration_s if duration_s is None else duration_s
        max_step = self.max_joint_step_rad if max_joint_step_rad is None else max_joint_step_rad

        # 刚使能时关节消息可能尚未刷新，稍等再读当前真实位置
        time.sleep(0.1)
        start = self._read_joint_list()
        target = [_to_float(x) for x in target_joint]

        # 仅按 6 个关节的最大位移估算限速步数（夹爪不参与限速）
        max_delta = max((abs(t - s) for t, s in zip(target[:6], start[:6])), default=0.0)
        steps_by_time = max(int(round(duration_s * hz)), 1)
        steps_by_vel = max(int(math.ceil(max_delta / max_step)), 1) if max_step > 0 else 1
        steps = max(steps_by_time, steps_by_vel)

        dt = 1.0 / hz
        for i in range(1, steps + 1):
            alpha = i / steps
            s = alpha * alpha * (3 - 2 * alpha)  # smoothstep 缓入缓出
            interp = [st + (tg - st) * s for st, tg in zip(start, target)]
            self.write(interp)
            time.sleep(dt)

    def open_gripper(self, gripper_range=None):
        """张开夹爪（gripper_range 单位：米，范围 0~0.08）"""
        r = self.gripper_open_range if gripper_range is None else gripper_range
        self.piper.GripperCtrl(round(abs(r) * 1000 * 1000), 1000, 0x01, 0)

    def close_gripper(self):
        """闭合夹爪"""
        self.piper.GripperCtrl(0, 1000, 0x01, 0)


    def write(self, target_joint: list):

        """
            Joint control
            - target joint: in radians
                joint_1 (float): 关节1角度 -92000 ~ 92000 / 57324.840764
                joint_2 (float): 关节2角度 -2400 ~ 120000 / 57324.840764
                joint_3 (float): 关节3角度 3000 ~ -110000 / 57324.840764
                joint_4 (float): 关节4角度 -90000 ~ 90000 / 57324.840764
                joint_5 (float): 关节5角度 80000 ~ -80000 / 57324.840764
                joint_6 (float): 关节6角度 -90000 ~ 90000 / 57324.840764
                gripper_range: 夹爪角度 0~0.08
        """

        j0 = round(_to_float(target_joint[0]) * self.joint_factor)
        j1 = round(_to_float(target_joint[1]) * self.joint_factor)
        j2 = round(_to_float(target_joint[2]) * self.joint_factor)
        j3 = round(_to_float(target_joint[3]) * self.joint_factor)
        j4 = round(_to_float(target_joint[4]) * self.joint_factor)
        j5 = round(_to_float(target_joint[5]) * self.joint_factor)
        gripper_range = round(_to_float(target_joint[6]) * 1000 * 1000)

        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
        self.piper.JointCtrl(j0, j1, j2, j3, j4, j5)
        self.piper.GripperCtrl(abs(gripper_range), 1000, 0x01, 0)

    def read(self) -> Dict:
        """
            - 机械臂关节消息,单位0.001度
            - 机械臂夹爪消息
            
            Returns a dictionary with keys from self.motors configuration.
            Maps physical joint indices (1-7) to configured motor names.
        """
        joint_msg = self.piper.GetArmJointMsgs()
        joint_state = joint_msg.joint_state
        gripper_msg = self.piper.GetArmGripperMsgs()
        gripper_state = gripper_msg.gripper_state

        # Physical joint values from SDK (indices 1-7)
        physical_values = [
            joint_state.joint_1 / self.joint_factor,
            joint_state.joint_2 / self.joint_factor,
            joint_state.joint_3 / self.joint_factor,
            joint_state.joint_4 / self.joint_factor,
            joint_state.joint_5 / self.joint_factor,
            joint_state.joint_6 / self.joint_factor,
            gripper_state.grippers_angle / 1000000
        ]
        
        # Build result dict based on motors configuration
        result = {}
        for motor_name, (physical_idx, model) in self.motors.items():
            # physical_idx is 1-based, convert to 0-based for list access
            result[motor_name] = physical_values[physical_idx - 1]
        
        return result
    def safe_disconnect(self):
        """
            平滑移动到断电前的安全位置（同样避免 MIT 模式下的剧烈运动）
        """
        self.move_to_joint_smoothly(self.safe_disable_position)
