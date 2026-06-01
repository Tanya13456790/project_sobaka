import tkinter as tk
from tkinter import messagebox, scrolledtext
import time
import threading
import math
from datetime import datetime
import numpy as np

# --- Unitree SDK2 Python imports ---
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

# ==========================================
# 1.  ПРЕСЕТЫ УГЛОВ СУСТАВОВ (из официального примера stand_go2.py)
# ==========================================
# 12 суставов: [FR hip, FR thigh, FR calf, FL hip, FL thigh, FL calf,
#               RR hip, RR thigh, RR calf, RL hip, RL thigh, RL calf]

STAND_UP_POS = np.array([
    0.00571868,  0.608813, -1.21763,
   -0.00571868,  0.608813, -1.21763,
    0.00571868,  0.608813, -1.21763,
   -0.00571868,  0.608813, -1.21763
], dtype=float)

STAND_DOWN_POS = np.array([
    0.0473455,  1.22187, -2.44375,
   -0.0473455,  1.22187, -2.44375,
    0.0473455,  1.22187, -2.44375,
   -0.0473455,  1.22187, -2.44375
], dtype=float)

# ==========================================
# 2.  LOW-LEVEL КОНТРОЛЛЕР (работает с MuJoCo)
# ==========================================

class Go2LowLevelController:
    """
    Отправляет LowCmd в симулятор unitree_mujoco через DDS.
    В отличие от SportClient, этот интерфейс работает в симуляторе.
    """
    DT = 0.002          # 500 Гц — требование Unitree
    FREQ = 1.0 / DT

    def __init__(self):
        self.is_connected = False
        self.command_history = []
        self._lock = threading.Lock()
        self._running = False
        self._motion_thread = None
        self._stop_event = threading.Event()

        # DDS / LowCmd
        self.pub = None
        self.cmd = None
        self.crc = CRC()

    def connect(self):
        try:
            # Domain 1 + loopback "lo" — стандарт для симуляции
            ChannelFactoryInitialize(1, "lo")
            self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.pub.Init()

            # Формируем базовое сообщение LowCmd
            self.cmd = unitree_go_msg_dds__LowCmd_()
            self.cmd.head[0] = 0xFE
            self.cmd.head[1] = 0xEF
            self.cmd.level_flag = 0xFF
            self.cmd.gpio = 0

            # Инициализация 20 моторов (Go2 использует первые 12)
            for i in range(20):
                self.cmd.motor_cmd[i].mode = 0x01  # PMSM torque mode
                self.cmd.motor_cmd[i].q = 0.0
                self.cmd.motor_cmd[i].kp = 0.0
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = 0.0
                self.cmd.motor_cmd[i].tau = 0.0

            self.is_connected = True
            self._add_history("ПОДКЛЮЧЕНИЕ: LowCmd → rt/lowcmd (MuJoCo)")
            return True
        except Exception as e:
            self._add_history(f"ОШИБКА ПОДКЛЮЧЕНИЯ: {e}")
            return False

    def disconnect(self):
        with self._lock:
            self._stop_event.set()
            self._running = False
            self.is_connected = False
        if self._motion_thread and self._motion_thread.is_alive():
            self._motion_thread.join(timeout=1.0)
        self._add_history("ОТКЛЮЧЕНИЕ: Контроллер остановлен")

    # ---------- Внутренние утилиты ----------

    def _add_history(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.command_history.append(f"[{ts}] {msg}")
        if len(self.command_history) > 100:
            self.command_history.pop(0)

    def _send_cmd(self, positions, kp=50.0, kd=3.5, dq=0.0, tau=0.0):
        """Одна итерация отправки LowCmd для 12 суставов."""
        if not self.is_connected or self.cmd is None:
            return
        for i in range(12):
            self.cmd.motor_cmd[i].q = float(positions[i])
            self.cmd.motor_cmd[i].kp = float(kp)
            self.cmd.motor_cmd[i].kd = float(kd)
            self.cmd.motor_cmd[i].dq = float(dq)
            self.cmd.motor_cmd[i].tau = float(tau)
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def _sleep_cycle(self, t0):
        """Точное соблюдение периода 2 мс."""
        elapsed = time.perf_counter() - t0
        remaining = self.DT - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _stop_current_motion(self):
        """Прерывает текущий поток движения перед запуском нового."""
        self._stop_event.set()
        if self._motion_thread and self._motion_thread.is_alive():
            self._motion_thread.join(timeout=0.5)
        self._stop_event.clear()

    # ---------- Основные команды ----------

    def stand_up(self):
        """Плавно встает за ~3 с (интерполяция tanh)."""
        self._stop_current_motion()
        self._add_history("КОМАНДА: Подъём робота (LowCmd)")
        self._running = True
        self._motion_thread = threading.Thread(target=self._stand_up_task, daemon=True)
        self._motion_thread.start()

    def _stand_up_task(self):
        t = 0.0
        while t < 3.0 and not self._stop_event.is_set():
            t0 = time.perf_counter()
            t += self.DT
            phase = math.tanh(t / 1.2)
            pos = phase * STAND_UP_POS + (1.0 - phase) * STAND_DOWN_POS
            kp = phase * 50.0 + (1.0 - phase) * 20.0
            self._send_cmd(pos, kp=kp, kd=3.5)
            self._sleep_cycle(t0)
        self._running = False

    def stand_down(self):
        """Плавно ложится за ~3 с."""
        self._stop_current_motion()
        self._add_history("КОМАНДА: Укладка робота (LowCmd)")
        self._running = True
        self._motion_thread = threading.Thread(target=self._stand_down_task, daemon=True)
        self._motion_thread.start()

    def _stand_down_task(self):
        t = 0.0
        while t < 3.0 and not self._stop_event.is_set():
            t0 = time.perf_counter()
            t += self.DT
            phase = math.tanh(t / 1.2)
            pos = phase * STAND_DOWN_POS + (1.0 - phase) * STAND_UP_POS
            self._send_cmd(pos, kp=50.0, kd=3.5)
            self._sleep_cycle(t0)
        self._running = False

    def stop(self):
        """Emergency stop — переход в Damp (обнуление усилий)."""
        self._stop_current_motion()
        self._add_history("ЭКСТРЕННАЯ ОСТАНОВКА → DAMP")
        self._running = True
        self._motion_thread = threading.Thread(target=self._damp_task, daemon=True)
        self._motion_thread.start()

    def _damp_task(self):
        """1 секунда плавного снижения усилий, затем нулевой torque."""
        for _ in range(500):  # ~1 с
            if self._stop_event.is_set():
                break
            t0 = time.perf_counter()
            damp_pos = STAND_DOWN_POS.copy()
            self._send_cmd(damp_pos, kp=5.0, kd=2.0, tau=0.0)
            self._sleep_cycle(t0)
        self._running = False

    # ---------- Движение (упрощённый trot gait) ----------

    def start_gait(self, vx=0.0, vy=0.0, rot=0.0, duration=3.0):
        """
        Запускает простейший синусоидальный trot-gait.
        Параметры:
          vx  — скорость вперёд/назад (м/с), диапазон [-0.5, 0.5]
          vy  — скорость вбок, диапазон [-0.3, 0.3]
          rot — угловая скорость (рад/с), диапазон [-1.0, 1.0]
        """
        self._stop_current_motion()
        self._add_history(f"ДВИЖЕНИЕ: vx={vx:.2f} vy={vy:.2f} rot={rot:.2f}")
        self._running = True
        self._motion_thread = threading.Thread(
            target=self._gait_task, args=(vx, vy, rot, duration), daemon=True
        )
        self._motion_thread.start()

    def _gait_task(self, vx, vy, rot, duration):
        """
        Упрощённый trot: диагональные пары ног движутся в противофазе.
        База — поза stand_up, к ней добавляются синусоидальные смещения.
        """
        t = 0.0
        # Амплитуды (подобраны эмпирически для стабильности в MJCF Go2)
        amp_thigh = 0.25 * abs(vx) / 0.5      # размах бедра
        amp_knee  = 0.35 * abs(vx) / 0.5      # размах колена
        amp_abd   = 0.15 * abs(vy) / 0.3      # размах отведения
        amp_rot   = 0.20 * abs(rot) / 1.0     # размах поворота

        freq = 2.5  # Hz — частота шага

        while t < duration and not self._stop_event.is_set():
            t0 = time.perf_counter()
            t += self.DT

            phase = 2.0 * math.pi * freq * t
            # Диагональные пары: (FR+RL) и (FL+RR)
            s1 = math.sin(phase)          # первая пара
            s2 = math.sin(phase + math.pi)  # вторая пара (противофаза)

            # Направление движения учитывается знаком
            dir_vx = 1.0 if vx >= 0 else -1.0
            dir_vy = 1.0 if vy >= 0 else -1.0
            dir_rot = 1.0 if rot >= 0 else -1.0

            pos = STAND_UP_POS.copy()

            # --- FR (0,1,2) ---
            pos[0] += dir_vy * amp_abd * s1
            pos[1] += dir_vx * amp_thigh * s1
            pos[2] += dir_vx * amp_knee  * s1
            # --- FL (3,4,5) ---
            pos[3] += dir_vy * amp_abd * s2
            pos[4] += dir_vx * amp_thigh * s2
            pos[5] += dir_vx * amp_knee  * s2
            # --- RR (6,7,8) ---
            pos[6] += dir_vy * amp_abd * s2
            pos[7] += dir_vx * amp_thigh * s2
            pos[8] += dir_vx * amp_knee  * s2
            # --- RL (9,10,11) ---
            pos[9]  += dir_vy * amp_abd * s1
            pos[10] += dir_vx * amp_thigh * s1
            pos[11] += dir_vx * amp_knee  * s1

            # Поворот: добавляем смещение к hip-roll пары ног по-разному
            if abs(rot) > 0.01:
                pos[0] += dir_rot * amp_rot * s1   # FR hip
                pos[3] -= dir_rot * amp_rot * s2   # FL hip
                pos[6] += dir_rot * amp_rot * s2   # RR hip
                pos[9] -= dir_rot * amp_rot * s1   # RL hip

            self._send_cmd(pos, kp=55.0, kd=4.0)
            self._sleep_cycle(t0)

        # По окончании — вернуться в стойку
        if not self._stop_event.is_set():
            self._return_to_stand()
        self._running = False

    def _return_to_stand(self):
        """Короткая интерполяция обратно в stand_up."""
        start_pos = STAND_UP_POS.copy()  # упрощённо
        for i in range(150):  # ~0.3 с
            if self._stop_event.is_set():
                break
            t0 = time.perf_counter()
            self._send_cmd(start_pos, kp=50.0, kd=3.5)
            self._sleep_cycle(t0)

    def get_history(self):
        return self.command_history


# ==========================================
# 3.  GUI (адаптировано из твоего файла)
# ==========================================

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Управление роботом-собакой Unitree Go2 [MuJoCo LowCmd]")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.bg_color = '#1a1a1a'
        self.fg_color = '#e0e0e0'
        self.frame_bg = '#252525'
        self.btn_bg = '#2a2a2a'
        self.btn_hover = '#3a3a3a'
        self.stop_bg = '#4a2a2a'
        self.root.configure(bg=self.bg_color)

        self.robot = Go2LowLevelController()
        self.robot_mode = "СИМУЛЯЦИЯ (MuJoCo + LowCmd)"
        self.robot.connect()
        self.auto_stabilize = tk.BooleanVar(value=True)

        self._setup_ui()
        self._update_status()
        self._update_history_display()

    def _setup_ui(self):
        main_container = tk.Frame(self.root, bg=self.bg_color)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        top_frame = tk.Frame(main_container, bg=self.bg_color)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_frame,
                 text=f"Управление роботом-собакой Unitree Go2 [{self.robot_mode}]",
                 font=("Segoe UI", 14, "normal"),
                 bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT)

        status_frame = tk.Frame(top_frame, bg=self.frame_bg)
        status_frame.pack(side=tk.RIGHT)

        tk.Label(status_frame, text="Статус:", font=("Segoe UI", 10),
                 bg=self.frame_bg, fg=self.fg_color).pack(side=tk.LEFT, padx=8, pady=4)
        self.status_label = tk.Label(status_frame, text="ПОДКЛЮЧАЮСЬ...",
                                     fg="#ffaa00", font=("Segoe UI", 10),
                                     bg=self.frame_bg)
        self.status_label.pack(side=tk.LEFT, padx=(0, 8), pady=4)

        content_frame = tk.Frame(main_container, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(content_frame, bg=self.bg_color)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        right_panel = tk.Frame(content_frame, bg=self.bg_color, width=320)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)

        # --- Видеопоток (заглушка) ---
        video_frame = tk.LabelFrame(left_panel, text="Видеопоток",
                                  font=("Segoe UI", 10, "normal"),
                                  bg=self.bg_color, fg=self.fg_color, relief=tk.FLAT)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.video_label = tk.Label(video_frame,
                                    text="[ ВИДЕО НЕ ДОСТУПНО ]\n\nЗапусти unitree_mujoco.py\nв отдельном терминале",
                                    bg='#0d0d0d', fg='#606060', font=("Segoe UI", 10))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # --- Управление движением ---
        control_frame = tk.LabelFrame(left_panel, text="Управление",
                                    font=("Segoe UI", 10, "normal"),
                                    bg=self.bg_color, fg=self.fg_color, relief=tk.FLAT)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        buttons_grid = tk.Frame(control_frame, bg=self.bg_color)
        buttons_grid.pack(pady=10)

        buttons = [
            ("↺ Лево",   self._turn_left),
            ("▲ Вперёд", self._move_forward),
            ("↻ Право",  self._turn_right),
            ("◄ Влево",  self._move_left),
            ("СТОП",     self._stop_robot),
            ("Вправо ►", self._move_right),
            ("", None),
            ("▼ Назад",  self._move_back),
            ("", None),
        ]

        for i, (text, command) in enumerate(buttons):
            row, col = i // 3, i % 3
            if text:
                btn_bg = self.stop_bg if "СТОП" in text else self.btn_bg
                btn = tk.Button(buttons_grid, text=text, command=command,
                                width=12, height=1, font=("Segoe UI", 10),
                                bg=btn_bg, fg=self.fg_color,
                                activebackground=self.btn_hover,
                                activeforeground=self.fg_color,
                                relief=tk.FLAT, bd=0)
                btn.grid(row=row, column=col, padx=4, pady=4)

        # --- Специальные команды ---
        special_frame = tk.LabelFrame(left_panel, text="Специальные команды",
                                     font=("Segoe UI", 10, "normal"),
                                     bg=self.bg_color, fg=self.fg_color, relief=tk.FLAT)
        special_frame.pack(fill=tk.X)
        special_buttons_frame = tk.Frame(special_frame, bg=self.bg_color)
        special_buttons_frame.pack(pady=8)

        btn_standup = tk.Button(special_buttons_frame, text="▲ Поднять робота",
                                command=self._stand_up_robot, width=16, height=1,
                                font=("Segoe UI", 10), bg=self.btn_bg, fg=self.fg_color,
                                activebackground=self.btn_hover, relief=tk.FLAT, bd=0)
        btn_standup.pack(side=tk.LEFT, padx=8)

        btn_standdown = tk.Button(special_buttons_frame, text="▼ Положить робота",
                                  command=self._stand_down_robot, width=16, height=1,
                                  font=("Segoe UI", 10), bg=self.btn_bg, fg=self.fg_color,
                                  activebackground=self.btn_hover, relief=tk.FLAT, bd=0)
        btn_standdown.pack(side=tk.LEFT, padx=8)

        # --- Настройки ---
        settings_frame = tk.LabelFrame(right_panel, text="Настройки",
                                       font=("Segoe UI", 10, "normal"),
                                       bg=self.bg_color, fg=self.fg_color, relief=tk.FLAT)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        chk_auto = tk.Checkbutton(settings_frame, text="Автоматическая стабилизация",
                                  variable=self.auto_stabilize, font=("Segoe UI", 10),
                                  bg=self.bg_color, fg=self.fg_color,
                                  selectcolor=self.bg_color,
                                  activebackground=self.bg_color, relief=tk.FLAT)
        chk_auto.pack(anchor=tk.W, padx=10, pady=8)
        tk.Label(settings_frame,
                 text="→ после движения робот возвращается\n  в стойку (LowCmd stand_up)",
                 font=("Segoe UI", 8), fg="#707070", bg=self.bg_color,
                 justify=tk.LEFT).pack(anchor=tk.W, padx=(10, 0), pady=(0, 8))

        # --- Действия ---
        actions_frame = tk.LabelFrame(right_panel, text="Действия",
                                      font=("Segoe UI", 10, "normal"),
                                      bg=self.bg_color, fg=self.fg_color, relief=tk.FLAT)
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        btn_clear = tk.Button(actions_frame, text="Очистить историю",
                              command=self._clear_history, width=20, height=1,
                              bg=self.btn_bg, fg=self.fg_color,
                              activebackground=self.btn_hover, relief=tk.FLAT, bd=0)
        btn_clear.pack(pady=8)
        btn_exit = tk.Button(actions_frame, text="Выход",
                             command=self._on_closing, width=20, height=1,
                             bg=self.btn_bg, fg=self.fg_color,
                             activebackground=self.btn_hover, relief=tk.FLAT, bd=0)
        btn_exit.pack(pady=(0, 8))

        # --- История ---
        history_frame = tk.LabelFrame(right_panel, text="История команд",
                                      font=("Segoe UI", 10, "normal"),
                                      bg=self.bg_color, fg=self.fg_color, relief=tk.FLAT)
        history_frame.pack(fill=tk.BOTH, expand=True)
        self.history_text = scrolledtext.ScrolledText(
            history_frame, font=("Consolas", 9), wrap=tk.WORD,
            bg='#0d0d0d', fg='#b0b0b0', insertbackground='#b0b0b0',
            relief=tk.FLAT, bd=0
        )
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # --- Нижний статус ---
        bottom_frame = tk.Frame(main_container, bg=self.frame_bg)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        self.bottom_status = tk.Label(
            bottom_frame, text="Готов к работе", anchor=tk.W,
            bg=self.frame_bg, fg="#a0a0a0", font=("Segoe UI", 9),
            padx=8, pady=4
        )
        self.bottom_status.pack(fill=tk.X)

    def _update_status(self):
        self.status_label.config(
            text="ПОДКЛЮЧЕН" if self.robot.is_connected else "НЕ ПОДКЛЮЧЕН",
            fg="#7cb342" if self.robot.is_connected else "#e57373"
        )
        self.root.after(1000, self._update_status)

    def _update_history_display(self):
        self.history_text.delete(1.0, tk.END)
        for line in self.robot.get_history():
            self.history_text.insert(tk.END, line + "\n")
        self.history_text.see(tk.END)
        self.root.after(500, self._update_history_display)

    def _clear_history(self):
        self.robot.command_history.clear()
        self._update_bottom_status("История очищена", 2000)

    def _update_bottom_status(self, message, duration=2000):
        self.bottom_status.config(text=message)
        self.root.after(duration, lambda: self.bottom_status.config(text="Готов к работе"))

    # ---------- Обработчики кнопок ----------

    def _move_forward(self):
        self._execute_with_stabilization(self.robot.start_gait, 0.4, 0.0, 0.0, 2.0)

    def _move_back(self):
        self._execute_with_stabilization(self.robot.start_gait, -0.4, 0.0, 0.0, 2.0)

    def _move_left(self):
        self._execute_with_stabilization(self.robot.start_gait, 0.0, 0.25, 0.0, 2.0)

    def _move_right(self):
        self._execute_with_stabilization(self.robot.start_gait, 0.0, -0.25, 0.0, 2.0)

    def _turn_left(self):
        self._execute_with_stabilization(self.robot.start_gait, 0.0, 0.0, 0.8, 2.0)

    def _turn_right(self):
        self._execute_with_stabilization(self.robot.start_gait, 0.0, 0.0, -0.8, 2.0)

    def _stop_robot(self):
        self.robot.stop()
        self._update_bottom_status("Стоп (DAMP)", 1000)

    def _stand_up_robot(self):
        self.robot.stand_up()
        self._update_bottom_status("▲ Поднят", 1500)

    def _stand_down_robot(self):
        self.robot.stand_down()
        self._update_bottom_status("▼ Уложен", 1500)

    def _execute_with_stabilization(self, command_func, *args):
        """Запускает движение; если auto_stabilize=True — возвращает в стойку после."""
        command_func(*args)
        if self.auto_stabilize.get():
            # Через duration+0.1 с вернуть в стойку
            duration = args[-1] if args else 2.0
            self.root.after(int((duration + 0.2) * 1000), self.robot.stand_up)
            self.root.after(int((duration + 0.3) * 1000),
                            lambda: self._update_bottom_status("Выполнено", 1500))
        else:
            self._update_bottom_status("Выполнено", 1500)

    def _on_closing(self):
        if messagebox.askokcancel("Выход", "Вы уверены?"):
            self.robot.disconnect()
            self.root.destroy()

    def run(self):
        self.root.mainloop()


# ==========================================
# 4.  ТОЧКА ВХОДА
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Лабораторная работа: Управление Go2 через MuJoCo")
    print("Интерфейс: LowCmd (низкоуровневый)")
    print("Важно: перед запуском запусти симулятор:")
    print("  cd unitree_mujoco/simulate_python && python3 unitree_mujoco.py")
    print("=" * 60)
    app = MainWindow()
    app.run()