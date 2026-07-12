# 芯宝桌面机器人外壳 v1 方案

这个目录是第一版“工程验证壳”的 3D 打印设计包。目标不是一次做成最终外观，而是先验证：能不能装下、能不能散热、能不能维修、屏幕/麦克风/喇叭位置是否合理。

## 当前文件

- `xinbao_enclosure_v0.scad`：参数化 OpenSCAD 模型源文件，目前内容已更新到 v1。
- `export_stl.bat`：安装 OpenSCAD 后，用来导出 STL 的 Windows 脚本。
- 后续建议导出的 STL：
  - `xinbao_front_shell_v1.stl`
  - `xinbao_back_shell_v1.stl`
  - `xinbao_base_v1.stl`

## v1 结构方案

外壳分成三件：

1. 前壳：屏幕窗口、屏幕固定柱、单 40mm 喇叭出声孔、顶部/额头麦克风开孔。
2. 后壳：背部风扇/散热孔、接口检修窗口、侧面散热槽。
3. 底座：Orange Pi AI Pro 官方盒子放置舱、底部通风孔、后侧走线出口。

默认外形尺寸约为：

```text
宽 175mm
深 125mm
高 190mm
壁厚 3mm
```

这个尺寸是按你给的 Orange Pi AI Pro 20T 官方盒子图纸尺寸、常见 5 寸 HDMI 屏、40mm 小喇叭、顶部 M260C 环形六麦预留出来的。第一版会稍微宽松，方便塞线和调试。

## 默认硬件布局

```text
正面上方：屏幕，作为芯宝表情脸
正面下方：40mm 单喇叭
顶部偏前：M260C 环形六麦
背面上方：40mm 风扇/出风孔
背面中下：Type-C / USB / HDMI 检修窗口
底座内部：Orange Pi AI Pro 官方盒子放置舱
底座后方：电源线/音频线走线出口
```

## 需要你实际测量后修改的尺寸

打开 `xinbao_enclosure_v0.scad`，优先改顶部参数区：

```text
screen_window_w / screen_window_h     屏幕可视窗口，默认适配常见 5 寸 HDMI 屏
screen_mount_w / screen_mount_h       屏幕模块螺丝孔距
mic_outer_d / mic_hole_r / mic_hole_d M260C 外径和拾音孔位置
pi_case_w / pi_case_d / pi_case_h     Orange Pi 官方盒子外形尺寸
speaker_d                             喇叭直径
```

如果暂时没量到，先不要大量打印。建议先打印局部小样：屏幕窗口小样、麦克风顶部小样、开发板底座固定柱小样。

## 导出 STL

安装 OpenSCAD 后，在这个目录运行：

```bat
export_stl.bat
```

或者手动执行：

```bat
openscad -o stl\xinbao_front_shell_v1.stl -D PART_ID=1 xinbao_enclosure_v0.scad
openscad -o stl\xinbao_back_shell_v1.stl  -D PART_ID=2 xinbao_enclosure_v0.scad
openscad -o stl\xinbao_base_v1.stl        -D PART_ID=3 xinbao_enclosure_v0.scad
```

## 打印建议

```text
材料：PLA/PETG 都可以，比赛反复调试建议 PETG 更耐热
喷嘴：0.4mm
层高：0.2mm
壁厚：至少 3 道墙
填充：15% - 25%
支撑：前壳屏幕窗口附近可能需要少量支撑，底座一般不需要
螺丝：M2.5/M3 自攻或铜螺母热熔，先按手头螺丝微调孔径
```

## 第一版评审重点

1. Orange Pi 官方盒子放进去后，Type-C、USB、HDMI 插头是否会顶住外壳。
2. 散热片和风扇总高度是否足够。
3. 屏幕排线和 HDMI 转接头是否能自然弯折。
4. M260C 是否被顶盖遮挡，唤醒距离是否下降。
5. 喇叭出声是否闷，是否需要改成双侧喇叭。
6. 前壳、后壳、底座拆装是否方便。

## 下一步

把下面尺寸补齐后，就可以进入 v1，可直接按实物改到更接近可打印版本：

```text
屏幕外尺寸：
屏幕可视区域尺寸：
屏幕孔距：
M260C 外径：
M260C 孔位：
散热片+风扇总高度：
风扇直径和孔距：
喇叭直径和厚度：
PAM8403 功放板尺寸：
Orange Pi 官方盒子外形尺寸：
最大插头外露长度：
```
