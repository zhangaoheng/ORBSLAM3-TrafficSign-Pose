# Looming 几何原理

## 场景描述

车辆在平直道路上朝路牌行驶。从远帧到近帧，车辆前进距离为 Δd。

## 去旋（Derotation）

相机从远帧到近帧不仅有平移，还有旋转。在计算 Looming 之前，需要先用旋转矩阵 R₁₂ 将远帧的中心点去旋，补偿旋转的影响：

```
P_far_derot = π( K · R₁₂ᵀ · K⁻¹ · P̃_far )
```

去旋之后，两个帧之间的图像差异仅由平移引起。

## FOE（Focus of Expansion）

FOE 是车辆前进方向在像平面上的投影：

```
FOE = ( fx · tx/tz + cx,  fy · ty/tz + cy )
```

## 径向距离

在**去旋后**的远帧像平面中，路牌中心到 FOE 的像素距离为 r_far。
在近帧像平面中，路牌中心到 FOE 的像素距离为 r_near。

```
r_near = ||P_near − FOE||
r_far  = ||P_far_derot − FOE||
```

## 关键关系：r_near > r_far

车辆靠近路牌时，路牌在像平面上从 FOE 点向外"膨胀"。所以：

- **远帧**：路牌成像小，距离 FOE **近** → r_far **小**
- **近帧**：路牌成像大，距离 FOE **远** → r_near **大**

即 r_near > r_far。

## 膨胀量 dr

```
dr = r_near − r_far  (> 0)
```

## 深度公式

```
Z = r_near · Δd / dr − Δd
```

其中 Δd 是两帧之间的位移大小。

变换过程（推导）：r_far / r_near = Z / (Z + Δd) → Z = r_far · Δd / dr = r_near · Δd / dr − Δd

## 总结

这个公式仅依赖两个可观测的像素距离（r_far、r_near）和一个已知的运动量 Δd，不需要路牌的任何物理尺寸信息。
