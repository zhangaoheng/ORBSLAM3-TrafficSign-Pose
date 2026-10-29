const pptxgen = require('/tmp/codex_pptx_tool/node_modules/pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'ORBSLAM3-TrafficSign-Pose';
pptx.subject = '毕业论文核心两章汇报';
pptx.title = '基于道路交通标志的跨天气视觉重定位方法';
pptx.company = 'Graduate Thesis Progress Report';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: 'Microsoft YaHei',
  bodyFontFace: 'Microsoft YaHei',
  lang: 'zh-CN'
};
pptx.defineSlideMaster({
  title: 'MASTER',
  background: { color: 'F7F9FC' },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 0.08, fill: { color: '14B8A6' }, line: { color: '14B8A6' } } },
    { text: { text: '基于道路交通标志的跨天气视觉重定位', options: { x: 0.55, y: 7.13, w: 7.4, h: 0.16, fontFace: 'Microsoft YaHei', fontSize: 7.5, color: '718096', margin: 0 } } },
    { text: { text: '毕业论文核心章节汇报', options: { x: 9.6, y: 7.13, w: 2.8, h: 0.16, fontFace: 'Microsoft YaHei', fontSize: 7.5, color: '718096', align: 'right', margin: 0 } } }
  ],
  slideNumber: { x: 12.55, y: 7.10, color: '718096', fontFace: 'Aptos', fontSize: 8 }
});

const C = {
  navy: '0B1F33', blue: '2367D1', cyan: '0EA5C6', teal: '14B8A6',
  green: '16A36A', amber: 'F59E0B', red: 'E45555', purple: '7557D3',
  text: '172B3A', sub: '4A6275', muted: '718096', pale: 'EAF0F6',
  white: 'FFFFFF', bg: 'F7F9FC', line: 'D9E2EC', dark2: '16324A'
};
const ST = pptx.ShapeType;
const root = '/home/ehuy/ORBSLAM3-TrafficSign-Pose';
const imgRoadA = root + '/manual_split/seg_manual_0/rgb/1782374463514750464.png';
const imgRoadB = root + '/manual_split/seg_manual_1/rgb/1782374541542287360.png';
const imgLsq = root + '/landmarkslam/implement/data/test_pairs/pair_01_45/lsq_results/lsq_result.png';
const imgTraj = root + '/landmarkslam/implement/main/runs/20260630_164608/trajectory_alignment.png';

function addTitle(slide, title, section, subtitle='') {
  slide.addText(title, { x: 0.62, y: 0.34, w: 9.8, h: 0.48, fontFace: 'Microsoft YaHei', fontSize: 24, bold: true, color: C.navy, margin: 0, breakLine: false, fit: 'shrink' });
  if (subtitle) slide.addText(subtitle, { x: 0.64, y: 0.86, w: 10.8, h: 0.28, fontFace: 'Microsoft YaHei', fontSize: 10.5, color: C.sub, margin: 0, fit: 'shrink' });
  slide.addText(section, { x: 11.05, y: 0.40, w: 1.58, h: 0.28, fontFace: 'Microsoft YaHei', fontSize: 10, bold: true, color: C.teal, align: 'right', margin: 0 });
  slide.addShape(ST.line, { x: 0.62, y: 1.18, w: 12.0, h: 0, line: { color: C.line, width: 1 } });
}

function addPill(slide, text, x, y, w, color=C.blue, fill='EAF2FF') {
  slide.addShape(ST.roundRect, { x, y, w, h: 0.34, rectRadius: 0.07, fill: { color: fill }, line: { color: fill } });
  slide.addText(text, { x: x+0.08, y: y+0.07, w: w-0.16, h: 0.16, fontFace: 'Microsoft YaHei', fontSize: 9, bold: true, color, align: 'center', margin: 0, fit: 'shrink' });
}

function addCard(slide, x, y, w, h, title, body, accent=C.blue, opts={}) {
  slide.addShape(ST.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: opts.fill || C.white, transparency: opts.transparency || 0 }, line: { color: opts.line || C.line, width: 1 } });
  slide.addShape(ST.roundRect, { x: x+0.18, y: y+0.18, w: 0.08, h: Math.min(0.62,h-0.36), rectRadius: 0.03, fill: { color: accent }, line: { color: accent } });
  slide.addText(title, { x: x+0.40, y: y+0.18, w: w-0.58, h: 0.28, fontFace: 'Microsoft YaHei', fontSize: opts.titleSize || 14.5, bold: true, color: opts.titleColor || C.navy, margin: 0, fit: 'shrink' });
  if (body) slide.addText(body, { x: x+0.40, y: y+0.56, w: w-0.60, h: h-0.72, fontFace: 'Microsoft YaHei', fontSize: opts.bodySize || 11.5, color: opts.bodyColor || C.sub, margin: 0, breakLine: false, valign: 'top', fit: 'shrink', paraSpaceAfterPt: 4, bullet: opts.bullet });
}

function addBulletList(slide, items, x, y, w, h, opts={}) {
  const runs = [];
  items.forEach((it, idx) => {
    runs.push({ text: it, options: { bullet: { indent: opts.indent || 16 }, breakLine: idx < items.length-1, hanging: 4 } });
  });
  slide.addText(runs, { x, y, w, h, fontFace: 'Microsoft YaHei', fontSize: opts.fontSize || 14, color: opts.color || C.text, margin: 0.04, breakLine: false, valign: 'top', fit: 'shrink', paraSpaceAfterPt: opts.space || 10, lineSpacingMultiple: 1.05 });
}

function addArrow(slide, x1, y1, x2, y2, color=C.teal, width=2.2) {
  slide.addShape(ST.line, { x: x1, y: y1, w: x2-x1, h: y2-y1, line: { color, width, beginArrowType: 'none', endArrowType: 'triangle' } });
}

function addStep(slide, n, title, detail, x, y, w, color) {
  slide.addShape(ST.roundRect, { x, y, w, h: 1.10, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  slide.addShape(ST.ellipse, { x: x+0.15, y: y+0.18, w: 0.48, h: 0.48, fill: { color }, line: { color } });
  slide.addText(String(n), { x: x+0.15, y: y+0.275, w: 0.48, h: 0.16, fontFace: 'Aptos', fontSize: 12, bold: true, color: C.white, align: 'center', margin: 0 });
  slide.addText(title, { x: x+0.76, y: y+0.17, w: w-0.90, h: 0.26, fontFace: 'Microsoft YaHei', fontSize: 13, bold: true, color: C.navy, margin: 0, fit: 'shrink' });
  slide.addText(detail, { x: x+0.76, y: y+0.52, w: w-0.90, h: 0.38, fontFace: 'Microsoft YaHei', fontSize: 9.5, color: C.sub, margin: 0, fit: 'shrink' });
}

function addSectionSlide(num, title, subtitle, color) {
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  slide.addShape(ST.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.navy }, line: { color: C.navy } });
  slide.addShape(ST.arc, { x: 9.6, y: -1.1, w: 5.2, h: 5.2, adjustPoint: 0.35, rotate: 20, fill: { color, transparency: 28 }, line: { color, transparency: 100 } });
  slide.addShape(ST.ellipse, { x: 10.7, y: 4.7, w: 1.55, h: 1.55, fill: { color, transparency: 55 }, line: { color, transparency: 100 } });
  slide.addText(num, { x: 0.72, y: 1.02, w: 1.1, h: 0.75, fontFace: 'Aptos Display', fontSize: 42, bold: true, color, margin: 0 });
  slide.addText(title, { x: 0.72, y: 2.00, w: 10.6, h: 0.78, fontFace: 'Microsoft YaHei', fontSize: 32, bold: true, color: C.white, margin: 0, fit: 'shrink' });
  slide.addText(subtitle, { x: 0.75, y: 3.05, w: 9.2, h: 0.68, fontFace: 'Microsoft YaHei', fontSize: 15, color: 'B9CAD8', margin: 0, fit: 'shrink' });
  slide.addShape(ST.line, { x: 0.75, y: 4.04, w: 2.0, h: 0, line: { color, width: 4 } });
  return slide;
}

// 1. Cover
{
  const s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addShape(ST.rect, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.navy }, line: { color: C.navy } });
  s.addShape(ST.arc, { x: 8.9, y: -1.8, w: 6.7, h: 6.7, rotate: 25, fill: { color: C.teal, transparency: 35 }, line: { color: C.teal, transparency: 100 } });
  s.addShape(ST.ellipse, { x: 10.0, y: 4.8, w: 2.4, h: 2.4, fill: { color: C.blue, transparency: 45 }, line: { color: C.blue, transparency: 100 } });
  addPill(s, '毕业论文核心两章汇报', 0.78, 0.78, 2.25, C.teal, '123D4B');
  s.addText('基于道路交通标志的\n跨天气视觉重定位方法', { x: 0.78, y: 1.55, w: 9.65, h: 1.63, fontFace: 'Microsoft YaHei', fontSize: 32, bold: true, color: C.white, margin: 0, breakLine: false, fit: 'shrink', lineSpacingMultiple: 0.92 });
  s.addText('以交通标志为跨序列几何桥梁，直接恢复两次经过之间的米制相对位移', { x: 0.82, y: 3.45, w: 9.1, h: 0.45, fontFace: 'Microsoft YaHei', fontSize: 15.5, color: 'B9CAD8', margin: 0, fit: 'shrink' });
  s.addShape(ST.line, { x: 0.82, y: 4.22, w: 2.05, h: 0, line: { color: C.teal, width: 4 } });
  s.addText('汇报人：________    指导教师：________', { x: 0.82, y: 5.15, w: 5.3, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 12, color: 'D8E4EC', margin: 0 });
  s.addText('2026 年 10 月', { x: 0.82, y: 5.62, w: 3.0, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 11, color: '8FA6B8', margin: 0 });
}

// 2. Agenda
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '汇报结构', '总览', '围绕“问题—方法—证据—结论”展开两个核心章节');
  const items = [
    ['01', '问题描述', '为什么传统离线视觉地图在跨天气条件下失效？', C.red],
    ['02', '解决方案', '交通标志跨序列关联 + 米制相对位移 + GPS 推算', C.blue],
    ['03', '实验结果', '核心算法链路、单次实验结果与评价设计', C.amber],
    ['04', '结论', '创新点、阶段结论与汇报前优先工作', C.green]
  ];
  items.forEach((it,i) => {
    const y = 1.55 + i*1.25;
    s.addShape(ST.roundRect, { x: 0.85, y, w: 11.65, h: 0.92, rectRadius: 0.05, fill: { color: C.white }, line: { color: C.line } });
    s.addText(it[0], { x: 1.10, y: y+0.22, w: 0.65, h: 0.28, fontFace: 'Aptos Display', fontSize: 18, bold: true, color: it[3], margin: 0 });
    s.addText(it[1], { x: 2.02, y: y+0.17, w: 1.5, h: 0.32, fontFace: 'Microsoft YaHei', fontSize: 16, bold: true, color: C.navy, margin: 0 });
    s.addText(it[2], { x: 3.75, y: y+0.20, w: 7.9, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 12, color: C.sub, margin: 0, fit: 'shrink' });
  });
  s.addText('核心判断', { x: 0.92, y: 6.62, w: 1.0, h: 0.25, fontFace: 'Microsoft YaHei', fontSize: 10, bold: true, color: C.teal, margin: 0 });
  s.addText('传统方法依赖“看起来相似”；本文进一步利用“同一块固定交通标志的世界位置不变”。', { x: 2.0, y: 6.56, w: 9.9, h: 0.32, fontFace: 'Microsoft YaHei', fontSize: 13, bold: true, color: C.navy, margin: 0, fit: 'shrink' });
}

addSectionSlide('01', '问题描述', '跨天气外观变化，使传统离线视觉地图重定位失去稳定对应关系', C.red);

// 4. Conventional
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '现有方案：基于离线视觉地图的重定位', '问题描述');
  addStep(s, 1, '第一次经过', '采集图像并构建离线视觉地图', 0.72, 1.55, 3.15, C.blue);
  addArrow(s, 3.93, 2.10, 4.65, 2.10, C.blue);
  addStep(s, 2, '第二次经过', '当前图像与地图特征/外观匹配', 4.72, 1.55, 3.15, C.blue);
  addArrow(s, 7.93, 2.10, 8.65, 2.10, C.blue);
  addStep(s, 3, '恢复位置', '求解当前车辆在历史地图中的位姿', 8.72, 1.55, 3.85, C.green);
  s.addShape(ST.roundRect, { x: 0.92, y: 3.25, w: 11.45, h: 2.52, rectRadius: 0.08, fill: { color: 'EEF4FA' }, line: { color: 'D8E4EC' } });
  s.addText('隐含前提', { x: 1.25, y: 3.60, w: 1.55, h: 0.38, fontFace: 'Microsoft YaHei', fontSize: 20, bold: true, color: C.blue, margin: 0 });
  s.addText('第二次看到的道路场景，必须与离线地图中的图像“足够相似”', { x: 2.95, y: 3.60, w: 8.8, h: 0.40, fontFace: 'Microsoft YaHei', fontSize: 18, bold: true, color: C.navy, margin: 0, fit: 'shrink' });
  addBulletList(s, ['局部特征需要可重复检测', '全局外观描述需要保持相似', '跨序列必须建立足够多的正确对应'], 1.28, 4.42, 9.9, 0.95, { fontSize: 13, space: 6 });
}

// 5. Cross-weather failure
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '天气变化打破了“外观一致”假设', '问题描述');
  addCard(s, 0.70, 1.48, 3.55, 4.72, '天气 A：离线建图', '清晰纹理\n稳定阴影\n较高对比度', C.amber, { bodySize: 14 });
  s.addImage({ path: imgRoadA, x: 0.95, y: 3.10, w: 3.05, h: 1.73 });
  addArrow(s, 4.48, 3.73, 5.32, 3.73, C.red, 3);
  s.addShape(ST.ellipse, { x: 5.36, y: 3.30, w: 0.85, h: 0.85, fill: { color: 'FDECEC' }, line: { color: C.red, width: 2 } });
  s.addText('×', { x: 5.36, y: 3.40, w: 0.85, h: 0.42, fontFace: 'Aptos', fontSize: 25, bold: true, color: C.red, align: 'center', margin: 0 });
  addArrow(s, 6.25, 3.73, 7.05, 3.73, C.red, 3);
  addCard(s, 7.28, 1.48, 5.35, 4.72, '天气 B：再次经过', '雨雾、强弱光变化、能见度下降\n→ 关键点重复率下降\n→ 误匹配增加\n→ 重定位失败', C.red, { bodySize: 13.5 });
  s.addImage({ path: imgRoadB, x: 7.58, y: 3.47, w: 4.75, h: 2.12 });
  s.addText('示意说明：图中展示两次独立道路序列的场景外观差异。', { x: 7.62, y: 5.70, w: 4.65, h: 0.23, fontFace: 'Microsoft YaHei', fontSize: 8, color: C.muted, margin: 0, italic: true, fit: 'shrink' });
  s.addText('核心矛盾：不是地图中没有这个地方，而是天气变化后“认不出来”。', { x: 0.88, y: 6.52, w: 11.75, h: 0.34, fontFace: 'Microsoft YaHei', fontSize: 16, bold: true, color: C.red, align: 'center', margin: 0, fit: 'shrink' });
}

// 6. Sign anchor
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '本文思路：增加道路交通标志语义锚点', '问题描述');
  s.addShape(ST.roundRect, { x: 0.72, y: 1.48, w: 5.25, h: 4.90, rectRadius: 0.06, fill: { color: 'FFF7E8' }, line: { color: 'F5D59A' } });
  s.addText('普通道路背景', { x: 1.08, y: 1.85, w: 2.0, h: 0.35, fontFace: 'Microsoft YaHei', fontSize: 18, bold: true, color: '8B5A00', margin: 0 });
  addBulletList(s, ['纹理受雨、雾、光照影响明显', '阴影和动态车辆改变外观', '跨天气匹配稳定性不足'], 1.05, 2.48, 4.25, 1.65, { fontSize: 14, color: '71501A' });
  s.addText('外观线索', { x: 1.08, y: 5.28, w: 1.2, h: 0.28, fontFace: 'Microsoft YaHei', fontSize: 11, bold: true, color: C.muted, margin: 0 });
  s.addShape(ST.chevron, { x: 2.25, y: 5.14, w: 2.8, h: 0.55, fill: { color: C.red, transparency: 15 }, line: { color: C.red, transparency: 100 } });
  s.addText('易退化', { x: 2.78, y: 5.30, w: 1.5, h: 0.18, fontFace: 'Microsoft YaHei', fontSize: 11, bold: true, color: C.white, align: 'center', margin: 0 });
  s.addShape(ST.roundRect, { x: 6.30, y: 1.48, w: 6.30, h: 4.90, rectRadius: 0.06, fill: { color: 'EAF8F5' }, line: { color: 'A7DDD1' } });
  s.addText('道路交通标志', { x: 6.68, y: 1.85, w: 2.2, h: 0.35, fontFace: 'Microsoft YaHei', fontSize: 18, bold: true, color: '087565', margin: 0 });
  addBulletList(s, ['位置固定：可作为两次经过的共同参照', '语义明确：类别与文字可辅助重识别', '形状规范：可提供平面与结构几何约束', '无需预先计算或保存标志牌 GPS'], 6.70, 2.42, 5.20, 2.25, { fontSize: 14, color: '155E54' });
  s.addText('语义 + 几何', { x: 6.72, y: 5.28, w: 1.5, h: 0.28, fontFace: 'Microsoft YaHei', fontSize: 11, bold: true, color: C.muted, margin: 0 });
  s.addShape(ST.chevron, { x: 8.25, y: 5.14, w: 3.55, h: 0.55, fill: { color: C.teal }, line: { color: C.teal, transparency: 100 } });
  s.addText('长期定位锚点', { x: 8.90, y: 5.30, w: 2.2, h: 0.18, fontFace: 'Microsoft YaHei', fontSize: 11, bold: true, color: C.white, align: 'center', margin: 0 });
  s.addText('从“匹配整幅场景”转为“识别固定路标并恢复相对位姿”。', { x: 1.0, y: 6.65, w: 11.4, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 15, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
}

// 7. Direct inter-session displacement via a shared semantic target
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '关键价值：以同一交通标志直接连接两次独立行驶', '问题描述', '标志牌是跨序列几何桥梁，不需要预先计算或保存标志牌自身的 GPS');
  addCard(s, 0.72, 1.48, 5.65, 4.95, '序列 1｜参考经过', '', C.blue);
  addStep(s, 1, '记录参考 GPS', '获取参考相机位置 p_GPS1 与局部行驶朝向', 1.08, 2.26, 4.92, C.blue);
  addStep(s, 2, '观测交通标志', '提取目标 ROI、平面特征与多帧膨胀信息', 1.08, 3.62, 4.92, C.blue);
  addStep(s, 3, '形成参考序列', '保存图像、VIO 轨迹与标志观测，不保存标志 GPS', 1.08, 4.98, 4.92, C.green);
  addArrow(s, 6.60, 3.82, 7.16, 3.82, C.teal, 3);
  addCard(s, 7.25, 1.48, 5.36, 4.95, '序列 2｜跨天气再次经过', '', C.teal);
  addStep(s, 1, '重识别同一标志', '建立序列 1 与序列 2 的共同目标对应', 7.62, 2.26, 4.62, C.teal);
  addStep(s, 2, '直接恢复相对位姿', '计算 T_C1C2 = {R12, t_real}', 7.62, 3.62, 4.62, C.teal);
  addStep(s, 3, '推算当前 GPS', 'p_GPS2 = p_GPS1 + R_align · t_real', 7.62, 4.98, 4.62, C.green);
  s.addText('核心是恢复“两次相机之间的相对位移”；交通标志提供对应与几何约束，而不是作为已知 GPS 地标。', { x: 0.92, y: 6.64, w: 11.48, h: 0.32, fontFace: 'Microsoft YaHei', fontSize: 14.2, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
}

// 8 math
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '实际实现的坐标关系：直接计算跨序列相对位移', '问题描述', 'C₁：参考序列相机；C₂：再次经过相机；交通标志仅作为共享平面与重识别目标');
  addCard(s, 0.78, 1.62, 5.65, 3.55, '视觉部分：恢复两次相机关系', '', C.blue);
  s.addText('T_C₁C₂  =  { R₁₂ , t_real }', { x: 1.22, y: 2.45, w: 4.75, h: 0.58, fontFace: 'Cambria Math', fontSize: 25, bold: true, color: C.blue, align: 'center', margin: 0, fit: 'shrink' });
  s.addText('同一标志匹配     Looming 尺度注入', { x: 1.30, y: 3.31, w: 4.55, h: 0.28, fontFace: 'Microsoft YaHei', fontSize: 11.5, color: C.sub, align: 'center', margin: 0 });
  s.addText('输出：序列 1 到序列 2 的米制相对旋转与平移', { x: 1.20, y: 4.20, w: 4.75, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 13, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
  addCard(s, 6.87, 1.62, 5.65, 3.55, 'GPS 部分：由参考位置推算当前位置', '', C.teal);
  s.addText('p_GPS2 = p_GPS1 + R_align · t_real', { x: 7.20, y: 2.45, w: 5.00, h: 0.58, fontFace: 'Cambria Math', fontSize: 22, bold: true, color: C.teal, align: 'center', margin: 0, fit: 'shrink' });
  s.addText('参考帧 GPS       SLAM→GPS 航向对齐', { x: 7.42, y: 3.31, w: 4.55, h: 0.28, fontFace: 'Microsoft YaHei', fontSize: 11.5, color: C.sub, align: 'center', margin: 0 });
  s.addText('输出：第二次经过时车辆的预测 GPS', { x: 7.34, y: 4.20, w: 4.72, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 13, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
  s.addShape(ST.roundRect, { x: 2.00, y: 5.70, w: 9.32, h: 0.82, rectRadius: 0.05, fill: { color: 'EEF7F6' }, line: { color: 'B7DED8' } });
  s.addText('本方法无需构建 T_WS 或标志牌 GPS；第二次 GPS 仅在实验阶段作为预测真值使用。', { x: 2.25, y: 5.95, w: 8.82, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 15, bold: true, color: '0B6B60', align: 'center', margin: 0, fit: 'shrink' });
}

// 9 questions
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '由应用问题抽象出的四个科学问题', '问题描述');
  const cards = [
    ['01', '跨天气关联', '如何利用交通标志语义，建立天气 A 与天气 B 的可靠路标对应？', C.red],
    ['02', '绝对尺度', '单目视觉缺少米制尺度，如何恢复目标的真实深度？', C.blue],
    ['03', '位姿多解', '平面单应性分解存在多组候选，如何选出物理解？', C.purple],
    ['04', '位置推算', '如何将 t_real 对齐到 GPS 方向，并由参考 GPS 推算当前位置？', C.green]
  ];
  cards.forEach((a,i) => {
    const x = 0.78 + (i%2)*6.12, y = 1.52 + Math.floor(i/2)*2.42;
    s.addShape(ST.roundRect, { x, y, w: 5.68, h: 1.90, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.line } });
    s.addText(a[0], { x: x+0.28, y: y+0.27, w: 0.68, h: 0.35, fontFace: 'Aptos Display', fontSize: 20, bold: true, color: a[3], margin: 0 });
    s.addText(a[1], { x: x+1.15, y: y+0.25, w: 3.8, h: 0.32, fontFace: 'Microsoft YaHei', fontSize: 17, bold: true, color: C.navy, margin: 0 });
    s.addText(a[2], { x: x+1.15, y: y+0.82, w: 4.08, h: 0.58, fontFace: 'Microsoft YaHei', fontSize: 12, color: C.sub, margin: 0, fit: 'shrink' });
  });
  s.addText('论文核心两章分别回答：如何“算准相对位姿”，以及如何“用路标完成跨天气重定位”。', { x: 0.95, y: 6.54, w: 11.45, h: 0.36, fontFace: 'Microsoft YaHei', fontSize: 15, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
}

// 11 overview
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '核心章一：交通标志度量相对位姿估计', '解决方案', '序列内恢复尺度，跨序列恢复姿态，目标结构排除伪解');
  const steps = [
    ['多帧目标观测', 'ROI / 四角点序列', C.blue],
    ['Looming 测距', '时序运动恢复绝对深度', C.cyan],
    ['跨序列匹配', 'LoFTR + RANSAC 单应性', C.purple],
    ['结构消歧', '汉字横/竖拓扑约束', C.amber],
    ['尺度注入', '输出米制 6-DoF 位姿', C.green]
  ];
  steps.forEach((a,i) => {
    const x = 0.55 + i*2.55;
    s.addShape(ST.roundRect, { x, y: 2.05, w: 2.05, h: 2.65, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.line } });
    s.addShape(ST.ellipse, { x: x+0.67, y: 2.42, w: 0.72, h: 0.72, fill: { color: a[2] }, line: { color: a[2] } });
    s.addText(String(i+1), { x: x+0.67, y: 2.61, w: 0.72, h: 0.22, fontFace: 'Aptos', fontSize: 16, bold: true, color: C.white, align: 'center', margin: 0 });
    s.addText(a[0], { x: x+0.18, y: 3.45, w: 1.69, h: 0.32, fontFace: 'Microsoft YaHei', fontSize: 14, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
    s.addText(a[1], { x: x+0.18, y: 3.95, w: 1.69, h: 0.46, fontFace: 'Microsoft YaHei', fontSize: 10.5, color: C.sub, align: 'center', margin: 0, fit: 'shrink' });
    if (i<4) addArrow(s, x+2.08, 3.37, x+2.48, 3.37, C.muted, 1.6);
  });
  s.addShape(ST.roundRect, { x: 1.22, y: 5.47, w: 10.85, h: 0.76, rectRadius: 0.04, fill: { color: 'EEF4FA' }, line: { color: 'D9E5EF' } });
  s.addText('输入：两次独立行驶序列 + 相机内参 + 各自短时 VIO 轨迹    →    输出：具有真实尺度的相对位姿 T_C₁C₂', { x: 1.52, y: 5.70, w: 10.25, h: 0.29, fontFace: 'Microsoft YaHei', fontSize: 13.5, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
}

// 12 looming
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '步骤一：时序运动学尺度锚定', '解决方案', '利用车辆逼近目标产生的视觉膨胀，恢复标志牌绝对深度');
  addCard(s, 0.72, 1.48, 5.10, 4.88, '为什么需要多帧？', '', C.blue);
  addBulletList(s, ['单目 SLAM 的平移只有相对尺度', '相邻两帧像素膨胀量太小，1 px 抖动会显著放大深度误差', '使用更长时间基线与滑动窗口最小二乘，提高数值稳定性', '通过 H_R = K R₁₂ K⁻¹ 消除相机旋转分量'], 1.10, 2.18, 4.30, 2.75, { fontSize: 13 });
  s.addShape(ST.roundRect, { x: 1.12, y: 5.32, w: 4.28, h: 0.62, rectRadius: 0.03, fill: { color: 'EAF2FF' }, line: { color: 'C6D9F5' } });
  s.addText('尺度来源：短时视觉—惯性位移 Δd', { x: 1.35, y: 5.51, w: 3.82, h: 0.22, fontFace: 'Microsoft YaHei', fontSize: 12.5, bold: true, color: C.blue, align: 'center', margin: 0 });
  addCard(s, 6.18, 1.48, 6.42, 4.88, '核心模型', '', C.teal);
  s.addText('Z = r_near · Δd / Δr − Δd', { x: 6.78, y: 2.28, w: 5.18, h: 0.65, fontFace: 'Cambria Math', fontSize: 25, bold: true, color: C.teal, align: 'center', margin: 0, fit: 'shrink' });
  s.addText('r：目标中心到 FOE 的径向距离\nΔr：多帧观测中的径向膨胀量\nΔd：相机沿前进方向的真实位移', { x: 7.08, y: 3.20, w: 4.55, h: 1.26, fontFace: 'Microsoft YaHei', fontSize: 13, color: C.sub, margin: 0.04, breakLine: false, fit: 'shrink', paraSpaceAfterPt: 7 });
  s.addShape(ST.roundRect, { x: 6.82, y: 4.87, w: 5.10, h: 0.82, rectRadius: 0.04, fill: { color: 'EAF8F5' }, line: { color: 'B7DED8' } });
  s.addText('当前实现：16 帧滑动窗口 LSQ\n典型记录：Z = 7.951 m，Δr = 32.6 px', { x: 7.06, y: 5.08, w: 4.62, h: 0.41, fontFace: 'Microsoft YaHei', fontSize: 12, bold: true, color: '0B6B60', align: 'center', margin: 0, fit: 'shrink' });
}

// 13 homography
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '步骤二：跨序列平面几何初解', '解决方案', '在两次独立经过之间恢复旋转、平移方向与标志牌法向量');
  s.addImage({ path: imgRoadA, x: 0.75, y: 1.52, w: 4.65, h: 2.63 });
  s.addImage({ path: imgRoadB, x: 7.95, y: 1.52, w: 4.65, h: 2.63 });
  addPill(s, '序列 1 / 天气 A', 1.76, 4.30, 2.55, C.blue, 'EAF2FF');
  addPill(s, '序列 2 / 天气 B', 8.98, 4.30, 2.55, C.purple, 'F0ECFB');
  s.addShape(ST.roundRect, { x: 5.72, y: 2.01, w: 1.88, h: 1.55, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.line } });
  s.addText('LoFTR\n+\nRANSAC', { x: 6.05, y: 2.33, w: 1.22, h: 0.87, fontFace: 'Aptos', fontSize: 17, bold: true, color: C.purple, align: 'center', margin: 0, fit: 'shrink' });
  addArrow(s, 5.35, 2.78, 5.68, 2.78, C.purple);
  addArrow(s, 7.62, 2.78, 7.91, 2.78, C.purple);
  s.addShape(ST.roundRect, { x: 1.10, y: 5.12, w: 11.10, h: 1.22, rectRadius: 0.05, fill: { color: 'F2F0FA' }, line: { color: 'D8D0EF' } });
  s.addText('H → 分解得到多组候选 (R, t/d, n)', { x: 1.55, y: 5.42, w: 4.35, h: 0.35, fontFace: 'Cambria Math', fontSize: 20, bold: true, color: C.purple, margin: 0, fit: 'shrink' });
  s.addText('旋转 R 可估计｜平移只有方向 t/d｜存在 4 组候选解', { x: 6.02, y: 5.44, w: 5.72, h: 0.32, fontFace: 'Microsoft YaHei', fontSize: 13.5, bold: true, color: C.navy, align: 'right', margin: 0, fit: 'shrink' });
}

// 14 structure
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '步骤三：利用标志内部结构排除伪解', '解决方案');
  addCard(s, 0.75, 1.50, 4.50, 4.95, '结构先验', '交通标志中的汉字/图形包含大量横向与竖向笔画。将这些线段按候选平面逆投影到 3D 后，正确位姿应更符合平行与正交关系。', C.amber, { bodySize: 14 });
  s.addShape(ST.line, { x: 1.32, y: 4.23, w: 1.35, h: 0, line: { color: C.blue, width: 5 } });
  s.addShape(ST.line, { x: 2.02, y: 3.55, w: 0, h: 1.35, line: { color: C.red, width: 5 } });
  s.addText('v_h ⟂ v_v', { x: 2.87, y: 3.94, w: 1.45, h: 0.35, fontFace: 'Cambria Math', fontSize: 19, bold: true, color: C.navy, margin: 0 });
  s.addShape(ST.roundRect, { x: 5.72, y: 1.50, w: 6.86, h: 4.95, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.line } });
  s.addText('候选解结构一致性', { x: 6.10, y: 1.87, w: 2.8, h: 0.32, fontFace: 'Microsoft YaHei', fontSize: 17, bold: true, color: C.navy, margin: 0 });
  const bars = [
    ['候选解 1', 0.72, C.muted, '未通过正深度'],
    ['候选解 2', 0.116, C.green, '当前最优'],
    ['候选解 3', 0.65, C.muted, '未通过正深度'],
    ['候选解 4', 0.589, C.red, '结构误差较大']
  ];
  bars.forEach((b,i) => {
    const y = 2.56 + i*0.77;
    s.addText(b[0], { x: 6.12, y: y+0.06, w: 1.15, h: 0.22, fontFace: 'Microsoft YaHei', fontSize: 10.5, color: C.sub, margin: 0 });
    s.addShape(ST.roundRect, { x: 7.40, y, w: 3.65, h: 0.34, rectRadius: 0.03, fill: { color: 'EDF1F5' }, line: { color: 'EDF1F5' } });
    s.addShape(ST.roundRect, { x: 7.40, y, w: Math.max(0.35, 3.65*b[1]/0.75), h: 0.34, rectRadius: 0.03, fill: { color: b[2] }, line: { color: b[2] } });
    s.addText(b[1].toFixed(3), { x: 11.20, y: y+0.05, w: 0.55, h: 0.20, fontFace: 'Aptos', fontSize: 10, bold: true, color: b[2], margin: 0 });
    s.addText(b[3], { x: 11.78, y: y+0.05, w: 0.58, h: 0.21, fontFace: 'Microsoft YaHei', fontSize: 8.5, color: C.muted, margin: 0, fit: 'shrink' });
  });
  s.addText('E_ortho = | v_hᵀ v_v |', { x: 7.60, y: 5.79, w: 3.05, h: 0.34, fontFace: 'Cambria Math', fontSize: 18, bold: true, color: C.amber, align: 'center', margin: 0 });
  s.addText('数值来自 2026-06-30 单次可追溯运行记录。', { x: 5.98, y: 6.58, w: 6.35, h: 0.25, fontFace: 'Microsoft YaHei', fontSize: 8.8, italic: true, color: C.muted, align: 'right', margin: 0, fit: 'shrink' });
}

// 15 scale injection
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '步骤四：跨模态尺度注入', '解决方案', '把序列内恢复的绝对深度，注入跨序列无尺度平面几何');
  const blocks = [
    ['Looming 深度', 'Z', '米制尺度', C.blue],
    ['中心像素反投影', 'X_c = ZK⁻¹p_c', '目标中心 3D 点', C.cyan],
    ['投影到平面法向', 'd = nᵀX_c', '正交平面距离', C.purple],
    ['尺度恢复', 't_real = (t/d)·d', '米制平移', C.green]
  ];
  blocks.forEach((a,i) => {
    const x = 0.62 + i*3.15;
    s.addShape(ST.roundRect, { x, y: 1.77, w: 2.68, h: 3.72, rectRadius: 0.07, fill: { color: C.white }, line: { color: C.line } });
    s.addShape(ST.rect, { x, y: 1.77, w: 2.68, h: 0.12, fill: { color: a[3] }, line: { color: a[3] } });
    s.addText(a[0], { x: x+0.22, y: 2.28, w: 2.24, h: 0.35, fontFace: 'Microsoft YaHei', fontSize: 15, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
    s.addText(a[1], { x: x+0.20, y: 3.12, w: 2.28, h: 0.70, fontFace: 'Cambria Math', fontSize: i===1 || i===3 ? 17 : 23, bold: true, color: a[3], align: 'center', margin: 0, fit: 'shrink' });
    s.addText(a[2], { x: x+0.25, y: 4.44, w: 2.18, h: 0.40, fontFace: 'Microsoft YaHei', fontSize: 11.5, color: C.sub, align: 'center', margin: 0, fit: 'shrink' });
    if (i<3) addArrow(s, x+2.72, 3.64, x+3.08, 3.64, C.muted, 1.7);
  });
  s.addShape(ST.roundRect, { x: 2.85, y: 5.90, w: 7.65, h: 0.68, rectRadius: 0.04, fill: { color: 'EAF8F5' }, line: { color: 'B7DED8' } });
  s.addText('最终输出：T_C₁C₂ = {R, t_real}，平移具有真实物理单位', { x: 3.10, y: 6.10, w: 7.15, h: 0.26, fontFace: 'Microsoft YaHei', fontSize: 14, bold: true, color: '0B6B60', align: 'center', margin: 0, fit: 'shrink' });
}

// 16 robust
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '从“能运行”走向“可用”：稳健化设计', '解决方案');
  const items = [
    ['长基线 + 滑动窗口', '增大 Δr，减弱像素量化误差', C.blue],
    ['旋转补偿', '去除相机偏航/俯仰对 Looming 的污染', C.cyan],
    ['ROI + RANSAC', '提高平面匹配纯度，过滤错误对应', C.purple],
    ['正深度 + 结构筛选', '排除单应性分解的伪解', C.amber],
    ['自动退化判据', 'Δr、FOE 距离、条件数、覆盖率联合门控', C.red],
    ['联合非线性优化', '统一优化尺度、位姿、法向量与重投影误差', C.green]
  ];
  items.forEach((a,i) => {
    const x = 0.72 + (i%3)*4.18, y = 1.48 + Math.floor(i/3)*2.42;
    addCard(s, x, y, 3.80, 1.90, a[0], a[1], a[2], { titleSize: 14, bodySize: 11 });
  });
  s.addText('E = λ₁E_looming + λ₂E_H + λ₃E_ortho + λ₄E_reproj', { x: 2.65, y: 6.36, w: 8.05, h: 0.36, fontFace: 'Cambria Math', fontSize: 19, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
}

// 17 chapter 2 system
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '核心章二：跨天气重定位系统', '解决方案', '同一交通标志建立跨序列对应，算法直接恢复两次车辆观测之间的相对位姿');
  s.addShape(ST.roundRect, { x: 0.72, y: 1.43, w: 5.70, h: 4.95, rectRadius: 0.08, fill: { color: 'EDF4FF' }, line: { color: 'C8DCF8' } });
  s.addText('离线端｜天气 A', { x: 1.05, y: 1.78, w: 2.2, h: 0.36, fontFace: 'Microsoft YaHei', fontSize: 19, bold: true, color: C.blue, margin: 0 });
  const left = [['D456 图像 + IMU','ORB-SLAM3 局部轨迹'],['参考 RTK-GPS','p_GPS1 与行驶朝向'],['交通标志观测','ROI / 特征 / Looming 序列'],['参考序列保存','图像、轨迹与标志描述信息']];
  left.forEach((a,i)=> addStep(s,i+1,a[0],a[1],1.03,2.38+i*0.91,5.03,C.blue));
  s.addShape(ST.roundRect, { x: 6.90, y: 1.43, w: 5.70, h: 4.95, rectRadius: 0.08, fill: { color: 'EAF8F5' }, line: { color: 'B7DED8' } });
  s.addText('在线端｜天气 B', { x: 7.24, y: 1.78, w: 2.2, h: 0.36, fontFace: 'Microsoft YaHei', fontSize: 19, bold: true, color: C.teal, margin: 0 });
  const right = [['检测交通标志','语义类别与区域'],['跨天气重识别','确认同一物理路标'],['跨序列相对位姿','直接求 T_C₁C₂'],['车辆 GPS 推算','p₂ = p₁ + R_align · t_real']];
  right.forEach((a,i)=> addStep(s,i+1,a[0],a[1],7.22,2.38+i*0.91,5.03,C.teal));
  addArrow(s, 6.45, 3.88, 6.86, 3.88, C.green, 3);
  s.addText('共享：同一交通标志的跨序列对应关系（无需标志牌 GPS）', { x: 3.95, y: 6.57, w: 5.45, h: 0.29, fontFace: 'Microsoft YaHei', fontSize: 13.5, bold: true, color: C.green, align: 'center', margin: 0, fit: 'shrink' });
}

// 18 data sync
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '数据采集与时间同步', '解决方案', '保证相机、IMU、GPS 与 ORB-SLAM3 轨迹处于可审计的时间链路');
  const sensors = [
    ['D456 彩色图像', '848×480 @ 30 FPS', C.blue],
    ['D456 IMU', '约 200 Hz，设备时钟', C.cyan],
    ['G90 RTK-GNSS', 'NMEA UTC，高精度参考', C.green]
  ];
  sensors.forEach((a,i)=> addCard(s,0.75+i*4.20,1.45,3.75,1.55,a[0],a[1],a[2],{titleSize:15,bodySize:12}));
  addArrow(s, 2.62, 3.23, 2.62, 3.76, C.muted);
  addArrow(s, 6.82, 3.23, 6.82, 3.76, C.muted);
  addArrow(s, 11.02, 3.23, 11.02, 3.76, C.muted);
  s.addShape(ST.roundRect, { x: 1.42, y: 3.82, w: 10.50, h: 1.02, rectRadius: 0.06, fill: { color: 'EEF4FA' }, line: { color: 'D3E0EB' } });
  s.addText('设备时间 ↔ 主机 UTC 仿射拟合', { x: 1.78, y: 4.10, w: 3.45, h: 0.33, fontFace: 'Microsoft YaHei', fontSize: 16, bold: true, color: C.navy, margin: 0 });
  s.addText('输出同步残差、每帧 GPS 时间差、卫星数与 HDOP', { x: 5.30, y: 4.10, w: 6.10, h: 0.33, fontFace: 'Microsoft YaHei', fontSize: 13, color: C.sub, align: 'right', margin: 0, fit: 'shrink' });
  const outputs = [['EuRoC 图像/IMU目录',C.blue],['frame_gps.csv',C.green],['ORB-SLAM3 轨迹',C.purple],['metadata.json 质量指标',C.amber]];
  outputs.forEach((a,i)=> addPill(s,a[0],1.00+i*3.05,5.42,2.60,a[1], i===1?'EAF8F0':'EEF3F8'));
  s.addText('说明：未接 GPS PPS，因此属于“有误差度量的软件同步”，不宣称硬件零误差同步。', { x: 1.0, y: 6.34, w: 11.3, h: 0.35, fontFace: 'Microsoft YaHei', fontSize: 12.5, bold: true, color: C.red, align: 'center', margin: 0, fit: 'shrink' });
}

// 20 setup
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '实验平台与当前数据链', '实验结果');
  s.addImage({ path: imgRoadA, x: 0.74, y: 1.48, w: 5.25, h: 2.97 });
  addPill(s, '真实道路采集帧', 2.25, 4.58, 2.25, C.blue, 'EAF2FF');
  addCard(s, 6.35, 1.48, 6.20, 2.97, '硬件与数据', '• Intel RealSense D456\n• G90 RTK-GNSS / NMEA\n• 图像 848×480 @ 30 FPS\n• IMU 约 200 Hz\n• 多段道路序列、逐帧轨迹、ROI/四角点标注', C.blue, { bodySize: 13 });
  s.addShape(ST.roundRect, { x: 0.75, y: 5.12, w: 11.80, h: 1.15, rectRadius: 0.05, fill: { color: C.white }, line: { color: C.line } });
  const pipe = ['采集','SLAM','分段','标注','Looming','LoFTR/H','GPS验证'];
  pipe.forEach((a,i)=>{
    const x=1.05+i*1.62;
    addPill(s,a,x,5.50,1.18,i<5?C.blue:C.purple,i<5?'EAF2FF':'F0ECFB');
    if(i<pipe.length-1) addArrow(s,x+1.20,5.67,x+1.53,5.67,C.muted,1.3);
  });
  s.addText('现状：工程流水线已形成；需要用跨天气成对数据重新完成最终对比实验。', { x: 1.1, y: 6.56, w: 11.2, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 13.5, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
}

// 21 numeric result
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '单次可追溯运行结果', '实验结果', '来源：experiment_results.txt，运行时间 2026-06-30 16:46:08');
  const rows = [
    [{text:'模块',options:{bold:true,color:C.white}}, {text:'指标',options:{bold:true,color:C.white}}, {text:'记录值',options:{bold:true,color:C.white}}],
    ['Looming','滑动窗口 / 膨胀量','16 帧 / 32.6 px'],
    ['Looming','恢复光轴深度 Z','7.951 m'],
    ['跨序列匹配','LoFTR ROI 有效匹配','62 对'],
    ['结构消歧','候选 2 / 候选 4 正交误差','0.1160 / 0.5893'],
    ['尺度注入','恢复平面距离 d','8.697 m'],
    ['尺度注入','恢复平移模长 |t_real|','1.5162 m']
  ];
  s.addTable(rows, { x: 0.82, y: 1.46, w: 8.00, h: 4.70, border: { type:'solid', color:C.line, pt:1 }, fill: C.white, color:C.text, fontFace:'Microsoft YaHei', fontSize:12, margin:0.09, rowH:0.61, colW:[1.65,3.62,2.72], bold:false, valign:'mid', autoFit:false, breakLine:false });
  s.addShape(ST.rect, { x: 0.82, y: 1.46, w: 8.00, h: 0.61, fill: { color: C.navy }, line: { color: C.navy } });
  // redraw header because the overlay creates reliable consistent coloring in LO
  ['模块','指标','记录值'].forEach((t,i)=> s.addText(t,{x:[0.92,2.57,6.19][i],y:1.65,w:[1.35,3.32,2.30][i],h:0.22,fontFace:'Microsoft YaHei',fontSize:12,bold:true,color:C.white,align:'center',margin:0}));
  addCard(s, 9.22, 1.46, 3.30, 2.15, '能够说明', '核心链路已经跑通：\n测距 → 匹配 → 消歧 → 尺度恢复', C.green, { bodySize: 13 });
  addCard(s, 9.22, 3.95, 3.30, 2.15, '不能说明', '单次结果不能证明跨天气定位精度，也不能代替批量真值评估。', C.red, { bodySize: 13 });
  s.addText('阶段判断：方法原型“可运行”，但最终论文结论仍应以跨天气批量实验为准。', { x: 1.0, y: 6.58, w: 11.3, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 14, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
}

// 22 LSQ evidence
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, 'Looming 滑动窗口测距示例', '实验结果', '多帧最小二乘利用更长时间基线，降低两帧像素抖动的影响');
  s.addImage({ path: imgLsq, x: 0.67, y: 1.42, w: 12.00, h: 4.29 });
  addCard(s, 0.94, 5.95, 3.70, 0.76, '当前示例', '窗口 15，拟合深度约 8.2 m', C.blue, { titleSize: 12, bodySize: 10 });
  addCard(s, 4.82, 5.95, 3.70, 0.76, '图中含义', '红点为目标中心，蓝叉为 FOE', C.cyan, { titleSize: 12, bodySize: 10 });
  addCard(s, 8.70, 5.95, 3.70, 0.76, '评价指标', '深度 MAE、RMSE 与相对误差', C.amber, { titleSize: 12, bodySize: 10 });
}

// 23 progress and issue
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '系统实现与当前成果', '实验结果');
  addCard(s, 0.72, 1.45, 5.76, 4.90, '工程系统', '', C.green);
  addBulletList(s, ['D456 图像/IMU 与 GPS 同步采集工具', '在线/离线 ORB-SLAM3 与轨迹导出', '轨迹分段、标注、缓存与可视化', '多帧 Looming 深度恢复原型', 'LoFTR + 单应性分解 + 结构消歧', '跨序列尺度注入计算链'], 1.10, 2.18, 4.82, 3.55, { fontSize: 13, color: '225A48', space: 7 });
  addCard(s, 6.84, 1.45, 5.76, 4.90, '核心算法', '', C.blue);
  addBulletList(s, ['跨序列交通标志 ROI 匹配', '滑动窗口 Looming 深度恢复', '旋转补偿与 FOE 几何建模', '单应性分解与候选位姿生成', '汉字横竖结构约束消歧', '绝对尺度注入与轨迹对齐'], 7.22, 2.18, 4.82, 3.55, { fontSize: 13, color: '244E79', space: 7 });
  s.addText('已形成“数据采集—轨迹恢复—目标匹配—尺度恢复—全局定位”的完整技术链。', { x: 1.25, y: 6.60, w: 10.80, h: 0.30, fontFace: 'Microsoft YaHei', fontSize: 14.5, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
}

// 24 experiment design
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '完整实验设计：验证“跨天气重定位”是否成立', '实验结果');
  const cols = [
    ['A｜跨天气识别', ['晴→雨、晴→雾等天气组合','标志检出率与重识别 Recall@1','传统外观检索 vs 标志辅助'], C.red],
    ['B｜几何精度', ['深度 MAE / RMSE / MAPE','平移误差与旋转误差','GPS 平面定位误差 / ATE'], C.blue],
    ['C｜消融与鲁棒性', ['单帧 vs 多帧','无去旋 / 无结构筛选','距离、FOE、模糊、匹配数分组'], C.purple]
  ];
  cols.forEach((a,i)=>{
    const x=0.72+i*4.18;
    addCard(s,x,1.48,3.80,3.88,a[0],'',a[2],{titleSize:16});
    addBulletList(s,a[1],x+0.38,2.34,3.02,2.28,{fontSize:12.5,space:8});
  });
  s.addShape(ST.roundRect, { x: 0.92, y: 5.78, w: 11.48, h: 0.75, rectRadius: 0.04, fill: { color: 'FFF7E8' }, line: { color: 'F1D39A' } });
  s.addText('结果必须同时报告：样本数 N、均值、中位数、RMSE、95% 分位数、成功率，并按天气组合分组。', { x: 1.28, y: 6.00, w: 10.76, h: 0.29, fontFace: 'Microsoft YaHei', fontSize: 13.5, bold: true, color: '805A0B', align: 'center', margin: 0, fit: 'shrink' });
}

// 26 conclusion (actual total 25? section makes 25 then this 26? count later)
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '结论与创新点凝练', '结论');
  const items = [
    ['1', '跨天气语义锚点', '交通标志在传统视觉地图匹配退化时提供稳定定位参照', C.red],
    ['2', '时序运动尺度锚定', '多帧视觉膨胀与短时 VIO 运动联合恢复绝对深度', C.blue],
    ['3', '目标结构消歧', '利用标志内部横/竖拓扑筛选单应性物理解', C.purple],
    ['4', '相对位移定位', '不求标志牌 GPS；以参考 GPS 加上 t_real 推算当前 GPS', C.green]
  ];
  items.forEach((a,i)=>{
    const y=1.47+i*1.16;
    s.addShape(ST.roundRect,{x:0.85,y,w:11.66,h:0.87,rectRadius:0.05,fill:{color:C.white},line:{color:C.line}});
    s.addShape(ST.ellipse,{x:1.08,y:y+0.18,w:0.50,h:0.50,fill:{color:a[3]},line:{color:a[3]}});
    s.addText(a[0],{x:1.08,y:y+0.30,w:0.50,h:0.17,fontFace:'Aptos',fontSize:12,bold:true,color:C.white,align:'center',margin:0});
    s.addText(a[1],{x:1.83,y:y+0.19,w:2.28,h:0.30,fontFace:'Microsoft YaHei',fontSize:15,bold:true,color:C.navy,margin:0,fit:'shrink'});
    s.addText(a[2],{x:4.35,y:y+0.22,w:7.55,h:0.28,fontFace:'Microsoft YaHei',fontSize:12.5,color:C.sub,margin:0,fit:'shrink'});
  });
  s.addShape(ST.roundRect,{x:1.18,y:6.25,w:10.98,h:0.60,rectRadius:0.04,fill:{color:'EAF8F5'},line:{color:'B7DED8'}});
  s.addText('结论：同一交通标志作为跨序列桥梁，可直接恢复两次经过的米制相对位移并推算当前位置。',{x:1.48,y:6.43,w:10.38,h:0.24,fontFace:'Microsoft YaHei',fontSize:13.5,bold:true,color:'0B6B60',align:'center',margin:0,fit:'shrink'});
}

// 27 next steps
{
  const s = pptx.addSlide('MASTER');
  addTitle(s, '下一步工作与优先级', '结论');
  const tasks = [
    ['P0', '修复 GPS 真值链路', '核对时间戳、帧索引、ENU 转换与坐标方向', C.red],
    ['P0', '采集跨天气成对数据', '同一路段、同一标志、不同天气，保证 RTK 真值', C.red],
    ['P1', '完成批量对比与消融', '传统视觉检索 / PnP / 纯 SLAM / 本文方法', C.amber],
    ['P1', '实现自动退化检测', 'Δr、FOE、单应性条件数、匹配覆盖率联合门控', C.amber],
    ['P2', '统一联合优化', '鲁棒核 + 不确定性权重 + 多约束联合求解', C.green]
  ];
  tasks.forEach((a,i)=>{
    const y=1.40+i*1.02;
    addPill(s,a[0],0.85,y+0.22,0.65,a[3],a[0]==='P0'?'FDECEC':a[0]==='P1'?'FFF4DB':'EAF8F0');
    s.addText(a[1],{x:1.80,y:y+0.16,w:2.75,h:0.28,fontFace:'Microsoft YaHei',fontSize:15,bold:true,color:C.navy,margin:0,fit:'shrink'});
    s.addText(a[2],{x:4.62,y:y+0.18,w:7.35,h:0.27,fontFace:'Microsoft YaHei',fontSize:12.5,color:C.sub,margin:0,fit:'shrink'});
    s.addShape(ST.line,{x:1.82,y:y+0.72,w:10.25,h:0,line:{color:C.line,width:1}});
  });
  s.addShape(ST.roundRect,{x:1.02,y:6.46,w:11.28,h:0.50,rectRadius:0.04,fill:{color:C.navy},line:{color:C.navy}});
  s.addText('近期目标：形成 1 套可复现跨天气数据 + 1 张主结果表 + 1 组消融图 + 1 张完整系统流程图',{x:1.30,y:6.61,w:10.72,h:0.20,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.white,align:'center',margin:0,fit:'shrink'});
}

// 28 ending
{
  const s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addShape(ST.arc, { x: 8.8, y: -1.5, w: 6.4, h: 6.4, rotate: 25, fill: { color: C.teal, transparency: 38 }, line: { color: C.teal, transparency: 100 } });
  s.addText('总结', { x: 0.82, y: 0.90, w: 1.2, h: 0.38, fontFace: 'Microsoft YaHei', fontSize: 16, bold: true, color: C.teal, margin: 0 });
  s.addText('以固定交通标志为跨天气语义锚点', { x: 0.82, y: 1.65, w: 8.8, h: 0.65, fontFace: 'Microsoft YaHei', fontSize: 31, bold: true, color: C.white, margin: 0, fit: 'shrink' });
  s.addText('不计算交通标志牌 GPS\n直接由两次相机相对位移推算当前 GPS', { x: 0.86, y: 2.68, w: 7.5, h: 1.12, fontFace: 'Microsoft YaHei', fontSize: 21, bold: true, color: 'C6D5E1', margin: 0, fit: 'shrink', breakLine: false });
  s.addShape(ST.line,{x:0.86,y:4.28,w:2.15,h:0,line:{color:C.teal,width:4}});
  s.addText('谢谢，请批评指正', { x: 0.86, y: 5.05, w: 4.35, h: 0.55, fontFace: 'Microsoft YaHei', fontSize: 24, bold: true, color: C.white, margin: 0 });
  s.addText('Q & A', { x: 0.86, y: 5.78, w: 2.0, h: 0.45, fontFace: 'Aptos Display', fontSize: 22, bold: true, color: C.teal, margin: 0 });
}

pptx.writeFile({ fileName: root + '/landmarkslam/paper/毕业论文核心两章汇报.pptx' });
