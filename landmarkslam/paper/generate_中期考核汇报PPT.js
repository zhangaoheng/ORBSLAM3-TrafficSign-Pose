const pptxgen = require('/tmp/codex_pptx_tool/node_modules/pptxgenjs');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'ORBSLAM3-TrafficSign-Pose';
pptx.subject = '硕士研究生中期考核汇报';
pptx.title = '基于道路交通标志的跨天气视觉重定位方法研究';
pptx.company = 'Graduate Mid-term Assessment';
pptx.lang = 'zh-CN';
pptx.theme = { headFontFace: 'Microsoft YaHei', bodyFontFace: 'Microsoft YaHei', lang: 'zh-CN' };

const ST = pptx.ShapeType;
const C = {
  navy: '394179', blue: '2367D1', cyan: '0EA5C6', teal: '12B8A6', green: '18A36A',
  amber: 'F59E0B', red: 'E55353', purple: '7657D5', text: '172B3A', sub: '4D6475',
  muted: '74889A', bg: 'F6F8FB', white: 'FFFFFF', line: 'D8E2EC', pale: 'EAF1F7', dark: '102D49'
};
const root = '/home/ehuy/ORBSLAM3-TrafficSign-Pose';
const IMG = {
  roadA: root + '/manual_split/seg_manual_0/rgb/1782374463514750464.png',
  roadB: root + '/manual_split/seg_manual_1/rgb/1782374541542287360.png',
  repeatRef: root + '/manual_split/seg_manual_1/rgb/1782374541342045184.png',
  repeatQuery: root + '/manual_split/seg_manual_181/rgb/1782375931252223232.png',
  loomFar: root + '/manual_split/seg_manual_6/rgb/1782374885023607040.png',
  loomMid: root + '/manual_split/seg_manual_6/rgb/1782374886325147904.png',
  loomNear: root + '/manual_split/seg_manual_6/rgb/1782374887993715456.png',
  looming: root + '/landmarkslam/implement/data/test_pairs/pair_01_45/lsq_results/lsq_result.png',
  gpsFull: root + '/landmarkslam/implement/data/test_pairs/full_gps/gps_trajectory_full.png',
  gpsLocal: root + '/landmarkslam/implement/data/test_pairs/pair_01_45/gps_trajectory.png',
  traj: root + '/landmarkslam/implement/data/test_pairs/pair_15_59/lsq_results/20260605_172512/trajectory_alignment.png'
};
IMG.logoUni = root + '/landmarkslam/paper/assets/reference_theme/anhui_university.png';
IMG.logoCvpr = root + '/landmarkslam/paper/assets/reference_theme/cvpr.png';

pptx.defineSlideMaster({
  title: 'MASTER',
  background: { color: C.white },
  objects: [
    { rect: { x: 0.28, y: 0.22, w: 0.42, h: 0.42, fill: { color: 'FFFFFF' }, line: { color: '8C8C8C', width: 2 } } },
    { rect: { x: 0.53, y: 0.38, w: 0.42, h: 0.42, fill: { color: C.navy }, line: { color: C.navy } } },
    { image: { path: IMG.logoCvpr, x: 9.20, y: 0.02, w: 0.95, h: 0.82, transparency: 0 } },
    { image: { path: IMG.logoUni, x: 10.35, y: 0.13, w: 2.55, h: 0.60, transparency: 0 } },
    { rect: { x: 0, y: 0.92, w: 13.333, h: 0.08, fill: { color: C.navy }, line: { color: C.navy } } },
    { rect: { x: 0, y: 1.00, w: 13.333, h: 0.035, fill: { color: 'B8B8B8', transparency: 30 }, line: { color: 'B8B8B8', transparency: 100 } } }
  ],
  slideNumber: { x: 12.62, y: 7.18, color: '8A8A8A', fontFace: 'Aptos', fontSize: 8 }
});

function title(slide, text, section, sub='') {
  slide.addText(text, { x: 1.18, y: 0.28, w: 7.85, h: 0.43, fontFace: 'Microsoft YaHei', fontSize: 22, bold: true, color: '000000', margin: 0, fit: 'shrink' });
  if (sub) slide.addText(sub, { x: 0.64, y: 1.10, w: 10.8, h: 0.24, fontFace: 'Microsoft YaHei', fontSize: 9.2, color: C.sub, margin: 0, fit: 'shrink' });
  slide.addText(section, { x: 10.45, y: 1.10, w: 2.05, h: 0.22, fontFace: 'Microsoft YaHei', fontSize: 8.8, bold: true, color: C.navy, align: 'right', margin: 0 });
}

function card(slide, x, y, w, h, heading, body='', color=C.blue, opts={}) {
  slide.addShape(ST.roundRect, { x, y, w, h, rectRadius: 0.05, fill: { color: opts.fill || C.white }, line: { color: opts.line || C.line, width: 1 } });
  slide.addShape(ST.roundRect, { x: x+0.16, y: y+0.16, w: 0.075, h: Math.min(0.60, h-0.32), rectRadius: 0.02, fill: { color }, line: { color } });
  slide.addText(heading, { x: x+0.35, y: y+0.15, w: w-0.52, h: 0.27, fontFace: 'Microsoft YaHei', fontSize: opts.titleSize || 14, bold: true, color: opts.titleColor || C.navy, margin: 0, fit: 'shrink' });
  if (body) slide.addText(body, { x: x+0.35, y: y+0.52, w: w-0.55, h: h-0.65, fontFace: 'Microsoft YaHei', fontSize: opts.bodySize || 10.5, color: opts.bodyColor || C.sub, margin: 0, valign: opts.valign || 'top', fit: 'shrink', breakLine: false, paraSpaceAfterPt: 5 });
}

function pill(slide, text, x, y, w, color=C.teal, fill='EAF8F5') {
  slide.addShape(ST.roundRect, { x, y, w, h: 0.34, rectRadius: 0.06, fill: { color: fill }, line: { color: fill } });
  slide.addText(text, { x: x+0.06, y: y+0.07, w: w-0.12, h: 0.16, fontFace: 'Microsoft YaHei', fontSize: 8.8, bold: true, color, align: 'center', margin: 0, fit: 'shrink' });
}

function arrow(slide, x1, y1, x2, y2, color=C.teal, width=2.2) {
  slide.addShape(ST.line, { x: x1, y: y1, w: x2-x1, h: y2-y1, line: { color, width, beginArrowType: 'none', endArrowType: 'triangle' } });
}

function node(slide, x, y, w, h, heading, detail, color=C.blue, fill=C.white) {
  slide.addShape(ST.roundRect, { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { color: C.line, width: 1 } });
  slide.addShape(ST.ellipse, { x: x+0.14, y: y+(h-0.42)/2, w: 0.42, h: 0.42, fill: { color }, line: { color } });
  slide.addText(heading, { x: x+0.68, y: y+0.18, w: w-0.82, h: 0.24, fontFace: 'Microsoft YaHei', fontSize: 12.5, bold: true, color: C.navy, margin: 0, fit: 'shrink' });
  slide.addText(detail, { x: x+0.68, y: y+0.49, w: w-0.82, h: h-0.60, fontFace: 'Microsoft YaHei', fontSize: 8.7, color: C.sub, margin: 0, fit: 'shrink' });
}

function section(num, cn, en, color) {
  const s = pptx.addSlide('MASTER');
  s.addText(num, { x: 1.62, y: 2.95, w: 2.28, h: 1.48, fontFace: 'Times New Roman', fontSize: 82, bold: true, color: C.navy, align: 'center', margin: 0 });
  s.addText('PART  ' + ['ONE','TWO','THREE','FOUR','FIVE','SIX'][Number(num)-1], { x: 1.62, y: 3.70, w: 2.28, h: 0.40, fontFace: 'Times New Roman', fontSize: 19, bold: true, color: C.navy, align: 'center', margin: 0, fit: 'shrink' });
  s.addText(cn, { x: 5.65, y: 3.05, w: 5.20, h: 0.55, fontFace: 'Microsoft YaHei', fontSize: 27, color: '000000', margin: 0, fit: 'shrink' });
  s.addShape(ST.line, { x: 5.45, y: 3.98, w: 6.15, h: 0, line: { color: C.navy, width: 1.2 } });
  s.addShape(ST.ellipse, { x: 11.52, y: 3.91, w: 0.16, h: 0.16, fill: { color: C.navy }, line: { color: C.navy } });
  s.addText(en, { x: 5.75, y: 4.40, w: 5.75, h: 0.35, fontFace: 'Times New Roman', fontSize: 16, color: '000000', margin: 0, fit: 'shrink' });
}

function metric(slide, x, y, w, value, label, color) {
  slide.addShape(ST.roundRect, { x, y, w, h: 0.95, rectRadius: 0.05, fill: { color: C.white }, line: { color: C.line } });
  slide.addText(value, { x: x+0.12, y: y+0.13, w: w-0.24, h: 0.34, fontFace: 'Aptos Display', fontSize: 21, bold: true, color, align: 'center', margin: 0, fit: 'shrink' });
  slide.addText(label, { x: x+0.12, y: y+0.56, w: w-0.24, h: 0.18, fontFace: 'Microsoft YaHei', fontSize: 8.5, color: C.sub, align: 'center', margin: 0, fit: 'shrink' });
}

// 1 封面
{
  const s = pptx.addSlide('MASTER');
  s.addText('中期考核汇报', { x: 3.25, y: 2.18, w: 6.85, h: 0.70, fontFace: 'Microsoft YaHei', fontSize: 37, bold: true, color: '000000', align: 'center', margin: 0 });
  s.addText('基于道路交通标志的跨天气视觉重定位方法研究', { x: 1.40, y: 3.28, w: 10.55, h: 0.50, fontFace: 'Microsoft YaHei', fontSize: 22, color: '000000', align: 'center', margin: 0, fit: 'shrink' });
  s.addText('姓名：________\n学号：________\n导师：________', { x: 4.85, y: 4.35, w: 3.65, h: 1.15, fontFace: 'Microsoft YaHei', fontSize: 16, color: '000000', margin: 0, breakLine: false, lineSpacingMultiple: 1.15 });
  s.addText('2026.10.29', { x: 4.60, y: 6.55, w: 4.10, h: 0.40, fontFace: 'Times New Roman', fontSize: 21, color: '000000', align: 'center', margin: 0 });
}

// 2 目录
{
  const s = pptx.addSlide('MASTER');
  s.addShape(ST.rect,{x:0,y:1.04,w:3.45,h:6.46,fill:{color:C.navy},line:{color:C.navy}});
  s.addShape(ST.line,{x:0.12,y:2.76,w:3.20,h:0,line:{color:'AAB1D1',width:1}});
  s.addShape(ST.line,{x:0.12,y:4.18,w:3.20,h:0,line:{color:'AAB1D1',width:1}});
  s.addText('目 录',{x:0.92,y:3.15,w:1.72,h:0.45,fontFace:'Microsoft YaHei',fontSize:25,bold:true,color:C.white,align:'center',margin:0});
  s.addText('CONTENTS',{x:0.92,y:3.68,w:1.72,h:0.25,fontFace:'Aptos',fontSize:12,bold:true,color:C.white,align:'center',margin:0});
  const items = [
    ['01','研究背景与意义','跨天气重定位为什么需要语义路标',C.red],
    ['02','研究现状与趋势','交通标志定位、语义地图与长期视觉定位',C.amber],
    ['03','研究内容与方案','尺度恢复、跨序列几何、结构消歧与 GPS 推算',C.blue],
    ['04','研究进度与计划','已完成工作、实验状态与后续安排',C.green],
    ['05','论文框架','核心两章与全文组织结构',C.purple],
    ['06','主要参考文献','与本课题最接近的代表性工作',C.teal]
  ];
  items.forEach((it,i)=>{const y=1.45+i*0.90; s.addShape(ST.ellipse,{x:5.12,y,w:0.58,h:0.58,fill:{color:C.navy},line:{color:C.navy}}); s.addText(String(i+1),{x:5.12,y:y+0.13,w:0.58,h:0.20,fontFace:'Aptos',fontSize:13,color:C.white,align:'center',margin:0}); s.addText(it[1],{x:6.35,y:y+0.10,w:4.70,h:0.34,fontFace:'Microsoft YaHei',fontSize:18,color:'000000',margin:0,fit:'shrink'});});
}

// 3
section('01','研究背景与意义','Research background and significance',C.red);

// 4 真实数据场景：同一路段两次经过
{
  const s=pptx.addSlide('MASTER'); title(s,'跨条件重定位：同一路段，两次经过','研究背景与意义','真实采集数据：车辆、阴影和观察距离改变，固定道路标志仍可对应');

  // 两次经过的真实图像，保留完整道路场景
  s.addShape(ST.roundRect,{x:0.52,y:1.42,w:6.02,h:3.60,rectRadius:0.04,fill:{color:C.white},line:{color:'BCCBE0',width:1.2}});
  s.addShape(ST.roundRect,{x:6.79,y:1.42,w:6.02,h:3.60,rectRadius:0.04,fill:{color:C.white},line:{color:'8FD5CA',width:1.2}});
  s.addImage({path:IMG.repeatRef,x:0.65,y:1.55,w:5.76,h:3.26});
  s.addImage({path:IMG.repeatQuery,x:6.92,y:1.55,w:5.76,h:3.26});

  // 标出两帧中的同一组指路牌
  s.addShape(ST.rect,{x:2.72,y:2.23,w:1.10,h:0.46,fill:{color:C.white,transparency:100},line:{color:C.amber,width:3,dash:'dash'}});
  s.addShape(ST.rect,{x:9.38,y:2.29,w:1.10,h:0.48,fill:{color:C.white,transparency:100},line:{color:C.teal,width:3,dash:'dash'}});
  s.addShape(ST.line,{x:3.82,y:2.45,w:5.56,h:0.08,line:{color:C.teal,width:2,dash:'dash',beginArrowType:'none',endArrowType:'triangle'}});
  s.addShape(ST.roundRect,{x:5.02,y:2.13,w:3.20,h:0.64,rectRadius:0.05,fill:{color:'EAF8F5',transparency:5},line:{color:'77CDBF',width:1}});
  s.addText('同一固定交通标志',{x:5.22,y:2.31,w:2.80,h:0.22,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:'087666',align:'center',margin:0,fit:'shrink'});

  // 图像标签直接说明它们在定位闭环中的角色
  s.addShape(ST.rect,{x:0.65,y:4.29,w:5.76,h:0.52,fill:{color:'102D49',transparency:12},line:{color:'102D49',transparency:100}});
  s.addText('第一次经过｜参考序列：GPS + SLAM/VIO + 标志观测',{x:0.86,y:4.46,w:5.35,h:0.20,fontFace:'Microsoft YaHei',fontSize:10.3,bold:true,color:C.white,align:'center',margin:0,fit:'shrink'});
  s.addShape(ST.rect,{x:6.92,y:4.29,w:5.76,h:0.52,fill:{color:'087666',transparency:10},line:{color:'087666',transparency:100}});
  s.addText('第二次经过｜待定位序列：重识别同一标志',{x:7.13,y:4.46,w:5.35,h:0.20,fontFace:'Microsoft YaHei',fontSize:10.3,bold:true,color:C.white,align:'center',margin:0,fit:'shrink'});

  // 下方只保留两条对照关系，避免用抽象流程框代替场景
  s.addShape(ST.roundRect,{x:0.65,y:5.27,w:4.02,h:1.05,rectRadius:0.06,fill:{color:'FFF1F1'},line:{color:'F2B9B9',width:1}});
  s.addText('传统外观重定位',{x:0.92,y:5.48,w:1.68,h:0.23,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.red,margin:0});
  s.addText('车辆 / 阴影 / 天气改变\n历史地图场景特征易失配',{x:2.45,y:5.42,w:1.88,h:0.55,fontFace:'Microsoft YaHei',fontSize:10.1,color:'824040',align:'center',margin:0,fit:'shrink'});
  s.addText('×',{x:4.02,y:5.46,w:0.35,h:0.42,fontFace:'Aptos',fontSize:28,bold:true,color:C.red,align:'center',margin:0});

  arrow(s,4.90,5.79,5.72,5.79,C.teal,2.5);
  s.addShape(ST.roundRect,{x:5.92,y:5.27,w:6.76,h:1.05,rectRadius:0.06,fill:{color:'EAF8F5'},line:{color:'8FD5CA',width:1.2}});
  s.addText('本文思路',{x:6.20,y:5.48,w:1.05,h:0.23,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:'087666',margin:0});
  s.addText('同一标志重识别',{x:7.48,y:5.43,w:1.55,h:0.23,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:C.teal,align:'center',margin:0,fit:'shrink'});
  arrow(s,9.08,5.73,9.55,5.73,C.teal,2.2);
  s.addText('两次观测几何',{x:9.62,y:5.43,w:1.40,h:0.23,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:C.teal,align:'center',margin:0,fit:'shrink'});
  arrow(s,11.07,5.73,11.45,5.73,C.teal,2.2);
  s.addText('T_C1C2 + 参考 GPS\n→ 当前 GPS',{x:11.40,y:5.38,w:1.12,h:0.50,fontFace:'Aptos',fontSize:10.8,bold:true,color:C.green,align:'center',margin:0,fit:'shrink'});

  s.addShape(ST.roundRect,{x:1.45,y:6.55,w:10.45,h:0.48,rectRadius:0.05,fill:{color:'EEF3FA'},line:{color:'C9D6E6'}});
  s.addText('核心：不先求交通标志的 GPS；以其为共同参照，直接恢复两次车辆之间的米制相对位移。',{x:1.72,y:6.69,w:9.90,h:0.20,fontFace:'Microsoft YaHei',fontSize:12.2,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 5 挑战
{
  const s=pptx.addSlide('MASTER'); title(s,'真实道路画面中，问题具体发生在哪里？','研究背景与意义','以同一路段两次采集画面为例');
  s.addImage({path:IMG.repeatRef,x:0.65,y:1.48,w:5.75,h:3.25});
  s.addImage({path:IMG.repeatQuery,x:6.93,y:1.48,w:5.75,h:3.25});
  s.addShape(ST.rect,{x:2.70,y:2.15,w:1.16,h:0.52,fill:{color:C.white,transparency:100},line:{color:C.amber,width:3}});
  s.addShape(ST.rect,{x:9.36,y:2.22,w:1.18,h:0.52,fill:{color:C.white,transparency:100},line:{color:C.teal,width:3}});
  s.addShape(ST.ellipse,{x:1.05,y:2.62,w:1.20,h:0.74,fill:{color:C.white,transparency:100},line:{color:C.red,width:2,dash:'dash'}});
  s.addShape(ST.ellipse,{x:7.62,y:2.58,w:1.52,h:0.90,fill:{color:C.white,transparency:100},line:{color:C.red,width:2,dash:'dash'}});
  pill(s,'参考序列',2.35,4.28,1.55,C.blue,'EDF4FF'); pill(s,'待定位序列',8.95,4.28,1.75,C.teal,'EAF8F5');
  const notes=[
    ['外观变化','车辆、阴影、天气使历史地图特征失配',C.red],
    ['弱视差','前向行驶时标志靠近 FOE，像素变化小',C.amber],
    ['尺度模糊','单目单应性只给出 t/d，缺少米制尺度',C.blue],
    ['多解歧义','同一 H 产生多组姿态，需判断物理真解',C.purple]
  ];
  notes.forEach((n,i)=>{const x=0.66+i*3.12; s.addShape(ST.roundRect,{x,y:5.08,w:2.88,h:1.12,rectRadius:0.05,fill:{color:i===0?'FFF1F1':i===1?'FFF7E8':i===2?'EDF4FF':'F3EFFF'},line:{color:n[2],width:1.2}}); s.addText(String(i+1),{x:x+0.15,y:5.29,w:0.38,h:0.38,fontFace:'Aptos',fontSize:15,bold:true,color:C.white,align:'center',margin:0,fill:{color:n[2]},line:{color:n[2]},shape:ST.ellipse}); s.addText(n[0],{x:x+0.62,y:5.24,w:1.96,h:0.24,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:n[2],margin:0}); s.addText(n[1],{x:x+0.20,y:5.62,w:2.48,h:0.38,fontFace:'Microsoft YaHei',fontSize:9.2,color:C.sub,align:'center',margin:0,fit:'shrink'});});
  s.addText('因此：用“可重复识别的固定标志 + 几何尺度恢复”替代对整个场景外观的强依赖。',{x:1.10,y:6.54,w:11.15,h:0.32,fontFace:'Microsoft YaHei',fontSize:13.5,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 6 数据与意义图
{
  const s=pptx.addSlide('MASTER'); title(s,'应用场景与数据基础','研究背景与意义','D456 图像/IMU + RTK-GNSS + 真实道路重复行驶');
  s.addImage({path:IMG.repeatRef,x:0.62,y:1.45,w:4.08,h:2.30});
  s.addImage({path:IMG.repeatQuery,x:0.62,y:3.93,w:4.08,h:2.30});
  s.addShape(ST.rect,{x:2.06,y:1.92,w:0.82,h:0.36,fill:{color:C.white,transparency:100},line:{color:C.amber,width:2.5}});
  s.addShape(ST.rect,{x:2.35,y:4.46,w:0.84,h:0.38,fill:{color:C.white,transparency:100},line:{color:C.teal,width:2.5}});
  pill(s,'第一次经过',0.86,3.27,1.60,C.blue,'EDF4FF'); pill(s,'第二次经过',0.86,5.75,1.60,C.teal,'EAF8F5');
  s.addImage({path:IMG.gpsFull,x:4.92,y:1.43,w:7.72,h:4.78});
  s.addShape(ST.roundRect,{x:5.42,y:5.73,w:6.72,h:0.72,rectRadius:0.05,fill:{color:C.white,transparency:5},line:{color:C.line}});
  s.addText('D456 图像 30 FPS',{x:5.62,y:5.96,w:1.72,h:0.22,fontFace:'Microsoft YaHei',fontSize:10.5,bold:true,color:C.blue,align:'center',margin:0});
  s.addText('IMU 约 200 Hz',{x:7.72,y:5.96,w:1.56,h:0.22,fontFace:'Microsoft YaHei',fontSize:10.5,bold:true,color:C.teal,align:'center',margin:0});
  s.addText('RTK-GNSS 真值',{x:9.70,y:5.96,w:1.74,h:0.22,fontFace:'Microsoft YaHei',fontSize:10.5,bold:true,color:C.green,align:'center',margin:0});
  s.addText('数据形成完整证据链：真实画面 → 两次重复观测 → SLAM/VIO 运动 → GPS 结果验证',{x:1.10,y:6.56,w:11.05,h:0.30,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 7
section('02','研究现状与趋势','Research status and trend',C.amber);

// 8 研究现状矩阵
{
  const s=pptx.addSlide('MASTER'); title(s,'相关研究在同一个真实定位场景中有什么不同？','研究现状与趋势');
  s.addImage({path:IMG.repeatRef,x:0.62,y:1.48,w:5.72,h:3.23});
  s.addShape(ST.rect,{x:2.66,y:2.14,w:1.16,h:0.50,fill:{color:C.white,transparency:100},line:{color:C.amber,width:3}});
  s.addShape(ST.roundRect,{x:0.92,y:4.96,w:5.12,h:0.92,rectRadius:0.05,fill:{color:'FFF7E8'},line:{color:'F2D9A4'}});
  s.addText('共识：交通标志位置固定、语义明确，适合作为长期语义路标',{x:1.18,y:5.24,w:4.60,h:0.34,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:C.amber,align:'center',margin:0,fit:'shrink'});
  const routes=[['标志绝对地图','需预存标志 GPS / 3D 坐标','Yuan 2017 · Qu 2015',C.red,'FFF1F1'],['稀疏语义地图','需先建立标志点线的 2D–3D 对应','Chen 2022 · You 2019',C.amber,'FFF7E8'],['跨条件语义定位','用语义/3D 描述子抑制季节光照变化','Schönberger 2018 · LAPS-II',C.blue,'EDF4FF'],['本文路线','不建标志绝对地图，用两次观测恢复相对运动','Looming + H + 结构消歧',C.teal,'EAF8F5']];
  routes.forEach((d,i)=>{const y=1.46+i*1.20; s.addShape(ST.roundRect,{x:6.68,y,w:5.92,h:0.98,rectRadius:0.05,fill:{color:d[4]},line:{color:d[3],width:i===3?2.2:1.0}}); s.addText(d[0],{x:6.92,y:y+0.19,w:1.62,h:0.26,fontFace:'Microsoft YaHei',fontSize:12,bold:true,color:d[3],align:'center',margin:0,fit:'shrink'}); s.addText(d[1],{x:8.65,y:y+0.14,w:2.55,h:0.38,fontFace:'Microsoft YaHei',fontSize:9.8,color:C.sub,align:'center',margin:0,fit:'shrink'}); s.addText(d[2],{x:11.25,y:y+0.20,w:1.10,h:0.30,fontFace:'Aptos',fontSize:8.8,bold:true,color:d[3],align:'center',margin:0,fit:'shrink'});});
  s.addText('本课题的位置：保留交通标志的长期稳定性，但把“必须预存标志绝对坐标”改为“两次观测直接建立米制几何联系”。',{x:1.05,y:6.43,w:11.25,h:0.42,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 9
section('03','研究内容与方案','Research contents and methods',C.blue);

// 10 两个研究内容
{
  const s=pptx.addSlide('MASTER'); title(s,'两个核心研究内容：从相对位姿到定位闭环','研究内容与方案');
  s.addShape(ST.roundRect,{x:0.55,y:1.40,w:6.05,h:5.30,rectRadius:0.06,fill:{color:'EDF4FF'},line:{color:'BFD3F2',width:1.3}});
  s.addText('内容一｜恢复两次车辆的米制相对位姿',{x:0.88,y:1.68,w:5.38,h:0.30,fontFace:'Microsoft YaHei',fontSize:15,bold:true,color:C.blue,align:'center',margin:0,fit:'shrink'});
  const loomImgs=[IMG.loomFar,IMG.loomMid,IMG.loomNear];
  loomImgs.forEach((p,i)=>s.addImage({path:p,x:0.83+i*1.78,y:2.15,w:1.62,h:0.92}));
  s.addText('目标逐帧变大',{x:2.18,y:3.20,w:2.75,h:0.24,fontFace:'Microsoft YaHei',fontSize:11,bold:true,color:C.amber,align:'center',margin:0});
  s.addImage({path:IMG.repeatRef,x:0.84,y:3.62,w:2.48,h:1.40});
  s.addImage({path:IMG.repeatQuery,x:3.76,y:3.62,w:2.48,h:1.40});
  for(let i=0;i<7;i++) s.addShape(ST.line,{x:3.12,y:3.90+i*0.12,w:0.72,h:(i%3-1)*0.05,line:{color:i%2?C.teal:C.amber,width:0.9,transparency:20}});
  s.addText('Looming 深度 + 单应性姿态 + 结构消歧 → T_C1C2',{x:0.92,y:5.43,w:5.20,h:0.34,fontFace:'Microsoft YaHei',fontSize:12.2,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
  s.addShape(ST.roundRect,{x:6.75,y:1.40,w:6.03,h:5.30,rectRadius:0.06,fill:{color:'EAF8F5'},line:{color:'A8DDD4',width:1.3}});
  s.addText('内容二｜把相对位姿放入跨条件重定位闭环',{x:7.05,y:1.68,w:5.45,h:0.30,fontFace:'Microsoft YaHei',fontSize:15,bold:true,color:'087666',align:'center',margin:0,fit:'shrink'});
  s.addImage({path:IMG.gpsFull,x:7.05,y:2.12,w:5.42,h:2.90});
  s.addImage({path:IMG.traj,x:7.43,y:4.35,w:2.25,h:1.55});
  s.addText('D456 + IMU + RTK',{x:9.92,y:4.54,w:2.12,h:0.26,fontFace:'Microsoft YaHei',fontSize:12,bold:true,color:C.blue,align:'center',margin:0});
  s.addText('SLAM—GPS 方向对齐\n第二次 GPS 只用于评价',{x:9.82,y:4.96,w:2.36,h:0.62,fontFace:'Microsoft YaHei',fontSize:10.2,bold:true,color:C.sub,align:'center',margin:0,fit:'shrink'});
  arrow(s,6.25,4.02,6.82,4.02,C.purple,3);
  s.addText('方法输出进入系统',{x:5.62,y:3.55,w:2.00,h:0.28,fontFace:'Microsoft YaHei',fontSize:10,bold:true,color:C.purple,align:'center',margin:0,rotate:270});
}

// 11 问题贡献映射
{
  const s=pptx.addSlide('MASTER'); title(s,'一张图看懂：从真实画面到米制相对位姿','研究内容与方案');
  s.addImage({path:IMG.loomFar,x:0.60,y:1.45,w:3.02,h:1.70});
  s.addImage({path:IMG.loomNear,x:0.60,y:3.38,w:3.02,h:1.70});
  s.addShape(ST.rect,{x:1.74,y:1.82,w:0.43,h:0.25,fill:{color:C.white,transparency:100},line:{color:C.amber,width:2.2}});
  s.addShape(ST.rect,{x:1.86,y:3.40,w:0.88,h:0.52,fill:{color:C.white,transparency:100},line:{color:C.amber,width:2.2}});
  s.addText('序列内：标志变大 + SLAM/VIO 米制位移 → 深度 Z',{x:0.67,y:5.25,w:2.90,h:0.58,fontFace:'Microsoft YaHei',fontSize:10.5,bold:true,color:C.blue,align:'center',margin:0,fit:'shrink'});
  arrow(s,3.72,3.28,4.34,3.28,C.blue,3);
  s.addImage({path:IMG.repeatRef,x:4.46,y:1.48,w:3.15,h:1.78});
  s.addImage({path:IMG.repeatQuery,x:4.46,y:3.43,w:3.15,h:1.78});
  s.addShape(ST.rect,{x:5.58,y:1.85,w:0.66,h:0.30,fill:{color:C.white,transparency:100},line:{color:C.teal,width:2.2}});
  s.addShape(ST.rect,{x:5.79,y:3.84,w:0.66,h:0.30,fill:{color:C.white,transparency:100},line:{color:C.teal,width:2.2}});
  s.addText('跨序列：同一标志匹配 → H → (R, t/d, n)',{x:4.56,y:5.38,w:2.94,h:0.38,fontFace:'Microsoft YaHei',fontSize:10.5,bold:true,color:C.purple,align:'center',margin:0,fit:'shrink'});
  arrow(s,7.72,3.28,8.34,3.28,C.purple,3);
  s.addImage({path:IMG.looming,x:8.48,y:1.48,w:4.20,h:1.50});
  s.addImage({path:IMG.traj,x:8.48,y:3.26,w:4.20,h:2.10});
  s.addShape(ST.roundRect,{x:8.70,y:5.55,w:3.74,h:0.58,rectRadius:0.05,fill:{color:'EAF8F5'},line:{color:'8FD5CA'}});
  s.addText('Z 注入 t/d → T_C1C2 = {R, t_real}',{x:8.90,y:5.74,w:3.34,h:0.22,fontFace:'Aptos',fontSize:12,bold:true,color:'087666',align:'center',margin:0,fit:'shrink'});
  s.addText('图像是观测，SLAM/VIO 提供米制运动，交通标志是两次经过的共同几何参照。',{x:1.12,y:6.52,w:11.05,h:0.32,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 12 总体架构
{
  const s=pptx.addSlide('MASTER'); title(s,'总体技术路线：用真实数据把定位闭环串起来','研究内容与方案');
  const xs=[0.48,3.17,5.86,8.55];
  const paths=[IMG.loomNear,IMG.repeatQuery,IMG.looming,IMG.traj];
  const heads=['1  参考序列建立尺度','2  同一标志跨序列对应','3  恢复米制相对位姿','4  对齐并推算当前 GPS'];
  const desc=['多帧膨胀 + SLAM/VIO Δd\n得到标志深度 Z','LoFTR + H\n得到 R、t/d、n 候选','结构消歧 + 尺度注入\nT_C1C2={R,t_real}','参考 GPS + 方向对齐\np₂=p₁+R_align t_real'];
  xs.forEach((x,i)=>{s.addShape(ST.roundRect,{x,y:1.48,w:2.42,h:4.92,rectRadius:0.05,fill:{color:i%2?'EAF8F5':'EDF4FF'},line:{color:i%2?'8FD5CA':'BFD3F2',width:1.2}}); s.addImage({path:paths[i],x:x+0.12,y:1.62,w:2.18,h:i===2?1.36:1.55}); s.addText(heads[i],{x:x+0.16,y:3.37,w:2.10,h:0.54,fontFace:'Microsoft YaHei',fontSize:12.2,bold:true,color:i%2?'087666':C.blue,align:'center',margin:0,fit:'shrink'}); s.addText(desc[i],{x:x+0.18,y:4.20,w:2.06,h:0.74,fontFace:'Microsoft YaHei',fontSize:10.1,color:C.sub,align:'center',margin:0,fit:'shrink'}); s.addText(i===0?'Z':i===1?'R, t/d, n':i===2?'T_C1C2':'GPS₂',{x:x+0.48,y:5.38,w:1.46,h:0.46,fontFace:'Aptos Display',fontSize:21,bold:true,color:i%2?C.teal:C.blue,align:'center',margin:0}); if(i<3) arrow(s,x+2.44,3.72,x+2.67,3.72,C.purple,2.4);});
  s.addShape(ST.rect,{x:9.41,y:1.87,w:0.74,h:0.36,fill:{color:C.white,transparency:100},line:{color:C.green,width:2.2}});
  s.addText('整个链路只需标志在两次经过时能够被重新识别；不需要预先建立标志牌的绝对 GPS 地图。',{x:1.02,y:6.64,w:11.25,h:0.30,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 13 Looming 原理
{
  const s=pptx.addSlide('MASTER'); title(s,'模块一：车辆接近标志时，它在图像中逐帧变大','研究内容与方案');
  const ps=[IMG.loomFar,IMG.loomMid,IMG.loomNear], bx=[2.05,6.20,10.49], by=[2.03,1.92,1.53], bw=[0.20,0.29,0.55], bh=[0.19,0.27,0.45];
  ps.forEach((p,i)=>{const x=0.55+i*4.18; s.addImage({path:p,x,y:1.53,w:3.72,h:2.10}); s.addShape(ST.rect,{x:bx[i],y:by[i],w:bw[i],h:bh[i],fill:{color:C.white,transparency:100},line:{color:C.amber,width:2.8}}); pill(s,['远帧 r₁','中间帧 rₖ','近帧 rₖₐₐₓ'][i],x+1.16,3.82,1.42,i===2?C.teal:C.blue,i===2?'EAF8F5':'EDF4FF'); if(i<2) arrow(s,x+3.78,2.60,x+4.11,2.60,C.teal,2.5);});
  s.addShape(ST.roundRect,{x:0.72,y:4.45,w:3.44,h:1.28,rectRadius:0.05,fill:{color:'EDF4FF'},line:{color:'BFD3F2'}});
  s.addText('SLAM/VIO 给出短时米制位移',{x:0.96,y:4.70,w:2.96,h:0.28,fontFace:'Microsoft YaHei',fontSize:12,bold:true,color:C.blue,align:'center',margin:0,fit:'shrink'});
  s.addText('Δd（米）',{x:1.65,y:5.15,w:1.62,h:0.30,fontFace:'Aptos Display',fontSize:19,bold:true,color:C.blue,align:'center',margin:0});
  arrow(s,4.32,5.10,4.92,5.10,C.purple,3);
  s.addShape(ST.roundRect,{x:5.04,y:4.45,w:4.18,h:1.28,rectRadius:0.05,fill:{color:'FFF7E8'},line:{color:'F2D9A4'}});
  s.addText('像素尺度变化',{x:5.34,y:4.68,w:1.45,h:0.28,fontFace:'Microsoft YaHei',fontSize:12,bold:true,color:C.amber,align:'center',margin:0});
  s.addText('Δr = r_near − r_far',{x:6.86,y:4.66,w:2.02,h:0.34,fontFace:'Aptos Display',fontSize:18,bold:true,color:C.amber,align:'center',margin:0,fit:'shrink'});
  s.addText('Z = r_near · Δd / Δr − Δd',{x:5.38,y:5.14,w:3.48,h:0.34,fontFace:'Aptos Display',fontSize:18,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
  arrow(s,9.38,5.10,9.98,5.10,C.teal,3);
  s.addShape(ST.roundRect,{x:10.10,y:4.45,w:2.50,h:1.28,rectRadius:0.05,fill:{color:'EAF8F5'},line:{color:'8FD5CA'}});
  s.addText('输出绝对深度',{x:10.32,y:4.69,w:2.06,h:0.27,fontFace:'Microsoft YaHei',fontSize:12,bold:true,color:'087666',align:'center',margin:0});
  s.addText('Z（米）',{x:10.62,y:5.14,w:1.46,h:0.34,fontFace:'Aptos Display',fontSize:20,bold:true,color:C.teal,align:'center',margin:0});
  s.addText('不是凭单张图像猜距离：利用实际行驶位移 Δd 与标志在图中的增大量 Δr 恢复米制深度。',{x:1.03,y:6.45,w:11.28,h:0.40,fontFace:'Microsoft YaHei',fontSize:13.2,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 14 去旋与窗口
{
  const s=pptx.addSlide('MASTER'); title(s,'模块一稳健化：先去除相机旋转，再用多帧联合拟合','研究内容与方案');
  s.addImage({path:IMG.loomFar,x:0.62,y:1.52,w:3.55,h:2.00});
  s.addImage({path:IMG.loomNear,x:4.52,y:1.52,w:3.55,h:2.00});
  s.addShape(ST.rect,{x:2.05,y:1.99,w:0.46,h:0.27,fill:{color:C.white,transparency:100},line:{color:C.red,width:2.5,dash:'dash'}});
  s.addShape(ST.rect,{x:6.03,y:1.52,w:0.52,h:0.43,fill:{color:C.white,transparency:100},line:{color:C.amber,width:2.5}});
  s.addShape(ST.arc,{x:3.72,y:2.05,w:1.05,h:0.86,rotate:18,fill:{color:C.white,transparency:100},line:{color:C.blue,width:2.5,endArrowType:'triangle'}});
  s.addText('旋转补偿 H_R = K R K⁻¹',{x:2.98,y:3.72,w:2.78,h:0.28,fontFace:'Aptos',fontSize:12,bold:true,color:C.blue,align:'center',margin:0});
  s.addImage({path:IMG.looming,x:8.45,y:1.50,w:4.25,h:2.72});
  s.addShape(ST.roundRect,{x:0.72,y:4.35,w:3.30,h:1.35,rectRadius:0.05,fill:{color:'FFF1F1'},line:{color:'F2B9B9'}});
  s.addText('如果直接用两帧',{x:0.98,y:4.61,w:2.78,h:0.26,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.red,align:'center',margin:0});
  s.addText('目标膨胀 + 相机旋转 + 像素抖动混在一起',{x:1.00,y:5.03,w:2.74,h:0.44,fontFace:'Microsoft YaHei',fontSize:10.2,color:C.sub,align:'center',margin:0,fit:'shrink'});
  arrow(s,4.20,5.02,4.80,5.02,C.blue,3);
  s.addShape(ST.roundRect,{x:4.94,y:4.35,w:3.30,h:1.35,rectRadius:0.05,fill:{color:'EDF4FF'},line:{color:'BFD3F2'}});
  s.addText('去除旋转成分',{x:5.20,y:4.61,w:2.78,h:0.26,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.blue,align:'center',margin:0});
  s.addText('把后续帧统一到参考相机方向，保留主要的尺度变化',{x:5.20,y:5.03,w:2.78,h:0.44,fontFace:'Microsoft YaHei',fontSize:10.2,color:C.sub,align:'center',margin:0,fit:'shrink'});
  arrow(s,8.42,5.02,9.02,5.02,C.teal,3);
  s.addShape(ST.roundRect,{x:9.16,y:4.35,w:3.42,h:1.35,rectRadius:0.05,fill:{color:'EAF8F5'},line:{color:'8FD5CA'}});
  s.addText('15 帧滑动窗口 LSQ',{x:9.40,y:4.61,w:2.94,h:0.26,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:'087666',align:'center',margin:0});
  s.addText('增大有效 Δr，用多帧共同约束抑制单帧噪声',{x:9.42,y:5.03,w:2.90,h:0.44,fontFace:'Microsoft YaHei',fontSize:10.2,color:C.sub,align:'center',margin:0,fit:'shrink'});
  s.addText('结果已落到真实数据：Frame 406，15 帧窗口，恢复深度 Z ≈ 8.2 m。',{x:1.28,y:6.42,w:10.78,h:0.36,fontFace:'Microsoft YaHei',fontSize:13.5,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 15 LoFTR H
{
  const s=pptx.addSlide('MASTER'); title(s,'模块二：跨序列匹配与单应性初解','研究内容与方案');
  s.addImage({path:IMG.repeatRef,x:0.62,y:1.45,w:5.45,h:3.08});
  s.addImage({path:IMG.repeatQuery,x:7.26,y:1.45,w:5.45,h:3.08});
  s.addShape(ST.rect,{x:2.56,y:2.09,w:1.10,h:0.48,fill:{color:C.white,transparency:100},line:{color:C.amber,width:3}});
  s.addShape(ST.rect,{x:9.57,y:2.15,w:1.12,h:0.50,fill:{color:C.white,transparency:100},line:{color:C.teal,width:3}});
  for(let i=0;i<12;i++){const y=2.13+(i%6)*0.07; s.addShape(ST.line,{x:3.64,y,w:5.90,h:(i%4-1.5)*0.07,line:{color:i%2?C.teal:C.amber,width:0.9,transparency:22}});}
  pill(s,'参考序列 ROI',2.15,4.18,1.75,C.amber,'FFF6E5'); pill(s,'待定位序列 ROI',9.00,4.18,2.05,C.teal,'EAF8F5');
  s.addShape(ST.roundRect,{x:0.82,y:4.78,w:3.25,h:1.18,rectRadius:0.05,fill:{color:'EDF4FF'},line:{color:'BFD3F2'}});
  s.addText('① 先重识别同一标志',{x:1.05,y:5.03,w:2.80,h:0.28,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.blue,align:'center',margin:0});
  s.addText('匹配范围限定在标志牌内部',{x:1.06,y:5.45,w:2.78,h:0.24,fontFace:'Microsoft YaHei',fontSize:10,color:C.sub,align:'center',margin:0});
  arrow(s,4.25,5.36,4.90,5.36,C.purple,2.6);
  s.addShape(ST.roundRect,{x:5.02,y:4.78,w:3.25,h:1.18,rectRadius:0.05,fill:{color:'F3EFFF'},line:{color:'CFC2F3'}});
  s.addText('② LoFTR + RANSAC',{x:5.25,y:5.03,w:2.80,h:0.28,fontFace:'Aptos',fontSize:13,bold:true,color:C.purple,align:'center',margin:0});
  s.addText('建立稠密对应，剔除错误匹配',{x:5.26,y:5.45,w:2.78,h:0.24,fontFace:'Microsoft YaHei',fontSize:10,color:C.sub,align:'center',margin:0});
  arrow(s,8.45,5.36,9.10,5.36,C.teal,2.6);
  s.addShape(ST.roundRect,{x:9.22,y:4.78,w:3.25,h:1.18,rectRadius:0.05,fill:{color:'EAF8F5'},line:{color:'8FD5CA'}});
  s.addText('③ 求 H 并分解',{x:9.45,y:5.03,w:2.80,h:0.28,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:'087666',align:'center',margin:0});
  s.addText('得到多组 R、t/d 和平面法向 n',{x:9.46,y:5.45,w:2.78,h:0.24,fontFace:'Microsoft YaHei',fontSize:10,color:C.sub,align:'center',margin:0,fit:'shrink'});
  s.addText('这一步连接的是两次实际经过的同一块标志，不是两张无关的示意图。',{x:1.28,y:6.48,w:10.78,h:0.32,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 16 多解
{
  const s=pptx.addSlide('MASTER'); title(s,'同一组真实匹配，单应性分解仍会给出多组候选','研究内容与方案');
  s.addImage({path:IMG.repeatRef,x:0.62,y:1.42,w:3.10,h:1.75});
  s.addImage({path:IMG.repeatQuery,x:0.62,y:3.36,w:3.10,h:1.75});
  s.addShape(ST.rect,{x:1.72,y:1.78,w:0.64,h:0.30,fill:{color:C.white,transparency:100},line:{color:C.amber,width:2.4}});
  s.addShape(ST.rect,{x:1.93,y:3.77,w:0.66,h:0.31,fill:{color:C.white,transparency:100},line:{color:C.teal,width:2.4}});
  s.addText('两次真实观测 → H',{x:0.78,y:5.34,w:2.78,h:0.34,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.navy,align:'center',margin:0});
  arrow(s,3.86,3.34,4.48,3.34,C.purple,3);
  const sol=[['候选 1','负深度/法向异常',C.red],['候选 2','结构一致',C.green],['候选 3','负深度',C.red],['候选 4','正交误差较大',C.amber]];
  sol.forEach((d,i)=>{const x=4.62+(i%2)*3.92,y=1.48+Math.floor(i/2)*2.18; s.addShape(ST.roundRect,{x,y,w:3.50,h:1.82,rectRadius:0.06,fill:{color:i===1?'EAF8F2':C.white},line:{color:d[2],width:i===1?2.5:1.2}}); s.addText(d[0],{x:x+0.18,y:y+0.18,w:1.10,h:0.27,fontFace:'Microsoft YaHei',fontSize:13.5,bold:true,color:d[2],margin:0}); s.addShape(ST.line,{x:x+1.46,y:y+1.37,w:0,h:-0.72,line:{color:C.navy,width:2.6,endArrowType:'triangle'}}); s.addShape(ST.line,{x:x+1.46,y:y+1.06,w:0.72*(i%2?1:-1),h:-0.54,line:{color:d[2],width:2.6,endArrowType:'triangle'}}); s.addText(d[1],{x:x+2.12,y:y+0.72,w:1.12,h:0.42,fontFace:'Microsoft YaHei',fontSize:9.6,bold:true,color:d[2],align:'center',margin:0,fit:'shrink'});});
  s.addShape(ST.roundRect,{x:4.95,y:5.94,w:6.85,h:0.70,rectRadius:0.05,fill:{color:'F3EFFF'},line:{color:'CFC2F3'}});
  s.addText('数学上都能解释 H ≠ 物理上都可能；必须用标志内部结构进一步筛选。',{x:5.20,y:6.16,w:6.35,h:0.28,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.purple,align:'center',margin:0,fit:'shrink'});
}

// 17 结构先验
{
  const s=pptx.addSlide('MASTER'); title(s,'模块三：利用真实标志内部的横竖结构筛选物理解','研究内容与方案');
  s.addImage({path:IMG.loomNear,x:0.65,y:1.48,w:6.20,h:3.50});
  s.addShape(ST.rect,{x:3.29,y:1.48,w:0.90,h:0.74,fill:{color:C.white,transparency:100},line:{color:C.amber,width:3}});
  for(let i=0;i<4;i++) s.addShape(ST.line,{x:3.35,y:1.62+i*0.13,w:0.78,h:0,line:{color:C.amber,width:2.0}});
  for(let i=0;i<4;i++) s.addShape(ST.line,{x:3.47+i*0.18,y:1.54,w:0,h:0.62,line:{color:C.teal,width:2.0}});
  s.addShape(ST.roundRect,{x:0.92,y:5.18,w:5.65,h:0.76,rectRadius:0.05,fill:{color:C.white},line:{color:C.line}});
  s.addText('真实交通标志中，文字笔画、箭头与边框提供稳定的横/竖方向',{x:1.15,y:5.43,w:5.18,h:0.28,fontFace:'Microsoft YaHei',fontSize:11.2,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
  arrow(s,7.04,3.23,7.64,3.23,C.purple,3);
  s.addShape(ST.roundRect,{x:7.78,y:1.48,w:4.82,h:1.25,rectRadius:0.05,fill:{color:'F3EFFF'},line:{color:'CFC2F3'}});
  s.addText('① 将横/竖线逆投影到每个候选平面',{x:8.05,y:1.86,w:4.28,h:0.34,fontFace:'Microsoft YaHei',fontSize:12.4,bold:true,color:C.purple,align:'center',margin:0,fit:'shrink'});
  s.addShape(ST.roundRect,{x:7.78,y:3.02,w:4.82,h:1.25,rectRadius:0.05,fill:{color:'EDF4FF'},line:{color:'BFD3F2'}});
  s.addText('② 计算正交一致性  E_ortho = |v_hᵀv_v|',{x:8.05,y:3.40,w:4.28,h:0.34,fontFace:'Microsoft YaHei',fontSize:12.4,bold:true,color:C.blue,align:'center',margin:0,fit:'shrink'});
  s.addShape(ST.roundRect,{x:7.78,y:4.56,w:4.82,h:1.25,rectRadius:0.05,fill:{color:'EAF8F2'},line:{color:'A8DCC5'}});
  s.addText('③ 正深度 + 结构误差最小 → 真实候选解',{x:8.05,y:4.94,w:4.28,h:0.34,fontFace:'Microsoft YaHei',fontSize:12.4,bold:true,color:C.green,align:'center',margin:0,fit:'shrink'});
  s.addText('这个先验来自画面中真实的标志结构，不依赖标志牌的绝对 GPS。',{x:1.40,y:6.48,w:10.45,h:0.34,fontFace:'Microsoft YaHei',fontSize:13.5,bold:true,color:C.navy,align:'center',margin:0});
}

// 18 尺度注入
{
  const s=pptx.addSlide('MASTER'); title(s,'模块四：把参考序列恢复的米制深度传给跨序列位姿','研究内容与方案');
  s.addImage({path:IMG.looming,x:0.62,y:1.45,w:5.82,h:2.08});
  pill(s,'Looming 输出：Z（米）',2.16,3.70,2.70,C.blue,'EDF4FF');
  s.addImage({path:IMG.repeatRef,x:6.82,y:1.45,w:2.78,h:1.57});
  s.addImage({path:IMG.repeatQuery,x:9.88,y:1.45,w:2.78,h:1.57});
  for(let i=0;i<7;i++) s.addShape(ST.line,{x:9.26,y:1.78+i*0.12,w:0.62,h:(i%3-1)*0.05,line:{color:C.teal,width:0.9}});
  pill(s,'H 分解：R、t/d、n',8.31,3.22,2.90,C.purple,'F3EFFF');
  s.addShape(ST.roundRect,{x:0.72,y:4.43,w:3.30,h:1.32,rectRadius:0.05,fill:{color:'EDF4FF'},line:{color:'BFD3F2'}});
  s.addText('X_c = Z K⁻¹p_c',{x:1.03,y:4.83,w:2.68,h:0.37,fontFace:'Aptos Display',fontSize:19,bold:true,color:C.blue,align:'center',margin:0});
  arrow(s,4.18,5.09,4.76,5.09,C.cyan,3);
  s.addShape(ST.roundRect,{x:4.90,y:4.43,w:3.30,h:1.32,rectRadius:0.05,fill:{color:'F3EFFF'},line:{color:'CFC2F3'}});
  s.addText('d = nᵀX_c',{x:5.22,y:4.83,w:2.66,h:0.37,fontFace:'Aptos Display',fontSize:19,bold:true,color:C.purple,align:'center',margin:0});
  arrow(s,8.36,5.09,8.94,5.09,C.teal,3);
  s.addShape(ST.roundRect,{x:9.08,y:4.43,w:3.50,h:1.32,rectRadius:0.05,fill:{color:'EAF8F2'},line:{color:'A8DCC5'}});
  s.addText('t_real = (t/d) · d',{x:9.36,y:4.83,w:2.94,h:0.37,fontFace:'Aptos Display',fontSize:19,bold:true,color:C.green,align:'center',margin:0});
  s.addText('最终得到两次车辆的米制相对位姿 T_C1C2；整个过程没有先计算交通标志牌的 GPS。',{x:1.05,y:6.42,w:11.25,h:0.42,fontFace:'Microsoft YaHei',fontSize:13.5,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 19 系统闭环
{
  const s=pptx.addSlide('MASTER'); title(s,'研究内容二：交通标志如何把两次行驶连成定位闭环','研究内容与方案');
  s.addImage({path:IMG.repeatRef,x:0.55,y:1.45,w:5.35,h:3.02});
  s.addImage({path:IMG.repeatQuery,x:7.43,y:1.45,w:5.35,h:3.02});
  s.addShape(ST.rect,{x:2.46,y:2.08,w:1.08,h:0.46,fill:{color:C.white,transparency:100},line:{color:C.amber,width:3}});
  s.addShape(ST.rect,{x:9.70,y:2.14,w:1.10,h:0.48,fill:{color:C.white,transparency:100},line:{color:C.teal,width:3}});
  arrow(s,3.54,2.31,9.70,2.38,C.teal,2.4);
  s.addShape(ST.roundRect,{x:5.28,y:1.98,w:2.68,h:0.66,rectRadius:0.05,fill:{color:'EAF8F5',transparency:5},line:{color:'8FD5CA'}});
  s.addText('同一固定交通标志',{x:5.50,y:2.20,w:2.24,h:0.22,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:'087666',align:'center',margin:0});
  pill(s,'第一次：参考 GPS p₁',1.68,4.18,2.80,C.blue,'EDF4FF');
  pill(s,'第二次：待定位',8.86,4.18,2.48,C.teal,'EAF8F5');
  s.addImage({path:IMG.gpsLocal,x:0.78,y:4.72,w:3.40,h:1.70});
  arrow(s,4.38,5.58,5.10,5.58,C.blue,2.6);
  s.addShape(ST.roundRect,{x:5.20,y:4.78,w:2.90,h:1.56,rectRadius:0.05,fill:{color:'F3EFFF'},line:{color:'CFC2F3'}});
  s.addText('T_C1C2',{x:5.54,y:5.10,w:2.22,h:0.34,fontFace:'Aptos Display',fontSize:22,bold:true,color:C.purple,align:'center',margin:0});
  s.addText('两次车辆的米制相对位姿',{x:5.50,y:5.60,w:2.30,h:0.34,fontFace:'Microsoft YaHei',fontSize:9.8,color:C.sub,align:'center',margin:0,fit:'shrink'});
  arrow(s,8.20,5.58,8.92,5.58,C.green,2.6);
  s.addShape(ST.roundRect,{x:9.04,y:4.78,w:3.20,h:1.56,rectRadius:0.05,fill:{color:'EAF8F2'},line:{color:'A8DCC5'}});
  s.addText('p₂_pred = p₁ + R_align t_real',{x:9.28,y:5.05,w:2.72,h:0.40,fontFace:'Aptos',fontSize:14,bold:true,color:C.green,align:'center',margin:0,fit:'shrink'});
  s.addText('当前 GPS 预测',{x:9.48,y:5.68,w:2.32,h:0.28,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:'087655',align:'center',margin:0});
  s.addText('第二次 GPS 不参与预测，只在实验阶段用于与 p₂_pred 比较误差。',{x:1.35,y:6.57,w:10.62,h:0.30,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 20 坐标对齐
{
  const s=pptx.addSlide('MASTER'); title(s,'从相机相对位移到 GPS/ENU：为什么还要做方向对齐？','研究内容与方案');
  s.addImage({path:IMG.traj,x:0.62,y:1.45,w:5.80,h:4.36});
  s.addShape(ST.line,{x:2.05,y:4.95,w:2.10,h:-1.56,line:{color:C.blue,width:3,endArrowType:'triangle'}});
  s.addText('SLAM 方向',{x:2.34,y:3.96,w:1.36,h:0.26,fontFace:'Microsoft YaHei',fontSize:11,bold:true,color:C.blue,align:'center',margin:0});
  s.addShape(ST.line,{x:2.05,y:4.95,w:2.58,h:-0.80,line:{color:C.green,width:3,endArrowType:'triangle'}});
  s.addText('GPS/ENU 方向',{x:3.33,y:4.55,w:1.60,h:0.26,fontFace:'Microsoft YaHei',fontSize:11,bold:true,color:C.green,align:'center',margin:0});
  s.addShape(ST.arc,{x:2.32,y:3.78,w:1.40,h:1.05,rotate:8,fill:{color:C.white,transparency:100},line:{color:C.amber,width:2,endArrowType:'triangle'}});
  s.addText('θ_align',{x:2.83,y:4.30,w:0.90,h:0.24,fontFace:'Aptos',fontSize:11,bold:true,color:C.amber,align:'center',margin:0});
  const steps=[['① 从两段轨迹估计航向差','θ_align = θ_GPS − θ_SLAM',C.amber,'FFF7E8'],['② 将米制相对位移旋到 ENU','t_G = R(θ_align)t_real',C.blue,'EDF4FF'],['③ 叠加到参考 GPS 位置','p₂_pred = p₁ + t_G',C.green,'EAF8F2']];
  steps.forEach((d,i)=>{const y=1.48+i*1.58; s.addShape(ST.roundRect,{x:6.84,y,w:5.70,h:1.24,rectRadius:0.05,fill:{color:d[3]},line:{color:d[2],width:1.2}}); s.addText(d[0],{x:7.12,y:y+0.24,w:5.14,h:0.28,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:d[2],align:'center',margin:0,fit:'shrink'}); s.addText(d[1],{x:7.36,y:y+0.70,w:4.66,h:0.30,fontFace:'Aptos Display',fontSize:16,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});});
  s.addText('对齐改变的是坐标轴方向，不改变 T_C1C2 已经恢复的米制位移大小。',{x:1.35,y:6.46,w:10.62,h:0.34,fontFace:'Microsoft YaHei',fontSize:13.3,bold:true,color:C.navy,align:'center',margin:0});
}

// 21 平台数据
{
  const s=pptx.addSlide('MASTER'); title(s,'实验数据是怎样组织成完整验证链的？','研究内容与方案');
  s.addImage({path:IMG.repeatRef,x:0.62,y:1.45,w:3.35,h:1.90});
  s.addImage({path:IMG.repeatQuery,x:0.62,y:3.55,w:3.35,h:1.90});
  pill(s,'D456 RGB + IMU',1.25,2.97,2.05,C.blue,'EDF4FF'); pill(s,'重复经过同一路段',0.98,5.14,2.58,C.teal,'EAF8F5');
  s.addImage({path:IMG.gpsFull,x:4.22,y:1.45,w:5.18,h:4.00});
  pill(s,'G90 RTK-GNSS 完整长轨迹',5.40,5.15,2.82,C.green,'EAF8F2');
  s.addImage({path:IMG.traj,x:9.66,y:1.45,w:3.02,h:2.15});
  s.addImage({path:IMG.gpsLocal,x:9.66,y:3.84,w:3.02,h:2.15});
  s.addText('ORB-SLAM3 轨迹',{x:10.15,y:3.27,w:2.02,h:0.25,fontFace:'Microsoft YaHei',fontSize:11,bold:true,color:C.blue,align:'center',margin:0});
  s.addText('9 组 test_pairs 局部 GPS',{x:9.94,y:5.66,w:2.46,h:0.25,fontFace:'Microsoft YaHei',fontSize:11,bold:true,color:C.teal,align:'center',margin:0});
  s.addText('图像/IMU 采集 → ORB-SLAM3 → 局部序列分段 → 相对位姿 → 用第二次 GPS 做误差评价',{x:0.95,y:6.45,w:11.45,h:0.36,fontFace:'Microsoft YaHei',fontSize:13.2,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 22 结果1
{
  const s=pptx.addSlide('MASTER'); title(s,'阶段结果一：多帧 Looming 深度恢复','研究内容与方案','现有实验图直接来自项目结果');
  s.addImage({path:IMG.looming,x:0.62,y:1.36,w:8.62,h:3.08});
  metric(s,9.48,1.53,1.43,'15 帧','滑动窗口',C.blue);
  metric(s,11.05,1.53,1.43,'Frame 406','参考图像',C.amber);
  metric(s,9.48,2.72,1.43,'8.2 m','恢复深度 Z',C.teal);
  metric(s,11.05,2.72,1.43,'LSQ','多帧直线拟合',C.green);
  card(s,0.72,4.75,11.85,1.45,'结果说明','该保存结果使用 15 帧窗口，通过多帧最小二乘恢复标志光轴深度 Z≈8.2 m，验证了“图像膨胀 + 米制短时运动 → 标志绝对深度”的计算闭环。下一阶段需与可靠真值进行批量比较。',C.blue,{bodySize:12.2,fill:'EDF4FF',line:'BFD3F2'});
}

// 23 结果2
{
  const s=pptx.addSlide('MASTER'); title(s,'阶段结果二：真实匹配上的多解筛选与米制位移','研究内容与方案');
  s.addImage({path:IMG.repeatRef,x:0.58,y:1.45,w:4.05,h:2.29});
  s.addImage({path:IMG.repeatQuery,x:0.58,y:3.95,w:4.05,h:2.29});
  s.addShape(ST.rect,{x:2.02,y:1.92,w:0.82,h:0.36,fill:{color:C.white,transparency:100},line:{color:C.amber,width:2.5}});
  s.addShape(ST.rect,{x:2.30,y:4.49,w:0.84,h:0.38,fill:{color:C.white,transparency:100},line:{color:C.teal,width:2.5}});
  for(let i=0;i<8;i++) s.addShape(ST.line,{x:2.84,y:2.00+i*0.07,w:0.48,h:2.52+(i%3-1)*0.05,line:{color:i%2?C.teal:C.amber,width:0.8,transparency:30}});
  pill(s,'LoFTR 有效匹配 62 对',1.23,6.06,2.75,C.purple,'F3EFFF');
  s.addText('结构正交误差（越小越好）',{x:5.05,y:1.52,w:3.35,h:0.30,fontFace:'Microsoft YaHei',fontSize:12,bold:true,color:C.navy,align:'center',margin:0});
  const bars=[['候选解 2',0.1160,C.green],['候选解 4',0.5893,C.red]];
  bars.forEach((b,i)=>{const y=2.03+i*1.05; s.addText(b[0],{x:4.88,y:y+0.10,w:1.18,h:0.24,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:C.navy,margin:0}); s.addShape(ST.roundRect,{x:6.08,y,w:2.38,h:0.52,rectRadius:0.03,fill:{color:'E9EEF3'},line:{color:'E9EEF3'}}); s.addShape(ST.roundRect,{x:6.08,y,w:Math.max(0.48,2.38*b[1]/0.65),h:0.52,rectRadius:0.03,fill:{color:b[2]},line:{color:b[2]}}); s.addText(b[1].toFixed(4),{x:8.56,y:y+0.10,w:0.72,h:0.24,fontFace:'Aptos',fontSize:12,bold:true,color:b[2],margin:0});});
  s.addShape(ST.roundRect,{x:4.88,y:4.35,w:4.40,h:1.44,rectRadius:0.05,fill:{color:'EAF8F2'},line:{color:'A8DCC5'}});
  s.addText('候选解 2 的结构误差明显更小',{x:5.16,y:4.72,w:3.84,h:0.30,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.green,align:'center',margin:0});
  s.addText('选中物理解 → 尺度注入',{x:5.50,y:5.22,w:3.16,h:0.28,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:'087655',align:'center',margin:0});
  s.addShape(ST.roundRect,{x:9.62,y:1.48,w:3.02,h:4.32,rectRadius:0.05,fill:{color:'F3EFFF'},line:{color:'CFC2F3'}});
  s.addText('2026-06-30\n可追溯结果',{x:9.94,y:1.83,w:2.38,h:0.66,fontFace:'Microsoft YaHei',fontSize:14,bold:true,color:C.purple,align:'center',margin:0});
  s.addText('62 对',{x:10.15,y:2.88,w:1.96,h:0.42,fontFace:'Aptos Display',fontSize:23,bold:true,color:C.blue,align:'center',margin:0});
  s.addText('有效匹配',{x:10.36,y:3.34,w:1.54,h:0.22,fontFace:'Microsoft YaHei',fontSize:10,color:C.sub,align:'center',margin:0});
  s.addText('d = 8.697 m',{x:10.05,y:3.92,w:2.16,h:0.32,fontFace:'Aptos',fontSize:17,bold:true,color:C.purple,align:'center',margin:0});
  s.addText('|t_real| = 1.5162 m',{x:9.90,y:4.68,w:2.46,h:0.32,fontFace:'Aptos',fontSize:16,bold:true,color:C.green,align:'center',margin:0,fit:'shrink'});
  s.addText('这组结果证明“真实画面匹配—分解—消歧—尺度注入”已经跑通；定量精度仍以后续批量实验为准。',{x:4.95,y:6.32,w:7.55,h:0.44,fontFace:'Microsoft YaHei',fontSize:12.2,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 24 结果3
{
  const s=pptx.addSlide('MASTER'); title(s,'阶段结果三：GPS 数据恢复与实验对组织','研究内容与方案');
  s.addImage({path:IMG.gpsLocal,x:0.72,y:1.42,w:5.75,h:4.30});
  s.addImage({path:IMG.traj,x:6.75,y:1.42,w:5.83,h:4.30});
  pill(s,'同一路段两次 GPS 局部轨迹',1.74,5.92,3.65,C.blue,'EDF4FF');
  pill(s,'交通标志桥接后的 SLAM 轨迹可视化',7.55,5.92,4.25,C.teal,'EAF8F5');
  s.addText('已整理 9 组序列对及多次 LSQ/轨迹结果，为后续批量统计与消融实验提供数据基础。',{x:1.0,y:6.52,w:11.35,h:0.28,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 25? Reference slide 22 is section 4; ours now 25. Need preserve overall sequence, still 30 total.
section('04','研究进度与计划','Research progress and plan',C.green);

// 26 progress (but we need 30 total; section count shifted: currently slides 1-24 + section=25)
{
  const s=pptx.addSlide('MASTER'); title(s,'研究进度：已经形成哪些可视、可追溯的结果？','研究进度与计划');
  const ev=[
    [IMG.gpsFull,'数据与真值链','图像/IMU/RTK 同步采集\n9 组 test_pairs 已整理',C.green],
    [IMG.looming,'核心几何链路','Looming + H + 结构消歧\n单组结果已跑通',C.teal],
    [IMG.traj,'轨迹与位置验证','ORB-SLAM3 轨迹已生成\nGPS/SLAM 方向链复核中',C.blue]
  ];
  ev.forEach((e,i)=>{const x=0.55+i*4.22; s.addShape(ST.roundRect,{x,y:1.42,w:3.82,h:3.44,rectRadius:0.05,fill:{color:C.white},line:{color:e[3],width:1.3}}); s.addImage({path:e[0],x:x+0.12,y:1.56,w:3.58,h:1.82}); s.addText(e[1],{x:x+0.25,y:3.57,w:3.32,h:0.28,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:e[3],align:'center',margin:0}); s.addText(e[2],{x:x+0.34,y:4.02,w:3.14,h:0.55,fontFace:'Microsoft YaHei',fontSize:10.3,color:C.sub,align:'center',margin:0,fit:'shrink'});});
  s.addShape(ST.roundRect,{x:0.72,y:5.18,w:3.58,h:1.12,rectRadius:0.05,fill:{color:'FFF1F1'},line:{color:'F2B9B9'}});
  s.addText('近期先做',{x:0.98,y:5.42,w:1.05,h:0.25,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.red,margin:0});
  s.addText('复核 GPS/SLAM 真值链',{x:2.02,y:5.41,w:2.02,h:0.30,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
  arrow(s,4.46,5.74,5.10,5.74,C.amber,2.6);
  s.addShape(ST.roundRect,{x:5.22,y:5.18,w:3.58,h:1.12,rectRadius:0.05,fill:{color:'FFF7E8'},line:{color:'F2D9A4'}});
  s.addText('然后完成',{x:5.48,y:5.42,w:1.15,h:0.25,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.amber,margin:0});
  s.addText('批量统计 + 基线 + 消融',{x:6.61,y:5.41,w:1.93,h:0.30,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
  arrow(s,8.96,5.74,9.60,5.74,C.purple,2.6);
  s.addShape(ST.roundRect,{x:9.72,y:5.18,w:2.90,h:1.12,rectRadius:0.05,fill:{color:'F3EFFF'},line:{color:'CFC2F3'}});
  s.addText('同步进行',{x:9.96,y:5.42,w:1.02,h:0.25,fontFace:'Microsoft YaHei',fontSize:12.5,bold:true,color:C.purple,margin:0});
  s.addText('论文撰写与图表统一',{x:10.92,y:5.41,w:1.46,h:0.30,fontFace:'Microsoft YaHei',fontSize:10.8,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
  s.addText('进度不再只用百分比表示：每一项已完成工作都对应一类实际图像、轨迹或可追溯结果。',{x:1.18,y:6.52,w:10.98,h:0.34,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'});
}

// 27 thesis section
section('05','论文框架','Thesis framework',C.purple);

// 28 thesis framework
{
  const s=pptx.addSlide('MASTER'); title(s,'论文整体结构','论文框架');
  const chapters=[['第一章','绪论','背景、现状、问题与贡献',C.red],['第二章','理论与系统基础','视觉几何、SLAM、Looming、单应性',C.amber],['第三章','交通标志度量相对位姿估计','尺度锚定、结构消歧、米制位姿',C.blue],['第四章','跨天气重定位系统与实验','采集同步、位置推算、对比与消融',C.teal],['第五章','总结与展望','结论、适用边界与后续工作',C.purple]];
  chapters.forEach((c,i)=>{const y=1.38+i*1.02; s.addShape(ST.roundRect,{x:0.88,y,w:1.30,h:0.72,rectRadius:0.05,fill:{color:c[3]},line:{color:c[3]}}); s.addText(c[0],{x:0.98,y:y+0.22,w:1.10,h:0.24,fontFace:'Microsoft YaHei',fontSize:12,bold:true,color:C.white,align:'center',margin:0}); arrow(s,2.22,y+0.36,2.82,y+0.36,c[3]); s.addShape(ST.roundRect,{x:2.90,y,w:3.35,h:0.72,rectRadius:0.05,fill:{color:C.white},line:{color:c[3],width:1.4}}); s.addText(c[1],{x:3.12,y:y+0.20,w:2.92,h:0.26,fontFace:'Microsoft YaHei',fontSize:13,bold:true,color:C.navy,align:'center',margin:0,fit:'shrink'}); arrow(s,6.30,y+0.36,6.92,y+0.36,c[3]); s.addShape(ST.roundRect,{x:7.00,y,w:5.32,h:0.72,rectRadius:0.05,fill:{color:i===2||i===3?'F2EEFF':'F4F7FA'},line:{color:C.line}}); s.addText(c[2],{x:7.22,y:y+0.20,w:4.88,h:0.26,fontFace:'Microsoft YaHei',fontSize:11.3,bold:i===2||i===3,color:i===2||i===3?C.purple:C.sub,align:'center',margin:0,fit:'shrink'});});
  s.addText('论文核心两章',{x:7.18,y:6.48,w:1.60,h:0.24,fontFace:'Microsoft YaHei',fontSize:10,bold:true,color:C.purple,margin:0});
  s.addShape(ST.line,{x:8.82,y:6.59,w:2.55,h:0,line:{color:C.purple,width:3}});
}

// 29 references section + references compressed on one visual slide to keep 30 pages
section('06','主要参考文献','References',C.teal);

// 30 references + thanks would make 31. Instead current count recount: 1,2,3,4,5,6,7,8,9,10-24 (15 slides?) Let's rely on final count check.
{
  const s=pptx.addSlide('MASTER'); title(s,'主要参考文献与相关工作','主要参考文献');
  const refs=[
    '[1] Yuan J Z, Chen H, Zhao B, et al. Estimation of Vehicle Pose and Position with Monocular Camera at Urban Road Intersections. JCST, 2017.',
    '[2] Cui Z, Liu Y, Ren F. Homography-based Traffic Sign Localisation and Pose Estimation from Image Sequence. IET Image Processing, 2019.',
    '[3] Qu X, Soheilian B, Paparoditis N. Vehicle Localization Using Mono-Camera and Geo-Referenced Traffic Signs. IEEE IV, 2015.',
    '[4] Chen X, Xue J, Pang S. Sparse Semantic Map-Based Monocular Localization Using Learned 2D-3D Point-Line Correspondences. RA-L, 2022.',
    '[5] Schönberger J L, Pollefeys M, Geiger A, et al. Semantic Visual Localization. CVPR, 2018.',
    '[6] You C, Wen C, Wang C, et al. Joint 2-D–3-D Traffic Sign Landmark Data Set for Geo-Localization. IEEE T-ITS, 2019.',
    '[7] Maddern W, Stewart A D, McManus C, et al. LAPS-II: 6-DoF Day and Night Visual Localisation with Prior 3D Structure. IEEE IV, 2014.',
    '[8] Sarlin P E, DeTone D, Malisiewicz T, et al. LoFTR: Detector-Free Local Feature Matching with Transformers. CVPR, 2021.',
    '[9] Campos C, Elvira R, Rodríguez J J G, et al. ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual–Inertial and Multi-Map SLAM. IEEE TRO, 2021.'
  ];
  refs.forEach((r,i)=>{const col=i%2,row=Math.floor(i/2),x=0.72+col*6.18,y=1.34+row*1.05; s.addShape(ST.roundRect,{x,y,w:5.90,h:0.82,rectRadius:0.04,fill:{color:i<4?'EEF8F6':C.white},line:{color:i<4?'A8DDD4':C.line}}); s.addText(r,{x:x+0.18,y:y+0.13,w:5.54,h:0.50,fontFace:'Aptos',fontSize:8.8,color:i<4?C.navy:C.sub,margin:0,fit:'shrink'});});
  s.addText('与本课题最接近的工作集中在“交通标志地图定位”和“单应性标志位姿”；本文重点差异是无标志绝对坐标的跨序列尺度传递。',{x:1.0,y:6.68,w:11.3,h:0.25,fontFace:'Microsoft YaHei',fontSize:11.5,bold:true,color:C.teal,align:'center',margin:0,fit:'shrink'});
}

// 结束页
{
  const s=pptx.addSlide('MASTER');
  s.addText('请各位老师批评指正\n谢谢！',{x:1.55,y:2.60,w:10.25,h:1.35,fontFace:'Microsoft YaHei',fontSize:37,color:'000000',align:'center',margin:0,fit:'shrink'});
}

pptx.writeFile({ fileName: root + '/landmarkslam/paper/基于道路交通标志的跨天气视觉重定位_中期考核汇报.pptx' });
