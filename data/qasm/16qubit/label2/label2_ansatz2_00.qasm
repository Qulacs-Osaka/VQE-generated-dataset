OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.5707978529703999) q[0];
rz(1.2175040561423909e-05) q[0];
ry(-1.6695976760330506) q[1];
rz(-2.493781750592118) q[1];
ry(0.07408737140298882) q[2];
rz(0.9509212525422635) q[2];
ry(-3.1170781958103806) q[3];
rz(-1.6149997621533156) q[3];
ry(3.1404728131828006) q[4];
rz(-0.3302435726545153) q[4];
ry(-3.1415926119139477) q[5];
rz(1.8023184130454792) q[5];
ry(2.4851435940370404e-05) q[6];
rz(-0.8424151706007658) q[6];
ry(-3.141587251085003) q[7];
rz(2.2840036056210895) q[7];
ry(-3.141592641177379) q[8];
rz(-3.0657722179978997) q[8];
ry(1.8239179677692616e-07) q[9];
rz(2.7232524034080696) q[9];
ry(-5.910922515894868e-07) q[10];
rz(-1.422159181836685) q[10];
ry(-3.141592534256274) q[11];
rz(1.9562363010479613) q[11];
ry(-3.141592597422555) q[12];
rz(0.9542357851500904) q[12];
ry(-3.1415926181112717) q[13];
rz(2.54908355740205) q[13];
ry(3.141592643482367) q[14];
rz(2.0621277608303314) q[14];
ry(3.1415925318099225) q[15];
rz(-1.954504735519067) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-1.5708116039135858) q[0];
rz(2.0547485659498923) q[0];
ry(-1.9586150448300135e-05) q[1];
rz(-1.73783789057629) q[1];
ry(-3.1415924073384116) q[2];
rz(2.740520491523326) q[2];
ry(3.1415926263468434) q[3];
rz(-2.969163083401424) q[3];
ry(-3.1415876215761123) q[4];
rz(-2.2002523519384067) q[4];
ry(-3.141592565099484) q[5];
rz(-0.3893906616083647) q[5];
ry(3.141403975241651) q[6];
rz(-0.0841000266726697) q[6];
ry(-3.1404160714782194) q[7];
rz(0.9793099052364624) q[7];
ry(-0.019040341907602683) q[8];
rz(-0.784363435254216) q[8];
ry(0.060865904428911044) q[9];
rz(0.6988573281885113) q[9];
ry(1.5709886170041902) q[10];
rz(-3.141592575760093) q[10];
ry(-1.7609398751963559) q[11];
rz(3.1381955947072666) q[11];
ry(0.07160225575762218) q[12];
rz(2.356387612608763) q[12];
ry(0.022401823263500376) q[13];
rz(-1.758494495910643) q[13];
ry(3.1401620480170775) q[14];
rz(-0.4205144583164149) q[14];
ry(0.00024173021253304228) q[15];
rz(-1.4169604691480815) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(0.012441210695484985) q[0];
rz(-1.2504655138646414) q[0];
ry(3.1291359594711974) q[1];
rz(-1.9307589342786295) q[1];
ry(0.023951350307490824) q[2];
rz(-1.7556739522529048) q[2];
ry(3.069425692586465) q[3];
rz(-1.348581966480235) q[3];
ry(1.3806967677522866) q[4];
rz(-0.3835916242606615) q[4];
ry(-1.5707779795640964) q[5];
rz(3.1415612467879783) q[5];
ry(-3.0807180611266745) q[6];
rz(-0.5354038966977505) q[6];
ry(-0.019046849273794242) q[7];
rz(-2.571561835608376) q[7];
ry(-0.0011874671127705128) q[8];
rz(2.617718154963796) q[8];
ry(3.1854800911611666e-05) q[9];
rz(1.1567921393415315) q[9];
ry(1.56977308975826) q[10];
rz(-1.6562208930381574) q[10];
ry(0.0009977085872066025) q[11];
rz(1.4871650384538322) q[11];
ry(-2.4839934767113555e-06) q[12];
rz(-0.5084700946289752) q[12];
ry(-3.141592434328458) q[13];
rz(0.12712465666676476) q[13];
ry(3.1415925981929127) q[14];
rz(-2.6643168269182853) q[14];
ry(3.1415926253634883) q[15];
rz(-2.5410827733492662) q[15];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.1415719959704944) q[0];
rz(-2.793207239617908) q[0];
ry(2.0465458099116564e-05) q[1];
rz(0.38483444604309547) q[1];
ry(3.141564174426228) q[2];
rz(2.7196892355503355) q[2];
ry(2.9093435963467584e-05) q[3];
rz(-0.4605701572312402) q[3];
ry(3.141510010474142) q[4];
rz(-2.192810780029124) q[4];
ry(-1.5707182812069966) q[5];
rz(1.332373230985755) q[5];
ry(-3.713765319623974e-05) q[6];
rz(1.867885717942736) q[6];
ry(-3.1415563732739686) q[7];
rz(-1.2381193486764803) q[7];
ry(-3.516345499221529e-05) q[8];
rz(1.9206983009351486) q[8];
ry(3.487828302617402e-05) q[9];
rz(-1.2422004796134756) q[9];
ry(-3.1415613718840865) q[10];
rz(-1.0426591566091141) q[10];
ry(-3.1415615087408377) q[11];
rz(-1.0442627842124654) q[11];
ry(2.2730802880843726e-05) q[12];
rz(-1.234295845916388) q[12];
ry(-2.23132695964523e-05) q[13];
rz(1.8700210679931406) q[13];
ry(-3.141571638912082) q[14];
rz(-1.5510453114602614) q[14];
ry(3.1415719516005485) q[15];
rz(1.671378266768457) q[15];