OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(3.03908838134553) q[0];
rz(2.6389099836858163) q[0];
ry(3.085454831791317) q[1];
rz(2.2008427288624253) q[1];
ry(-3.141591959883553) q[2];
rz(-0.06756144393772166) q[2];
ry(-0.01244744263864117) q[3];
rz(-2.6307518976024835) q[3];
ry(-0.02003680563259813) q[4];
rz(-3.0176175354307424) q[4];
ry(-0.8690518095780018) q[5];
rz(0.5834419711739756) q[5];
ry(1.5707958161691644) q[6];
rz(3.141592072301196) q[6];
ry(0.9575345976909339) q[7];
rz(0.618379832500292) q[7];
ry(1.5707953893710795) q[8];
rz(1.6725327931510492e-06) q[8];
ry(1.4079799639734011e-09) q[9];
rz(1.3128267276730616) q[9];
ry(1.59186678631111) q[10];
rz(-3.111149846965773) q[10];
ry(-3.1415736364573674) q[11];
rz(2.551146245507151) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.31207613582143856) q[0];
rz(0.8008168296256919) q[0];
ry(-0.20559690586461607) q[1];
rz(-0.8052146953185538) q[1];
ry(-3.1415871993027378) q[2];
rz(2.917359673440561) q[2];
ry(-3.1005933102944807) q[3];
rz(-2.911383047467645) q[3];
ry(-3.0935746926629424) q[4];
rz(-2.1657670201680093) q[4];
ry(-0.8794710956817172) q[5];
rz(1.08036993008252) q[5];
ry(1.0012287678432479) q[6];
rz(-3.1701815661477895e-06) q[6];
ry(-2.6467171852226588) q[7];
rz(2.026677768927371) q[7];
ry(-1.3017217364201734) q[8];
rz(1.5707948920592738) q[8];
ry(-3.141591502579086) q[9];
rz(-2.835268395493151) q[9];
ry(-0.7453788482912501) q[10];
rz(-2.8735860274739973) q[10];
ry(-1.221489219210282e-05) q[11];
rz(1.2955669485278158) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.7412248795545369) q[0];
rz(0.37592639111233817) q[0];
ry(-0.5357925793309056) q[1];
rz(2.49311521243149) q[1];
ry(1.570794785482983) q[2];
rz(3.0548920806320696) q[2];
ry(1.593707071915726) q[3];
rz(-2.3687641932714802) q[3];
ry(-1.557960467484853) q[4];
rz(-0.029416331335197835) q[4];
ry(-2.1498440154107037) q[5];
rz(-1.897040767732679) q[5];
ry(-1.5707963041565964) q[6];
rz(-0.14923806039888632) q[6];
ry(-0.8555139961389743) q[7];
rz(1.7795845637604448) q[7];
ry(1.162110781851881) q[8];
rz(-1.364751454113411) q[8];
ry(3.1554905532971134e-07) q[9];
rz(2.1875259379479273) q[9];
ry(3.0781691986748765) q[10];
rz(1.0762521513661998) q[10];
ry(3.1415846310433735) q[11];
rz(2.767755919963256) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-4.600752569583477e-05) q[0];
rz(1.0628708592285303) q[0];
ry(-3.141580683484962) q[1];
rz(0.34114441056115863) q[1];
ry(-8.535492701611247e-05) q[2];
rz(1.7128252552632501) q[2];
ry(3.1415915024457166) q[3];
rz(-0.8078817211205579) q[3];
ry(1.5706469544573185) q[4];
rz(5.58768138792888e-06) q[4];
ry(-1.5707964668582886) q[5];
rz(2.5190403925253437) q[5];
ry(-3.141290265868975) q[6];
rz(-0.34937733171238605) q[6];
ry(1.570876944650915) q[7];
rz(2.5923938614147064e-05) q[7];
ry(-3.8144663445649485e-06) q[8];
rz(-2.690421461077875) q[8];
ry(-3.1415923313631606) q[9];
rz(0.1358479970045421) q[9];
ry(-3.818556169717676e-05) q[10];
rz(-2.1166124341698938) q[10];
ry(1.5707937557602893) q[11];
rz(3.1415919948219564) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1415630289997027) q[0];
rz(0.8224873698544529) q[0];
ry(-3.1415880938106713) q[1];
rz(0.7262692045320108) q[1];
ry(-3.139921425088614) q[2];
rz(-1.5154405777247102) q[2];
ry(1.5696504828306101) q[3];
rz(-3.122861433153171) q[3];
ry(-1.5707863163985838) q[4];
rz(-0.7750925922630562) q[4];
ry(-1.398711103917094e-05) q[5];
rz(-0.6692779629220298) q[5];
ry(-1.5707916511112379) q[6];
rz(9.072147691822543e-06) q[6];
ry(-1.5707957097866707) q[7];
rz(-1.7242582601432925) q[7];
ry(-3.1415878906596246) q[8];
rz(0.3367154659828819) q[8];
ry(-0.00017135614448005043) q[9];
rz(-0.005684293408474043) q[9];
ry(-3.1415889065358096) q[10];
rz(-2.9025751049049804) q[10];
ry(-1.5707945212217551) q[11];
rz(-2.5416709310705126) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.00026634938227660965) q[0];
rz(0.4002815225496679) q[0];
ry(-3.141559022788391) q[1];
rz(-1.8973136602810765) q[1];
ry(-1.569859174480194) q[2];
rz(-6.9471332439514776e-06) q[2];
ry(-2.0366665659847114e-06) q[3];
rz(1.4922446614017444) q[3];
ry(3.1401233272802296) q[4];
rz(0.8973985779009315) q[4];
ry(-0.007029794903090193) q[5];
rz(0.33026122803259383) q[5];
ry(-1.5707961593268882) q[6];
rz(1.6212110205711392) q[6];
ry(-0.00470007533740624) q[7];
rz(-1.0523478998431486) q[7];
ry(7.924363913806622e-06) q[8];
rz(-1.4056862250879913) q[8];
ry(1.5707963549299049) q[9];
rz(-3.1415924387307452) q[9];
ry(-1.5707945130750884) q[10];
rz(2.4266998567094684) q[10];
ry(-1.5617876945598776) q[11];
rz(1.4285202118653402) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5708326290327446) q[0];
rz(1.5709714173149596) q[0];
ry(-1.5708180811431722) q[1];
rz(-3.259738429419282e-05) q[1];
ry(1.5708117407109827) q[2];
rz(-1.5615025121215962) q[2];
ry(2.9846198713654113) q[3];
rz(-1.631369799463735) q[3];
ry(-5.62517483730259e-06) q[4];
rz(1.9407834694726294) q[4];
ry(3.141584810226543) q[5];
rz(1.385747428234928) q[5];
ry(-0.00024377996492574994) q[6];
rz(-0.05047714543978632) q[6];
ry(-3.028480056702628e-06) q[7];
rz(1.2058007860834634) q[7];
ry(-2.5933448093128326e-06) q[8];
rz(0.11855631060691785) q[8];
ry(-1.5707965505545731) q[9];
rz(2.8743082353240164) q[9];
ry(1.8285396397033082e-07) q[10];
rz(1.9521894832056346) q[10];
ry(1.201514837976578e-06) q[11];
rz(2.0546221929253647) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-1.5707930267136545) q[0];
rz(-0.47053009103824817) q[0];
ry(-1.5707927996102065) q[1];
rz(2.5971475810920133) q[1];
ry(1.5125152615067927) q[2];
rz(2.2631312156413346) q[2];
ry(-1.5707943936639968) q[3];
rz(-2.1848221143226962) q[3];
ry(3.138684471743715) q[4];
rz(-0.9336141607936908) q[4];
ry(-3.1414848014948844) q[5];
rz(-0.7942843235054359) q[5];
ry(-1.5707965093761045) q[6];
rz(-1.5707889935629016) q[6];
ry(-1.5707533284671622) q[7];
rz(-1.8720474302510257) q[7];
ry(-0.00015549679369097902) q[8];
rz(0.036835267726840655) q[8];
ry(1.256265148530389e-05) q[9];
rz(0.5899182611865604) q[9];
ry(-5.728511714367964e-05) q[10];
rz(0.3100877768164019) q[10];
ry(7.142066168341898e-05) q[11];
rz(0.30437398513451447) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.00165532845694488) q[0];
rz(-2.67107857066946) q[0];
ry(-4.57656494278002e-05) q[1];
rz(-2.4984384102651225) q[1];
ry(0.0009158164079649852) q[2];
rz(-2.263135063908134) q[2];
ry(3.1414119042381237) q[3];
rz(0.9567403201554895) q[3];
ry(-3.1415890012913623) q[4];
rz(1.7358002619973156) q[4];
ry(1.5708125276300728) q[5];
rz(3.0797622268812344) q[5];
ry(2.845264704105022) q[6];
rz(-1.570791412703308) q[6];
ry(-3.141586179785313) q[7];
rz(-0.3045406656039575) q[7];
ry(1.6112707606651675) q[8];
rz(-1.7754925836970408) q[8];
ry(-3.1415907952339115) q[9];
rz(0.3232681900738865) q[9];
ry(3.1412823900633815) q[10];
rz(1.546666860258074) q[10];
ry(-1.5454327654929045e-05) q[11];
rz(-3.137431185425182) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.5707982325481558) q[0];
rz(-1.6462247499975096) q[0];
ry(3.1322050921379048) q[1];
rz(-0.6682231878828464) q[1];
ry(-1.5707995027436785) q[2];
rz(1.495312674773881) q[2];
ry(1.5557250254055326) q[3];
rz(0.8038759221347886) q[3];
ry(-1.5707972682383717) q[4];
rz(1.495263121491427) q[4];
ry(1.570799808927297) q[5];
rz(-0.7668882448350666) q[5];
ry(1.570929703619958) q[6];
rz(1.4952693750018469) q[6];
ry(-1.5707865291127705) q[7];
rz(0.8038592796002304) q[7];
ry(3.1415121681058005) q[8];
rz(-1.8510239887855295) q[8];
ry(-1.5707946307832934) q[9];
rz(-2.337714108241547) q[9];
ry(1.5708041355688556) q[10];
rz(-0.07552926820719641) q[10];
ry(-1.5707919119313276) q[11];
rz(2.374663119675154) q[11];