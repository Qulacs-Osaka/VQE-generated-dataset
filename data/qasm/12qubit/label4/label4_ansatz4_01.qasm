OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5707990461297783) q[0];
rz(-1.5707959270235454) q[0];
ry(-1.5708113974269489) q[1];
rz(2.008978091298596e-05) q[1];
ry(-1.5707993043344146) q[2];
rz(-2.35941223300136) q[2];
ry(-1.5561478701759341) q[3];
rz(-0.18713367824675586) q[3];
ry(-2.028138323355503) q[4];
rz(0.9567206044835768) q[4];
ry(1.5707964380457318) q[5];
rz(2.110219456302417) q[5];
ry(1.5708329451639091) q[6];
rz(-0.0005612234076236802) q[6];
ry(1.5707960475911327) q[7];
rz(-1.5707956595647907) q[7];
ry(1.570796683340932) q[8];
rz(-9.151430564457996e-07) q[8];
ry(-1.5707978944025467) q[9];
rz(-0.7946583428675476) q[9];
ry(-3.1415876632884148) q[10];
rz(0.15054580109256402) q[10];
ry(-1.667285320892351) q[11];
rz(1.5721502262472864) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.5707957104485795) q[0];
rz(2.8305952617289782) q[0];
ry(-2.830595759904421) q[1];
rz(-1.570775208312205) q[1];
ry(6.00932271515594e-06) q[2];
rz(2.2650776808445325) q[2];
ry(4.0426290376850305e-06) q[3];
rz(-2.7385848297993696) q[3];
ry(1.0257569741739886e-07) q[4];
rz(2.1848701618606565) q[4];
ry(3.141592479117117) q[5];
rz(2.1102190672096324) q[5];
ry(-0.11423634041332362) q[6];
rz(-3.1410333906180536) q[6];
ry(-1.5707980800515442) q[7];
rz(1.685033474541977) q[7];
ry(0.9748993663357943) q[8];
rz(1.5707987529466672) q[8];
ry(0.7895924473087591) q[9];
rz(0.9584836690957167) q[9];
ry(1.5710649380954793) q[10];
rz(1.570794096219275) q[10];
ry(-3.566628121045892e-07) q[11];
rz(-0.0013538886575095506) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5463498425818218) q[0];
rz(-1.7037628335079706) q[0];
ry(1.860091291970046) q[1];
rz(0.23293632326108238) q[1];
ry(-1.5427819475298667) q[2];
rz(2.853616502419762) q[2];
ry(-6.548476729761887e-07) q[3];
rz(-0.2158736237014622) q[3];
ry(-1.5707958000622266) q[4];
rz(0.09658572571571129) q[4];
ry(-1.570794977569345) q[5];
rz(-0.22443059458974357) q[5];
ry(-1.5707966859522813) q[6];
rz(-2.9340403350432016) q[6];
ry(1.5708045944682472) q[7];
rz(-3.141568426391236) q[7];
ry(0.6256432864997017) q[8];
rz(3.1415909697602378) q[8];
ry(0.8785173556713044) q[9];
rz(0.8334310555815195) q[9];
ry(-1.5707972209052588) q[10];
rz(-1.5707927300898685) q[10];
ry(-1.570794851084144) q[11];
rz(-3.094674435264329) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(7.902183041906612e-06) q[0];
rz(1.823030830030675) q[0];
ry(0.7742611415052691) q[1];
rz(-1.8335661173301825) q[1];
ry(1.5707980618573352) q[2];
rz(-1.3166884787200015) q[2];
ry(1.6058862623163364) q[3];
rz(-1.1185451697111149e-06) q[3];
ry(4.043240586781849e-07) q[4];
rz(-0.8550979566940367) q[4];
ry(-1.0977807234620537e-07) q[5];
rz(1.7965013827937857) q[5];
ry(-3.2082600700533703e-06) q[6];
rz(-1.4035669413361802) q[6];
ry(3.0856067449279623) q[7];
rz(2.527973388932736e-05) q[7];
ry(1.5707961865966047) q[8];
rz(-0.2985789792399745) q[8];
ry(-1.5707962632046768) q[9];
rz(1.5705806112590812) q[9];
ry(-1.5707965911281387) q[10];
rz(-0.2366593877836382) q[10];
ry(-3.0347286048170323) q[11];
rz(2.685240096619345) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.1415851059915876) q[0];
rz(1.3342004528374671) q[0];
ry(1.5564907016212581) q[1];
rz(1.4924178130313654) q[1];
ry(2.060769160070772e-06) q[2];
rz(0.9726674036938245) q[2];
ry(1.5707925972540773) q[3];
rz(1.5917245837428582) q[3];
ry(-6.229453397565976e-09) q[4];
rz(-2.730550280931449) q[4];
ry(1.7027412983001353) q[5];
rz(0.020256430521316048) q[5];
ry(-5.459294453835639e-06) q[6];
rz(0.8498409882166654) q[6];
ry(-1.570797116381814) q[7];
rz(-3.1213347258964985) q[7];
ry(1.616178854052234e-06) q[8];
rz(1.5182454510093448) q[8];
ry(1.4537695879123689) q[9];
rz(0.02069872231574621) q[9];
ry(1.5355632033421772e-06) q[10];
rz(-0.10946104274398734) q[10];
ry(8.072078287159348e-07) q[11];
rz(0.5243982779135887) q[11];