OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.5714179415953238) q[0];
rz(2.962970653322914) q[0];
ry(3.1080571686915386) q[1];
rz(-1.5678028939615727) q[1];
ry(-0.007803652635291592) q[2];
rz(-1.5737662057583304) q[2];
ry(-0.0026995532085286084) q[3];
rz(0.005221842260279989) q[3];
ry(0.00030916342054165824) q[4];
rz(2.4400496603719266) q[4];
ry(-3.140977864360376) q[5];
rz(1.5070016019630987) q[5];
ry(0.005466722311006598) q[6];
rz(-2.0765201304108) q[6];
ry(3.1415527115270794) q[7];
rz(-0.2959778342749243) q[7];
ry(3.141421138405193) q[8];
rz(1.1700462984766222) q[8];
ry(3.141550127864609) q[9];
rz(0.3965427231364194) q[9];
ry(3.140676421318641) q[10];
rz(0.9985597323394058) q[10];
ry(3.138131352788869) q[11];
rz(-2.5238291047035184) q[11];
ry(-0.010630369136006124) q[12];
rz(-0.655194677290174) q[12];
ry(-3.1100656687504413) q[13];
rz(0.9380668016166311) q[13];
ry(3.0501213694402294) q[14];
rz(-2.95663815251594) q[14];
ry(-2.8926681858680348) q[15];
rz(-0.1325003761075383) q[15];
ry(0.5938332353784581) q[16];
rz(-1.3133072942379247) q[16];
ry(-1.069831866875893) q[17];
rz(-2.16028294119444) q[17];
ry(0.9154961794114486) q[18];
rz(2.9085953410805625) q[18];
ry(2.8066254204979644) q[19];
rz(-0.7586376769997248) q[19];
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
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
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
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
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
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
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
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
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
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
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
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(-1.5709696869195022) q[0];
rz(0.9260712288118658) q[0];
ry(0.15529809134654207) q[1];
rz(1.5702010877263426) q[1];
ry(0.5879962400441672) q[2];
rz(1.5720535705451422) q[2];
ry(1.5707328055783822) q[3];
rz(-0.1167859105486162) q[3];
ry(2.562788237021612) q[4];
rz(-1.571884686601365) q[4];
ry(-0.1041461995977648) q[5];
rz(-1.5711411459507563) q[5];
ry(0.0012955532413775235) q[6];
rz(0.47468556681957136) q[6];
ry(0.00261915377670789) q[7];
rz(-1.6373257405934212) q[7];
ry(3.13401327158504) q[8];
rz(1.5609822675960814) q[8];
ry(-0.0038283548329652427) q[9];
rz(1.538730535874345) q[9];
ry(3.140219017418784) q[10];
rz(-1.6341110309876479) q[10];
ry(-3.1413357419459063) q[11];
rz(0.6994417666176659) q[11];
ry(2.5274530520924543e-05) q[12];
rz(0.7950823997617436) q[12];
ry(1.6075551250712992e-05) q[13];
rz(-2.1731976195281746) q[13];
ry(-3.1415815425500435) q[14];
rz(-1.30039241178555) q[14];
ry(3.1415457869652426) q[15];
rz(-3.080273616412594) q[15];
ry(3.1412204873992744) q[16];
rz(0.22850284158926826) q[16];
ry(-3.140707816666346) q[17];
rz(-3.135296990593775) q[17];
ry(0.00038762879570963804) q[18];
rz(-1.786732369997872) q[18];
ry(-3.141442177314324) q[19];
rz(0.3929871641426459) q[19];
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
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
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
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
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
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
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
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
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
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
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
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(0.0003146088720193574) q[0];
rz(-2.4968628748502524) q[0];
ry(-0.021882085118134498) q[1];
rz(-1.5865822193759505) q[1];
ry(3.081864421139988) q[2];
rz(1.5754623191139343) q[2];
ry(1.5705946668827258) q[3];
rz(-0.012709246337946432) q[3];
ry(2.9849574761512905) q[4];
rz(1.5709426698514841) q[4];
ry(2.568813133899553) q[5];
rz(1.5992597937312656) q[5];
ry(-1.5704121480868887) q[6];
rz(3.1343537746541004) q[6];
ry(-2.525089416068291) q[7];
rz(-1.5128525977737652) q[7];
ry(-2.9891234443916255) q[8];
rz(-2.1480075730117116) q[8];
ry(0.04053677286884661) q[9];
rz(1.4200719458987152) q[9];
ry(3.13102826611035) q[10];
rz(-2.2784353478444164) q[10];
ry(3.1388548822294737) q[11];
rz(2.51685171733317) q[11];
ry(-0.0006636718088284255) q[12];
rz(1.5937311432634704) q[12];
ry(4.970794229529289e-05) q[13];
rz(-0.5062684863188359) q[13];
ry(3.1415754898089183) q[14];
rz(1.9405581324831938) q[14];
ry(8.913279755162762e-05) q[15];
rz(1.2247766346093905) q[15];
ry(-1.120889253769519e-05) q[16];
rz(1.2899730086326242) q[16];
ry(3.141310116654136) q[17];
rz(3.0441374634630285) q[17];
ry(-1.017836233874192e-05) q[18];
rz(-2.5114753191535524) q[18];
ry(1.5651838179411138e-05) q[19];
rz(0.5839570100448891) q[19];
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
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
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
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
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
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
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
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
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
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
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
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(-1.6051055786136599) q[0];
rz(-3.1295178163912385) q[0];
ry(-3.077739193186235) q[1];
rz(3.125073644894613) q[1];
ry(-0.022153182845675048) q[2];
rz(3.137516084866963) q[2];
ry(-3.114498868612477) q[3];
rz(-0.011758978187875966) q[3];
ry(3.127220059405384) q[4];
rz(-3.1409823381776816) q[4];
ry(3.135205926699397) q[5];
rz(-3.1130010744162813) q[5];
ry(-1.5702564370263197) q[6];
rz(1.570833137732578) q[6];
ry(-0.003785742901766251) q[7];
rz(3.083603754492438) q[7];
ry(-0.0011475540281059256) q[8];
rz(0.5771709512726197) q[8];
ry(-3.1412284825077874) q[9];
rz(-0.15073074144174203) q[9];
ry(8.796612365635781e-05) q[10];
rz(-2.434378923298239) q[10];
ry(-2.3676411530169217e-06) q[11];
rz(2.268319697845617) q[11];
ry(-3.141542772617116) q[12];
rz(0.1641775727587671) q[12];
ry(0.00010812586865860396) q[13];
rz(0.17054056695628436) q[13];
ry(3.141491952586533) q[14];
rz(-1.2865035197709052) q[14];
ry(-0.0005202555013991691) q[15];
rz(0.15220365951065354) q[15];
ry(0.0002768226851088684) q[16];
rz(-1.2609805030156256) q[16];
ry(-3.1401391484586743) q[17];
rz(2.448320275767731) q[17];
ry(3.1385936837059027) q[18];
rz(-2.960404989638139) q[18];
ry(0.0013582663054103605) q[19];
rz(-0.16477023553744805) q[19];
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
cz q[0],q[16];
cz q[0],q[17];
cz q[0],q[18];
cz q[0],q[19];
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
cz q[1],q[16];
cz q[1],q[17];
cz q[1],q[18];
cz q[1],q[19];
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
cz q[2],q[16];
cz q[2],q[17];
cz q[2],q[18];
cz q[2],q[19];
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
cz q[3],q[16];
cz q[3],q[17];
cz q[3],q[18];
cz q[3],q[19];
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
cz q[4],q[16];
cz q[4],q[17];
cz q[4],q[18];
cz q[4],q[19];
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
cz q[5],q[16];
cz q[5],q[17];
cz q[5],q[18];
cz q[5],q[19];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[6],q[16];
cz q[6],q[17];
cz q[6],q[18];
cz q[6],q[19];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[7],q[16];
cz q[7],q[17];
cz q[7],q[18];
cz q[7],q[19];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[8],q[16];
cz q[8],q[17];
cz q[8],q[18];
cz q[8],q[19];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[9],q[16];
cz q[9],q[17];
cz q[9],q[18];
cz q[9],q[19];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[10],q[16];
cz q[10],q[17];
cz q[10],q[18];
cz q[10],q[19];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[11],q[16];
cz q[11],q[17];
cz q[11],q[18];
cz q[11],q[19];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[12],q[16];
cz q[12],q[17];
cz q[12],q[18];
cz q[12],q[19];
cz q[13],q[14];
cz q[13],q[15];
cz q[13],q[16];
cz q[13],q[17];
cz q[13],q[18];
cz q[13],q[19];
cz q[14],q[15];
cz q[14],q[16];
cz q[14],q[17];
cz q[14],q[18];
cz q[14],q[19];
cz q[15],q[16];
cz q[15],q[17];
cz q[15],q[18];
cz q[15],q[19];
cz q[16],q[17];
cz q[16],q[18];
cz q[16],q[19];
cz q[17],q[18];
cz q[17],q[19];
cz q[18],q[19];
ry(-1.0498756590706984) q[0];
rz(2.698187759440883) q[0];
ry(-2.082545085370731) q[1];
rz(-0.42255001866995906) q[1];
ry(0.9896243469538214) q[2];
rz(2.7187390157498226) q[2];
ry(1.7814684927205429) q[3];
rz(-0.42300739248021946) q[3];
ry(0.956881473035927) q[4];
rz(2.7186729731065395) q[4];
ry(0.9430387067377097) q[5];
rz(-0.42252644162574576) q[5];
ry(1.561895522462512) q[6];
rz(-0.4224516058744508) q[6];
ry(2.1197149569326985) q[7];
rz(2.7191551279074333) q[7];
ry(-0.9643265364965563) q[8];
rz(-0.42235521796042796) q[8];
ry(2.07868969361518) q[9];
rz(2.7192213554251516) q[9];
ry(-1.0424370323679728) q[10];
rz(-0.4222645835965651) q[10];
ry(1.0475457875822451) q[11];
rz(-0.4223745746001049) q[11];
ry(-1.0447944757436574) q[12];
rz(-0.4223878849755068) q[12];
ry(-1.038559465787194) q[13];
rz(2.7192082951493917) q[13];
ry(2.114672517832782) q[14];
rz(-0.42238758178783925) q[14];
ry(2.2048381119373706) q[15];
rz(2.719216545330837) q[15];
ry(-0.7699849219346171) q[16];
rz(-0.42236746656611457) q[16];
ry(0.5219329539020808) q[17];
rz(-0.42233024926247875) q[17];
ry(0.8159556182586005) q[18];
rz(2.719232573107265) q[18];
ry(2.8410968189969337) q[19];
rz(2.7192399188903478) q[19];