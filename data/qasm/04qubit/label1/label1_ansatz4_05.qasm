OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.2148069358739042) q[0];
rz(-1.1801849294299824) q[0];
ry(-1.6003321126297985) q[1];
rz(1.0833788021060222) q[1];
ry(-2.63615354240398) q[2];
rz(3.08954475515715) q[2];
ry(1.2638401873124019) q[3];
rz(-1.6542094207209113) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.717619363636233) q[0];
rz(0.4274716630519029) q[0];
ry(-1.6346650675099619) q[1];
rz(1.4669701733131755) q[1];
ry(1.0536616030025208) q[2];
rz(0.7491788212791269) q[2];
ry(2.656350744484269) q[3];
rz(-1.6024545189098598) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.3660987526910113) q[0];
rz(-2.9673608707287884) q[0];
ry(-2.826131873062262) q[1];
rz(-0.5606727714056651) q[1];
ry(-2.5404114862497753) q[2];
rz(-2.9306168872800233) q[2];
ry(1.3014437014256526) q[3];
rz(-0.9253578265144905) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.157993161530585) q[0];
rz(-1.5539734963011296) q[0];
ry(2.0961933667356445) q[1];
rz(1.333115963583471) q[1];
ry(2.210205397343666) q[2];
rz(0.821963853198298) q[2];
ry(-0.8919300126674489) q[3];
rz(0.8120837091802326) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.6019399938801113) q[0];
rz(1.824485061914995) q[0];
ry(1.6334584830187904) q[1];
rz(1.6524570376671812) q[1];
ry(0.16747184415686434) q[2];
rz(-0.07066654393894112) q[2];
ry(-2.266064603321102) q[3];
rz(-0.42775227141429706) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.9006638216349256) q[0];
rz(-1.9708238331521029) q[0];
ry(-1.9671028678532745) q[1];
rz(-3.0954469597934304) q[1];
ry(-0.35906149182917524) q[2];
rz(3.076781323202045) q[2];
ry(0.43494732014359444) q[3];
rz(2.4557074889971977) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.0136264528210397) q[0];
rz(-0.41833805269268604) q[0];
ry(-2.397632510576142) q[1];
rz(-1.4061520669660679) q[1];
ry(-1.0619518523100426) q[2];
rz(2.4502884155563702) q[2];
ry(-0.9791309423643249) q[3];
rz(-1.355099480378212) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(3.0915748649003305) q[0];
rz(-1.7880620194921324) q[0];
ry(-1.7994319319008927) q[1];
rz(0.027448268715313508) q[1];
ry(1.1971068578941957) q[2];
rz(0.5264124404823719) q[2];
ry(-0.7753668001854656) q[3];
rz(1.3947039036207398) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.0059438371186638) q[0];
rz(-2.3735609380468534) q[0];
ry(-1.9354300858058704) q[1];
rz(-2.262161922966477) q[1];
ry(2.569863448100005) q[2];
rz(1.679459395244089) q[2];
ry(-2.103810617452826) q[3];
rz(0.58893492251085) q[3];