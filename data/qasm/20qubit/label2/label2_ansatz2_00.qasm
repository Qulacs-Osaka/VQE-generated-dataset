OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.5707964137922588) q[0];
rz(-3.141592544954545) q[0];
ry(-1.5707961738057383) q[1];
rz(-1.2736278012809856) q[1];
ry(-1.5707962049469435) q[2];
rz(0.5384783984060632) q[2];
ry(-1.5707964766343427) q[3];
rz(1.2850404258557042) q[3];
ry(1.570796276572442) q[4];
rz(-3.1415900296559744) q[4];
ry(-1.570796332624691) q[5];
rz(0.007859225665026169) q[5];
ry(-1.5707962632053727) q[6];
rz(-3.1415918496721686) q[6];
ry(-1.5707963206883218) q[7];
rz(3.1415924891816918) q[7];
ry(1.5707962275836822) q[8];
rz(-2.6416609688026917) q[8];
ry(-1.5707966002736227) q[9];
rz(-2.2668881613044833) q[9];
ry(-1.5708043824943623) q[10];
rz(-1.5260621561746213) q[10];
ry(1.5707858941691168) q[11];
rz(-1.3564939052908846) q[11];
ry(-1.5708223415840132) q[12];
rz(3.1415909860901126) q[12];
ry(-1.5707661020280153) q[13];
rz(-2.729648326159542) q[13];
ry(-1.5939103998860595) q[14];
rz(3.1415922201213937) q[14];
ry(-1.6444525232962972) q[15];
rz(5.1127346977290974e-08) q[15];
ry(-0.09900665778927635) q[16];
rz(-3.141583268773837) q[16];
ry(-5.083178247516802e-06) q[17];
rz(-3.1141147750279305) q[17];
ry(3.1415926405709023) q[18];
rz(2.8430686088387156) q[18];
ry(-1.4096426130107787e-07) q[19];
rz(-2.719700710109286) q[19];
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
ry(2.684118482528485) q[0];
rz(-1.439890218933769) q[0];
ry(-2.859665393373234e-09) q[1];
rz(2.117894392189257) q[1];
ry(-2.2035420321201868e-07) q[2];
rz(1.0325465727653693) q[2];
ry(3.141592646115593) q[3];
rz(0.7613067801811457) q[3];
ry(-1.3661811121131393) q[4];
rz(2.474404665953062) q[4];
ry(0.0016246677891757228) q[5];
rz(-0.42269572787616566) q[5];
ry(1.0256411749369079) q[6];
rz(-1.6273911311510139) q[6];
ry(1.7944757479017142) q[7];
rz(1.2875568969817888) q[7];
ry(-3.1415924839616394) q[8];
rz(-1.070864815581439) q[8];
ry(3.141592628417537) q[9];
rz(2.4454996062850576) q[9];
ry(1.79905796926505e-08) q[10];
rz(-1.702044033719172) q[10];
ry(-3.141592617382584) q[11];
rz(-1.237953116390931) q[11];
ry(-3.0730045292826222) q[12];
rz(1.570806092462532) q[12];
ry(-3.1415914028944094) q[13];
rz(-1.158869104499396) q[13];
ry(1.5732456142832985) q[14];
rz(-1.576483264471955) q[14];
ry(1.5715920453408831) q[15];
rz(-1.5850002665200682) q[15];
ry(-1.5707999012293223) q[16];
rz(-0.8535044900692172) q[16];
ry(1.6207580007885527) q[17];
rz(-6.038786994793394e-05) q[17];
ry(0.049937877720196075) q[18];
rz(-9.270262718619193e-05) q[18];
ry(1.1316569620056782e-05) q[19];
rz(3.1221422153655753) q[19];
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
ry(5.337019053364145e-06) q[0];
rz(-1.312522301810788) q[0];
ry(1.3283377647262853e-06) q[1];
rz(1.5047865330954262) q[1];
ry(3.103702746526387) q[2];
rz(0.8290823398895579) q[2];
ry(-3.141589577307245) q[3];
rz(2.907888532796303) q[3];
ry(-3.1415924772498656) q[4];
rz(-0.2872044874028372) q[4];
ry(3.141592561073231) q[5];
rz(2.0589861731811894) q[5];
ry(-3.1415626074871525) q[6];
rz(-2.1786953952582335) q[6];
ry(-3.1415908431601904) q[7];
rz(0.975635669037246) q[7];
ry(0.6918711450955701) q[8];
rz(-1.0420005591478652) q[8];
ry(2.5626622572596904) q[9];
rz(-0.8688726560791827) q[9];
ry(-3.14159264841285) q[10];
rz(-2.938183989336563) q[10];
ry(-4.303333867028413e-08) q[11];
rz(-2.7840939594514764) q[11];
ry(-0.06697248725026306) q[12];
rz(-1.8926531560261184) q[12];
ry(-3.0577559699668813) q[13];
rz(1.9226963939803352) q[13];
ry(-3.1415722747657067) q[14];
rz(1.0266523336540039) q[14];
ry(-3.141571059264274) q[15];
rz(-1.0464954102428612) q[15];
ry(-8.026942634131728e-05) q[16];
rz(1.3920704262248729) q[16];
ry(-1.5708495329021455) q[17];
rz(-2.603025902555243) q[17];
ry(-1.5707957324390138) q[18];
rz(2.6026836607966906) q[18];
ry(1.570796356261452) q[19];
rz(-0.2314785557483274) q[19];
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
ry(2.541925087137598) q[0];
rz(1.4483194322934088) q[0];
ry(0.30542347331952646) q[1];
rz(-1.250856337637531) q[1];
ry(2.8191331783487072) q[2];
rz(-2.735363904686382) q[2];
ry(-0.8462170593392441) q[3];
rz(-2.215001076959103) q[3];
ry(-2.5261277686910666) q[4];
rz(-1.7044734984292627) q[4];
ry(-2.7884922036781443) q[5];
rz(0.48483124855566345) q[5];
ry(-0.4211545264259575) q[6];
rz(1.6332658524792352) q[6];
ry(0.7722766380316823) q[7];
rz(-1.7927097431789294) q[7];
ry(2.7032303421946797) q[8];
rz(1.6085061996892378) q[8];
ry(-0.3380410507935636) q[9];
rz(-2.6930607289169104) q[9];
ry(-2.2949707536922443) q[10];
rz(-1.8245736157050532) q[10];
ry(-2.6553846668278713) q[11];
rz(-1.5920096223412714) q[11];
ry(-0.7436304713530583) q[12];
rz(1.3625459605262407) q[12];
ry(-2.471284924003022) q[13];
rz(-1.7395246980366743) q[13];
ry(0.43077127726416187) q[14];
rz(-1.5224355508888625) q[14];
ry(2.7108590629750133) q[15];
rz(1.6192101454836054) q[15];
ry(2.710860627477605) q[16];
rz(1.6192776762647094) q[16];
ry(-2.710860594396783) q[17];
rz(-1.522314160618804) q[17];
ry(0.43054780807829396) q[18];
rz(-1.5219503525953906) q[18];
ry(1.9371510037099569) q[19];
rz(-2.103927304542327) q[19];