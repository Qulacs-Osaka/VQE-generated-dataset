OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.5707960511119055) q[0];
rz(-1.5109454622803693) q[0];
ry(1.5707966673141236) q[1];
rz(-1.739018060762258) q[1];
ry(1.5707884117541777) q[2];
rz(-1.699921362785517e-06) q[2];
ry(-1.5708000172246113) q[3];
rz(5.456131084713434e-07) q[3];
ry(-1.5711338466000047) q[4];
rz(3.177667833753617e-06) q[4];
ry(1.5707974805864087) q[5];
rz(-4.210553826755614e-05) q[5];
ry(0.0007631399580940723) q[6];
rz(-1.5361685437930799) q[6];
ry(0.005973466051039544) q[7];
rz(-0.3759499056632026) q[7];
ry(-5.7031839122423385e-05) q[8];
rz(-2.16649298524733) q[8];
ry(-4.5271500954270676e-07) q[9];
rz(-2.373075684719417) q[9];
ry(-3.1415911305711943) q[10];
rz(-2.8826804812715134) q[10];
ry(3.1415895020427307) q[11];
rz(1.5513617374093114) q[11];
ry(-8.27219542748789e-08) q[12];
rz(0.18450292910817015) q[12];
ry(-0.0001524903899134955) q[13];
rz(2.3148503828985207) q[13];
ry(-3.1415925001053897) q[14];
rz(0.20840848250719812) q[14];
ry(-4.8297230036760104e-08) q[15];
rz(0.02871046621417861) q[15];
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
ry(3.1381703271609154) q[0];
rz(-0.5004934377427641) q[0];
ry(-3.141581161178505) q[1];
rz(2.9684550968306582) q[1];
ry(-1.2044276554026085) q[2];
rz(1.5643045598532934) q[2];
ry(0.5544561444768622) q[3];
rz(2.451890651185083) q[3];
ry(-1.5744165507886763) q[4];
rz(1.3518204485774232) q[4];
ry(-2.9569216060099284) q[5];
rz(2.2546751965822374) q[5];
ry(3.0771851493142752) q[6];
rz(0.1265422159442) q[6];
ry(-3.1415132994676847) q[7];
rz(-1.332814704243222) q[7];
ry(-1.570796816735885) q[8];
rz(-2.7893685017628713) q[8];
ry(1.5707959428039509) q[9];
rz(-2.169563745607304) q[9];
ry(-1.570795054699053) q[10];
rz(-2.6403311389292765e-06) q[10];
ry(-1.5707957446888123) q[11];
rz(1.1815638606550347) q[11];
ry(-1.5047784083363505) q[12];
rz(0.8092392304820892) q[12];
ry(-1.5707811577680149) q[13];
rz(3.1415613023599276) q[13];
ry(0.000116349079539657) q[14];
rz(1.819476903548143) q[14];
ry(1.0611046911934399e-05) q[15];
rz(1.8325362148034579) q[15];
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
ry(-3.14159214129317) q[0];
rz(0.7968632368621065) q[0];
ry(-0.011140097035343466) q[1];
rz(0.7495170773780018) q[1];
ry(3.13045741149815) q[2];
rz(1.566506245387763) q[2];
ry(1.72721913941597e-07) q[3];
rz(0.6189825485975007) q[3];
ry(7.252225043608007e-06) q[4];
rz(1.0113637163150575) q[4];
ry(-3.141591992295843) q[5];
rz(0.9316339189208209) q[5];
ry(1.2385534320813463e-05) q[6];
rz(-1.6977290102561031) q[6];
ry(2.0553357720792887e-07) q[7];
rz(-0.5571854847563431) q[7];
ry(-3.141592474069951) q[8];
rz(2.2979589773187534) q[8];
ry(3.1415924082682705) q[9];
rz(-0.5343986302242026) q[9];
ry(0.9818975800243015) q[10];
rz(-1.5707912835811486) q[10];
ry(-1.4183265127111502e-07) q[11];
rz(0.3891163791503036) q[11];
ry(0.09101968642564362) q[12];
rz(2.3301904073553215) q[12];
ry(0.6516604245801049) q[13];
rz(-3.1415696402683007) q[13];
ry(-1.570892263260483) q[14];
rz(3.138020610518268) q[14];
ry(1.5708037543883444) q[15];
rz(0.0002594972729345291) q[15];
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
ry(3.5784090872681418e-06) q[0];
rz(1.7773397395589587) q[0];
ry(4.6467983239040216e-08) q[1];
rz(-0.7446174374701399) q[1];
ry(3.125222713518104) q[2];
rz(1.5956301769389467) q[2];
ry(-5.46225086504327e-05) q[3];
rz(-2.4521365828153856) q[3];
ry(-3.141586668073696) q[4];
rz(-2.3493625304161503) q[4];
ry(-3.075136699587766) q[5];
rz(-2.8153365691621826) q[5];
ry(1.5708047502408071) q[6];
rz(-0.3831827696931959) q[6];
ry(-1.5707967347949787) q[7];
rz(0.0691730405736335) q[7];
ry(7.363973555072967e-06) q[8];
rz(1.3606687840674745) q[8];
ry(-0.00011777734771367677) q[9];
rz(-0.055553985348791635) q[9];
ry(0.306494995523936) q[10];
rz(1.570510321365648) q[10];
ry(-1.1915302425184207) q[11];
rz(2.6363788385213303) q[11];
ry(1.5706808472316682) q[12];
rz(-1.570768999520624) q[12];
ry(1.570795515849941) q[13];
rz(1.4981598493085653) q[13];
ry(0.05864161047006815) q[14];
rz(-1.5361578827426794) q[14];
ry(-1.5121588598858542) q[15];
rz(-1.0951532063417264) q[15];
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
ry(0.00875424289027208) q[0];
rz(-3.1345484854494323) q[0];
ry(-0.20763891775261278) q[1];
rz(-3.141575374674343) q[1];
ry(-3.1396181466067614) q[2];
rz(-1.5481685141479646) q[2];
ry(-3.1415917547444625) q[3];
rz(0.6764043089700436) q[3];
ry(-2.94489702146973) q[4];
rz(-3.0389415893526714) q[4];
ry(3.141376679349547) q[5];
rz(-0.9318432890367266) q[5];
ry(0.00018954880293176046) q[6];
rz(2.9141789446712516) q[6];
ry(-3.141347659352754) q[7];
rz(-1.50167926746864) q[7];
ry(3.1415799647567604) q[8];
rz(-0.9563759417722784) q[8];
ry(3.1413342249667138) q[9];
rz(-1.2633561637181314) q[9];
ry(1.5707828036786644) q[10];
rz(1.210139728912878) q[10];
ry(3.141592323213759) q[11];
rz(-2.065155279040715) q[11];
ry(1.5707973526655374) q[12];
rz(-2.7608007983207448) q[12];
ry(-1.5707963612590512) q[13];
rz(3.1414668420685463) q[13];
ry(-3.1415595879360314) q[14];
rz(-1.9633384277318522) q[14];
ry(3.2397345304246983e-06) q[15];
rz(1.8694373034042056) q[15];
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
ry(1.5707956656005404) q[0];
rz(-0.12159135665559008) q[0];
ry(1.5707935052239916) q[1];
rz(-0.34275448759266514) q[1];
ry(0.543639595341725) q[2];
rz(-3.0944361585396822) q[2];
ry(1.0069122988825) q[3];
rz(-0.030844977331422996) q[3];
ry(3.14159230320217) q[4];
rz(-2.992076580588107) q[4];
ry(3.141584344269648) q[5];
rz(-1.0050603278255725) q[5];
ry(-6.756040424105426e-05) q[6];
rz(-2.530247025583253) q[6];
ry(-1.570754810166023) q[7];
rz(-1.5715151821495903) q[7];
ry(3.1415642654530145) q[8];
rz(0.4496392771213423) q[8];
ry(1.366371613482591e-06) q[9];
rz(-1.8402888469262244) q[9];
ry(3.368007662358884e-05) q[10];
rz(-1.1310968897228841) q[10];
ry(2.7624853125907123e-05) q[11];
rz(-0.01089810742318136) q[11];
ry(-0.0003290247289680081) q[12];
rz(-1.8954615881306403) q[12];
ry(1.5710986772403646) q[13];
rz(-0.05364223510195402) q[13];
ry(2.153495176713204e-06) q[14];
rz(-2.73778809205606) q[14];
ry(-1.7108854262204434e-06) q[15];
rz(2.389131782883823) q[15];
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
ry(-1.5707938124032585) q[0];
rz(-2.43764030878827) q[0];
ry(-1.5703908094070957) q[1];
rz(-1.7853457190823394) q[1];
ry(-1.5766389551236535) q[2];
rz(-0.1232725020829117) q[2];
ry(-1.570796613795543) q[3];
rz(3.141241756073525) q[3];
ry(-1.4471592542118512) q[4];
rz(3.1358278546338667) q[4];
ry(1.002963536400614e-05) q[5];
rz(-0.0047998310477908035) q[5];
ry(-1.5707163102408355) q[6];
rz(1.5709055358215471) q[6];
ry(-1.570876071255281) q[7];
rz(1.5706891426590284) q[7];
ry(-1.5708965681410951) q[8];
rz(0.28874849636437694) q[8];
ry(-1.6113374254830872) q[9];
rz(-2.5182286352123313) q[9];
ry(-2.4575561844558336) q[10];
rz(1.6321069979821479) q[10];
ry(1.7997658938960288) q[11];
rz(1.5624092454141298) q[11];
ry(-2.0196876879288502e-06) q[12];
rz(-2.8627988320523072) q[12];
ry(-3.1415917855145414) q[13];
rz(1.975825675260479) q[13];
ry(-3.1415079964297865) q[14];
rz(-1.9548362013082023) q[14];
ry(8.475992643522097e-05) q[15];
rz(-2.829997885687528) q[15];
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
ry(4.0658504595453836e-05) q[0];
rz(-0.2589065624313989) q[0];
ry(5.124193703390362e-06) q[1];
rz(2.073749585693788) q[1];
ry(1.5707964205103897) q[2];
rz(1.4814469917467932e-07) q[2];
ry(-1.5707953601091695) q[3];
rz(-3.1350515236753016) q[3];
ry(-1.570799732028302) q[4];
rz(-2.0009712107693645e-05) q[4];
ry(1.5707965096609289) q[5];
rz(1.5707742576538875) q[5];
ry(1.570648124218158) q[6];
rz(-1.7737421660338155) q[6];
ry(1.5709460131719517) q[7];
rz(-3.133208667343016) q[7];
ry(-1.5707953488289477) q[8];
rz(3.1415385197990053) q[8];
ry(1.5707993206044826) q[9];
rz(0.02528809916076367) q[9];
ry(-1.5705532427879152) q[10];
rz(1.5488192038822923) q[10];
ry(3.1413490445442425) q[11];
rz(1.5624092506302822) q[11];
ry(-1.878852084846727e-05) q[12];
rz(0.22921767752579572) q[12];
ry(3.141573800416559) q[13];
rz(0.2357203726685997) q[13];
ry(-2.0162928940782747e-07) q[14];
rz(0.39629201647890167) q[14];
ry(-4.340919070636615e-09) q[15];
rz(0.8075379164146868) q[15];
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
ry(3.1415917090909575) q[0];
rz(-1.1167657322594415) q[0];
ry(3.1415923081161368) q[1];
rz(1.8613630221224549) q[1];
ry(1.5706879671452596) q[2];
rz(1.1801792798182607) q[2];
ry(0.00010819921008510186) q[3];
rz(-1.5787109990826718) q[3];
ry(-1.5707964819077846) q[4];
rz(1.8205700694128906) q[4];
ry(1.570795568042219) q[5];
rz(3.1392209122574166) q[5];
ry(-3.141592619860109) q[6];
rz(-0.9869044183813802) q[6];
ry(-4.754623352809517e-07) q[7];
rz(2.341239401781892) q[7];
ry(-1.568631669513567) q[8];
rz(-2.8818806497103244) q[8];
ry(-0.0021629199979953384) q[9];
rz(-2.0968548894564645) q[9];
ry(-3.393675892038317e-05) q[10];
rz(-2.3295449297953716) q[10];
ry(-1.5707630464251645) q[11];
rz(-1.571596531929489) q[11];
ry(3.141543409623525) q[12];
rz(1.5191216302219106) q[12];
ry(-3.141559349009029) q[13];
rz(2.2347479652312736) q[13];
ry(-5.923461628178896e-06) q[14];
rz(-1.2145464559316768) q[14];
ry(3.1415867443843717) q[15];
rz(-2.043187955752717) q[15];
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
ry(1.565051414781375) q[0];
rz(1.3129740067734827) q[0];
ry(1.5617867924167714) q[1];
rz(0.2576123864355271) q[1];
ry(-3.1378545910457887) q[2];
rz(-0.32936574641994554) q[2];
ry(-1.5673447204277462) q[3];
rz(0.060839454191534706) q[3];
ry(3.1391641797643506) q[4];
rz(-2.9392871573390233) q[4];
ry(-1.5701647818076887) q[5];
rz(-1.6181057889434172) q[5];
ry(-1.6357019701665052) q[6];
rz(2.732484319317092) q[6];
ry(-1.6354019381432554) q[7];
rz(2.6025478826002724) q[7];
ry(-0.0001454339138346091) q[8];
rz(0.4785302794923237) q[8];
ry(-3.141470561021017) q[9];
rz(0.33471959184747474) q[9];
ry(3.140973248670602) q[10];
rz(3.076296096650793) q[10];
ry(-1.5707900630448652) q[11];
rz(-2.2733948971365603) q[11];
ry(-1.750418487663821) q[12];
rz(-2.896686178775966) q[12];
ry(1.8114917747126604) q[13];
rz(2.953539755604579) q[13];
ry(0.002070884298699838) q[14];
rz(-0.6883068754201845) q[14];
ry(-3.139765063435834) q[15];
rz(2.700144332622927) q[15];
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
ry(-2.592215280714172) q[0];
rz(-0.42752698025511) q[0];
ry(0.5493365546347836) q[1];
rz(-0.4038213857900487) q[1];
ry(2.584659406755698) q[2];
rz(-0.619827194858062) q[2];
ry(-0.5572449772285566) q[3];
rz(0.6075237346378605) q[3];
ry(0.7320180308549675) q[4];
rz(2.6860354109591165) q[4];
ry(2.406554945227561) q[5];
rz(0.4664096191038267) q[5];
ry(2.3445513026658705e-05) q[6];
rz(-2.6667387607916897) q[6];
ry(-1.89420379790306e-05) q[7];
rz(0.4728278604130573) q[7];
ry(1.576481776226424) q[8];
rz(-2.2204028632710067) q[8];
ry(-1.5656881440111912) q[9];
rz(0.6080547660286066) q[9];
ry(1.5578176187064736) q[10];
rz(1.539044119589622) q[10];
ry(-1.5559725269154017) q[11];
rz(-0.528958613053389) q[11];
ry(1.5067164451132289) q[12];
rz(-0.0008961627912998377) q[12];
ry(1.5067571727910196) q[13];
rz(-3.14073624833794) q[13];
ry(2.2299064133260824) q[14];
rz(1.0388768632284247) q[14];
ry(0.7780646128761148) q[15];
rz(2.485401797778616) q[15];
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
ry(-0.008880437223988102) q[0];
rz(-0.7229244834188009) q[0];
ry(-0.008880417793062813) q[1];
rz(0.987021615701658) q[1];
ry(-0.00362473903871498) q[2];
rz(-3.094772519488891) q[2];
ry(0.003591143542475166) q[3];
rz(0.17287639655515896) q[3];
ry(-0.002054680175017337) q[4];
rz(0.5140470062145154) q[4];
ry(0.0020723978326371295) q[5];
rz(2.6171130753855176) q[5];
ry(-1.571407089530771) q[6];
rz(2.508844408755509) q[6];
ry(-1.570284233558617) q[7];
rz(2.503791061490336) q[7];
ry(-0.00026483325575465955) q[8];
rz(0.6972059667836303) q[8];
ry(0.00018617108127472193) q[9];
rz(1.0049262426681966) q[9];
ry(-0.0005468260024212589) q[10];
rz(-0.05892519249381308) q[10];
ry(-3.1415490115659233) q[11];
rz(-1.189721631179309) q[11];
ry(-1.5708861214162306) q[12];
rz(2.402233090133601) q[12];
ry(1.5710463930020495) q[13];
rz(2.136974326465147) q[13];
ry(-3.1403335870154745) q[14];
rz(1.5448556508769564) q[14];
ry(3.1409576657398666) q[15];
rz(-1.5860895484977346) q[15];
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
ry(3.1415540213995894) q[0];
rz(2.042167345410931) q[0];
ry(3.141487602435267) q[1];
rz(-2.5074967509033685) q[1];
ry(-0.0018448257952459145) q[2];
rz(-2.519229283063981) q[2];
ry(-3.1398136629119024) q[3];
rz(0.8294053536904523) q[3];
ry(-0.15544409876695386) q[4];
rz(-0.020038572553928973) q[4];
ry(-2.983297930409474) q[5];
rz(3.121861858144502) q[5];
ry(-1.5609406331942985) q[6];
rz(1.6083640395579888) q[6];
ry(1.5609660879601819) q[7];
rz(-1.5332367867410657) q[7];
ry(0.14405254727966277) q[8];
rz(-3.14059490702204) q[8];
ry(-0.14679545661694338) q[9];
rz(0.006469047490202939) q[9];
ry(2.9801549371076828) q[10];
rz(-0.042839120646872786) q[10];
ry(-0.02270414088693098) q[11];
rz(-0.8601360401376098) q[11];
ry(-1.5764188413565945) q[12];
rz(-1.514640729280564) q[12];
ry(-1.5749638116506641) q[13];
rz(-1.512904161068812) q[13];
ry(0.15928113880957978) q[14];
rz(-3.1164498467357067) q[14];
ry(-3.1280038882848675) q[15];
rz(2.808309832704764) q[15];