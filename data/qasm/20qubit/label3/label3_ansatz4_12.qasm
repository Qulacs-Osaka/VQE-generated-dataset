OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(0.015785752556875643) q[0];
rz(-1.2225152370890873) q[0];
ry(1.555998816075807) q[1];
rz(3.0513915301827286) q[1];
ry(-1.6163341995489298) q[2];
rz(3.1212352040438254) q[2];
ry(-1.5614941816774774) q[3];
rz(-0.27336232708231023) q[3];
ry(-2.9107431188839916) q[4];
rz(-2.8472220714262377) q[4];
ry(-3.139930528434578) q[5];
rz(1.1439055864824006) q[5];
ry(1.729291133903616) q[6];
rz(3.0596897076037624) q[6];
ry(-1.572180123131778) q[7];
rz(-1.2650217390372354) q[7];
ry(-9.094043101764981e-05) q[8];
rz(2.484348589683913) q[8];
ry(3.1414804615911924) q[9];
rz(-2.8946958983798807) q[9];
ry(1.5707157394366307) q[10];
rz(-1.3386848353617553) q[10];
ry(-1.57074018764499) q[11];
rz(-3.1214175218044713) q[11];
ry(0.0006744779901100014) q[12];
rz(0.3744666822783751) q[12];
ry(-0.2910335844135137) q[13];
rz(2.778203698686682) q[13];
ry(3.1375770469231887) q[14];
rz(2.287567442873684) q[14];
ry(0.5387897055964928) q[15];
rz(-1.7654666180033411) q[15];
ry(-1.6052847000373305) q[16];
rz(-3.077573763146921) q[16];
ry(1.5850679649784296) q[17];
rz(0.8056237105619267) q[17];
ry(-0.4520861654585433) q[18];
rz(0.5156214002278547) q[18];
ry(-2.891453355613393) q[19];
rz(-2.5312916302893096) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.9858219750883417) q[0];
rz(-0.685196642621947) q[0];
ry(1.4895720829582535) q[1];
rz(1.177082601198757) q[1];
ry(-1.5013723140715727) q[2];
rz(1.5873741192502864) q[2];
ry(0.12494114222716772) q[3];
rz(1.848956457085952) q[3];
ry(-1.5704662533380755) q[4];
rz(-7.260794034458229e-06) q[4];
ry(1.5703745597097623) q[5];
rz(-0.00022328349401856684) q[5];
ry(-3.1386556311996823) q[6];
rz(-1.6188881763474443) q[6];
ry(-3.13314256326676) q[7];
rz(-1.2571464803514947) q[7];
ry(-1.579701454582394) q[8];
rz(-3.141503414360777) q[8];
ry(-2.390583843839595) q[9];
rz(-2.9754667885043737) q[9];
ry(-0.019643165718572265) q[10];
rz(1.4220888300649757) q[10];
ry(-1.5656746719167138) q[11];
rz(1.3666844065715484) q[11];
ry(1.6513126252686927) q[12];
rz(2.127636254009445) q[12];
ry(3.1382502570275372) q[13];
rz(2.517670378180538) q[13];
ry(1.577919683589414) q[14];
rz(-0.0005786401219982267) q[14];
ry(1.6297180898361274) q[15];
rz(-3.1406880428378248) q[15];
ry(1.630275005398045) q[16];
rz(2.031989661399357) q[16];
ry(-0.18734278706160842) q[17];
rz(1.6392676489590023) q[17];
ry(2.7585105947322863) q[18];
rz(0.37955372652098696) q[18];
ry(-2.6515983741304194) q[19];
rz(0.335614019767108) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5210932401043964) q[0];
rz(-3.0785101317251127) q[0];
ry(2.695130317951242) q[1];
rz(2.744718236796531) q[1];
ry(-1.3965746124220664) q[2];
rz(3.055319655762951) q[2];
ry(1.7106188259737385) q[3];
rz(-3.125261079765347) q[3];
ry(1.5707070704847004) q[4];
rz(-1.5707645039991713) q[4];
ry(-1.570727646493661) q[5];
rz(2.039609499836937) q[5];
ry(0.38529432622204896) q[6];
rz(-1.663873221045228) q[6];
ry(1.5484901767049062) q[7];
rz(2.7529091547985503) q[7];
ry(1.6027710312622145) q[8];
rz(-1.5712791549628742) q[8];
ry(3.068168966962897) q[9];
rz(2.079743191845707) q[9];
ry(5.6433902932084834e-05) q[10];
rz(-1.745223062693058) q[10];
ry(3.141575858134784) q[11];
rz(2.8557470605862387) q[11];
ry(0.011225499502249825) q[12];
rz(-0.5836224299652173) q[12];
ry(-1.5462278705057146) q[13];
rz(0.031121287066430803) q[13];
ry(-1.5727587135100345) q[14];
rz(-0.3385780772025295) q[14];
ry(1.5719942630056383) q[15];
rz(1.6141001208330399) q[15];
ry(3.1403453143244797) q[16];
rz(0.5841983993938661) q[16];
ry(0.008941467843960993) q[17];
rz(2.7902587415417743) q[17];
ry(1.930148340909729) q[18];
rz(2.309909533256967) q[18];
ry(-0.8128990632970509) q[19];
rz(-0.1469648562085782) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.697629837185737) q[0];
rz(0.029978295593674712) q[0];
ry(-1.752776242938384) q[1];
rz(1.8104717633057645) q[1];
ry(1.5327292337300067) q[2];
rz(0.17261813081080565) q[2];
ry(1.5741398168900946) q[3];
rz(0.7542732376406135) q[3];
ry(-1.5932789345908174) q[4];
rz(0.01331371837712947) q[4];
ry(1.6091016609562652) q[5];
rz(-3.1025045426222286) q[5];
ry(0.000948828596018636) q[6];
rz(0.059618298367429885) q[6];
ry(-1.5816311949797472) q[7];
rz(-1.0338263727781707) q[7];
ry(1.594851553437188) q[8];
rz(0.014161545983600199) q[8];
ry(-3.1240663807879776) q[9];
rz(2.392528239378243) q[9];
ry(-3.140694890762769) q[10];
rz(-0.4500940640520455) q[10];
ry(0.001061807618248664) q[11];
rz(2.5259941376742905) q[11];
ry(1.5849369966029754) q[12];
rz(-0.002697002987551045) q[12];
ry(1.5712031892411153) q[13];
rz(1.5681123462340525) q[13];
ry(-0.0052878189237977585) q[14];
rz(0.3290828332501082) q[14];
ry(-3.103333023297029) q[15];
rz(1.6121455995313756) q[15];
ry(-0.6831153073791951) q[16];
rz(0.9974778677022567) q[16];
ry(0.7986906324867888) q[17];
rz(1.908674342390638) q[17];
ry(2.1789057653167427) q[18];
rz(-0.17886823473621885) q[18];
ry(-0.8839695049278357) q[19];
rz(-2.5364794433141573) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.8813861331866493) q[0];
rz(-2.997083566389586) q[0];
ry(-1.7156178252490013) q[1];
rz(-2.329435085635742) q[1];
ry(-0.0647793508216922) q[2];
rz(-2.7335208958732284) q[2];
ry(3.1343299777898532) q[3];
rz(-2.7973397496911625) q[3];
ry(2.756908157873555) q[4];
rz(-0.9515912747164945) q[4];
ry(-0.004552275120083493) q[5];
rz(1.5437430290165641) q[5];
ry(1.5345803916309644) q[6];
rz(0.808838215943803) q[6];
ry(0.01026341793991745) q[7];
rz(1.716041796467255) q[7];
ry(1.389118694508733) q[8];
rz(3.071611107198614) q[8];
ry(3.1213507688325985) q[9];
rz(-1.1913596316836916) q[9];
ry(-3.1406355665338843) q[10];
rz(1.2364909046692114) q[10];
ry(-0.0008638291853474737) q[11];
rz(2.275312187002921) q[11];
ry(1.5677831281470278) q[12];
rz(-1.777047019251075) q[12];
ry(-3.134982205379229) q[13];
rz(3.1406220557161806) q[13];
ry(0.5308057628362112) q[14];
rz(-0.03402172755996511) q[14];
ry(-1.0388611957623946) q[15];
rz(1.39211671950686) q[15];
ry(-3.1373611301682836) q[16];
rz(-0.7544036439805346) q[16];
ry(3.1390931281748493) q[17];
rz(-1.5587021224675484) q[17];
ry(0.8380634026873949) q[18];
rz(0.5005372512367083) q[18];
ry(0.8904901378037327) q[19];
rz(0.3795279131878236) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-0.7560293988410318) q[0];
rz(-1.787228001803847) q[0];
ry(1.6610209002805247) q[1];
rz(-2.9925608804766775) q[1];
ry(3.1181627055415833) q[2];
rz(0.3152408948615457) q[2];
ry(0.18792166088802723) q[3];
rz(2.7753995830471) q[3];
ry(3.127007878770924) q[4];
rz(1.9929097315676358) q[4];
ry(-1.5319893575795218) q[5];
rz(1.545817451345857) q[5];
ry(3.1408345311268) q[6];
rz(-2.3307569427851402) q[6];
ry(0.013719500911627769) q[7];
rz(-0.03753134099040433) q[7];
ry(-3.1285830537240007) q[8];
rz(-2.838650493313235) q[8];
ry(-2.699780968333043) q[9];
rz(1.6002526721178019) q[9];
ry(0.04497270292524435) q[10];
rz(3.1129090698617383) q[10];
ry(3.0935410016017686) q[11];
rz(0.004570871642763981) q[11];
ry(-1.5856165120406462) q[12];
rz(3.1105797801519395) q[12];
ry(1.5481767413479472) q[13];
rz(0.0006408580783881645) q[13];
ry(1.5547635339560735) q[14];
rz(-1.5284833942556795) q[14];
ry(-0.0394869625416961) q[15];
rz(1.9181049795553662) q[15];
ry(-0.9987134070079956) q[16];
rz(2.760408381057365) q[16];
ry(1.2007966955340286) q[17];
rz(1.790894780627174) q[17];
ry(-2.384661979602148) q[18];
rz(2.3296780568514803) q[18];
ry(2.8174319433247565) q[19];
rz(-0.4402575919770957) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.2296436444801975) q[0];
rz(-1.525574577586441) q[0];
ry(2.1762345776028518) q[1];
rz(-1.5879208023088829) q[1];
ry(0.08232474069850021) q[2];
rz(-0.4426226917196343) q[2];
ry(0.3196174708392361) q[3];
rz(-0.057350548957442286) q[3];
ry(3.1413984005255946) q[4];
rz(-1.7540200361482832) q[4];
ry(0.21407146918583397) q[5];
rz(-1.8248676799839334) q[5];
ry(-0.4519760409055211) q[6];
rz(0.01910198877777238) q[6];
ry(3.136112327925147) q[7];
rz(2.0791733705876587) q[7];
ry(3.105902142336022) q[8];
rz(2.2681346781027294) q[8];
ry(1.5808663945824302) q[9];
rz(-2.3837752687343103) q[9];
ry(-1.5708271862074363) q[10];
rz(-3.1011164069758714) q[10];
ry(1.5712486301525288) q[11];
rz(-0.042487968316574615) q[11];
ry(1.534147952084234) q[12];
rz(0.357845447466861) q[12];
ry(1.5738775870825235) q[13];
rz(2.7324606176700907) q[13];
ry(-1.5382579030379901) q[14];
rz(-1.5726601030691052) q[14];
ry(1.242987718618489) q[15];
rz(0.0020207564890100116) q[15];
ry(-3.1399137501172243) q[16];
rz(-2.456894337176969) q[16];
ry(0.003260031652286018) q[17];
rz(-1.4468907059509253) q[17];
ry(-1.7638751940761468) q[18];
rz(3.113594946769317) q[18];
ry(1.5807529191792922) q[19];
rz(1.5204494806049795) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.9312053849268118) q[0];
rz(-2.6303684351120173) q[0];
ry(-2.2197145705516235) q[1];
rz(3.136185383173678) q[1];
ry(-1.8041898318804108) q[2];
rz(-1.88363538690557) q[2];
ry(2.9720446565469607) q[3];
rz(1.733293699242556) q[3];
ry(1.6008234060070157) q[4];
rz(0.06351127293980507) q[4];
ry(3.0977761126611667) q[5];
rz(-0.2912126741396719) q[5];
ry(-2.5902085655517046) q[6];
rz(2.2477532652009566) q[6];
ry(-0.00770351677210202) q[7];
rz(1.2076416868905906) q[7];
ry(0.03343914646278368) q[8];
rz(-1.8858404219601614) q[8];
ry(-0.1856712270407037) q[9];
rz(-1.494253575043929) q[9];
ry(1.5713121100581544) q[10];
rz(2.297436342729324) q[10];
ry(-1.5706494540861153) q[11];
rz(2.1332880812700497) q[11];
ry(0.1250592696129571) q[12];
rz(-2.2876839627352976) q[12];
ry(-2.798112447808128) q[13];
rz(2.7038583374513956) q[13];
ry(1.5721438972466824) q[14];
rz(-0.44791819111415615) q[14];
ry(-1.243111826184925) q[15];
rz(-1.6353298152776956) q[15];
ry(3.087437281704035) q[16];
rz(-2.561093529706071) q[16];
ry(3.069351380222585) q[17];
rz(1.2847753676819746) q[17];
ry(2.0460530613649435) q[18];
rz(-3.0708926734333093) q[18];
ry(-2.8376011076038634) q[19];
rz(1.9724225827453674) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.1343836811317227) q[0];
rz(-0.7229778030194868) q[0];
ry(1.5676315568287889) q[1];
rz(-3.139590346744492) q[1];
ry(0.7013208640145177) q[2];
rz(0.38624488679548463) q[2];
ry(1.6636278504147914) q[3];
rz(-0.15274054692525954) q[3];
ry(-0.07389250544674031) q[4];
rz(-1.6328312519946806) q[4];
ry(-1.565821567005023) q[5];
rz(-2.9155304110847378) q[5];
ry(-3.117892293741593) q[6];
rz(-1.8621910092187282) q[6];
ry(-0.0023703872672511395) q[7];
rz(-2.630876484799799) q[7];
ry(1.6218135536622897) q[8];
rz(-0.9445393467513874) q[8];
ry(2.815840313926002) q[9];
rz(1.370564734625062) q[9];
ry(3.137482690450963) q[10];
rz(-2.5206256659915605) q[10];
ry(3.13681006222878) q[11];
rz(2.2554036653440175) q[11];
ry(3.137445896359887) q[12];
rz(2.6017169062208176) q[12];
ry(0.10086430550338843) q[13];
rz(1.6185925267139283) q[13];
ry(0.00019991461475487552) q[14];
rz(-1.1219730998103936) q[14];
ry(3.094707277084939) q[15];
rz(1.3975626567368866) q[15];
ry(-3.1400252365728627) q[16];
rz(-1.2118345880868293) q[16];
ry(1.572726219343894) q[17];
rz(-0.029604526134389395) q[17];
ry(-1.7820351697121626) q[18];
rz(-2.661325084525655) q[18];
ry(3.1168286474022238) q[19];
rz(-1.5417641653469056) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.024072914244373855) q[0];
rz(2.8337538260773942) q[0];
ry(-1.4874054546263622) q[1];
rz(0.12582443412757963) q[1];
ry(-1.5707349716381178) q[2];
rz(-3.1330965427260495) q[2];
ry(-3.1331387289251373) q[3];
rz(1.4118174972914943) q[3];
ry(-0.25457415707737546) q[4];
rz(-1.9734121600382721) q[4];
ry(0.0012498539272840503) q[5];
rz(1.8838392356500056) q[5];
ry(0.00029719224303015324) q[6];
rz(-1.5112353877152844) q[6];
ry(1.5738900489180754) q[7];
rz(-1.157121293253745) q[7];
ry(-0.5549716574297081) q[8];
rz(2.5946756622089593) q[8];
ry(3.1258542177311686) q[9];
rz(-1.0098814887098246) q[9];
ry(-3.1394564284633724) q[10];
rz(-1.6229771982471857) q[10];
ry(-0.0004838437427385311) q[11];
rz(-0.17290902885493242) q[11];
ry(1.5850673176374162) q[12];
rz(-1.6933123192515014) q[12];
ry(1.6867787426612153) q[13];
rz(-0.5662145811090732) q[13];
ry(3.0973536216938116) q[14];
rz(1.5708194673574365) q[14];
ry(-3.14156686977378) q[15];
rz(2.8638964590735014) q[15];
ry(1.5707300920127487) q[16];
rz(-3.116829948455935) q[16];
ry(-3.1346132474638257) q[17];
rz(-3.015005308132089) q[17];
ry(0.7039046868007617) q[18];
rz(0.2977771803285221) q[18];
ry(1.5151186858960795) q[19];
rz(-0.130115700870704) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5693307128108378) q[0];
rz(-0.38977742936600096) q[0];
ry(-1.5701198093641198) q[1];
rz(1.375747874075897) q[1];
ry(-1.435649349091457) q[2];
rz(0.098783550463164) q[2];
ry(-0.06526127356250822) q[3];
rz(-1.5633384035797395) q[3];
ry(-3.1408289255647386) q[4];
rz(-2.126968767813117) q[4];
ry(-0.0032755014833622726) q[5];
rz(2.487563760913644) q[5];
ry(-3.137007855377652) q[6];
rz(-3.090205862661044) q[6];
ry(-0.002021567558298596) q[7];
rz(2.7347629884219398) q[7];
ry(1.5949198517007535) q[8];
rz(0.004257149543499964) q[8];
ry(1.5713084124378627) q[9];
rz(1.5677490286681977) q[9];
ry(-3.021617556321548) q[10];
rz(-3.0885328594082053) q[10];
ry(0.11962049613150717) q[11];
rz(-3.0914161398542674) q[11];
ry(1.490701969668728) q[12];
rz(1.6165121517457819) q[12];
ry(0.0005485882751932181) q[13];
rz(-2.270255158243904) q[13];
ry(1.570406024018151) q[14];
rz(-1.3719064505965082) q[14];
ry(1.6011075584520857) q[15];
rz(1.8925828249537897) q[15];
ry(-0.07485480832820764) q[16];
rz(0.5265986158168143) q[16];
ry(3.1411170888842155) q[17];
rz(-2.151592554878998) q[17];
ry(-2.8563077973309676) q[18];
rz(-0.034109729949344574) q[18];
ry(-3.041455314197676) q[19];
rz(-2.9488369423580427) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.010685050222228524) q[0];
rz(1.960555938375971) q[0];
ry(-3.1396335454264426) q[1];
rz(-1.6013750286266744) q[1];
ry(-3.1391273894393317) q[2];
rz(1.9688855754900496) q[2];
ry(-1.5687508027718973) q[3];
rz(1.5712376690581271) q[3];
ry(-0.03686096151101074) q[4];
rz(-3.0467619528622385) q[4];
ry(1.6014479368629768) q[5];
rz(-0.7857185441556539) q[5];
ry(3.140384145790172) q[6];
rz(2.193313877816771) q[6];
ry(1.5551463013154956) q[7];
rz(-0.4824540929802125) q[7];
ry(1.571831868380932) q[8];
rz(-3.13143176021384) q[8];
ry(-2.7587806819622864) q[9];
rz(-1.5690687775916103) q[9];
ry(1.5709508334634452) q[10];
rz(-3.078584312589213) q[10];
ry(1.5709940219022187) q[11];
rz(2.1391333735346736) q[11];
ry(3.071140451358633) q[12];
rz(-1.4427366369566812) q[12];
ry(1.5742872791370894) q[13];
rz(1.9543120136532233) q[13];
ry(1.347787512046204) q[14];
rz(-0.6186428311778877) q[14];
ry(-0.9206384027679242) q[15];
rz(1.1158818777519939) q[15];
ry(-3.1415452190408253) q[16];
rz(-0.8436977142217972) q[16];
ry(-3.140306040788092) q[17];
rz(1.2510672004991876) q[17];
ry(2.244544390722306) q[18];
rz(0.8731624321346844) q[18];
ry(-0.05920539704598255) q[19];
rz(0.18865681221026342) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5698081267161141) q[0];
rz(-1.5823993697105823) q[0];
ry(-0.0016197082377902195) q[1];
rz(-0.17609041901009773) q[1];
ry(-0.0007207828198838584) q[2];
rz(1.2733158510702183) q[2];
ry(0.287305302825574) q[3];
rz(-3.1408363367954384) q[3];
ry(3.1221933142193947) q[4];
rz(-1.8169584476524847) q[4];
ry(-3.141563849991791) q[5];
rz(2.357353419312592) q[5];
ry(3.1412648314989062) q[6];
rz(2.90523345079878) q[6];
ry(0.0004882798439238556) q[7];
rz(-2.7381977249893175) q[7];
ry(-1.5698443617277924) q[8];
rz(1.5732761647518936) q[8];
ry(1.535690645244637) q[9];
rz(-3.002870769712082) q[9];
ry(1.4017035653857668) q[10];
rz(1.6008745002357605) q[10];
ry(3.0414754181033095) q[11];
rz(2.1921319692616548) q[11];
ry(-0.02676095528111391) q[12];
rz(-0.7110997398598506) q[12];
ry(-3.132002580509806) q[13];
rz(0.3621628002357819) q[13];
ry(-0.0006922511571216816) q[14];
rz(3.0197416726497885) q[14];
ry(3.136778167960178) q[15];
rz(-1.5317191484429955) q[15];
ry(-3.1390753109174208) q[16];
rz(0.17560678070605107) q[16];
ry(-3.139800593729063) q[17];
rz(1.987313311073878) q[17];
ry(-1.7744903737414797) q[18];
rz(-2.850105610280653) q[18];
ry(3.0566669770825348) q[19];
rz(1.9553565384152165) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.5678968011395344) q[0];
rz(3.0143915609421392) q[0];
ry(1.5757454061847687) q[1];
rz(-0.0003397680456053303) q[1];
ry(1.5736157737223353) q[2];
rz(-0.13691669464581654) q[2];
ry(1.5704024382456543) q[3];
rz(-1.5484675405034596) q[3];
ry(-3.135978972803245) q[4];
rz(2.343993213465235) q[4];
ry(1.5935001637137738) q[5];
rz(-0.8059320863937911) q[5];
ry(3.0751979990260936) q[6];
rz(0.22531593844637868) q[6];
ry(3.11720732437881) q[7];
rz(-1.5021566009785028) q[7];
ry(-1.5734856182720038) q[8];
rz(1.5704193728759153) q[8];
ry(-0.0006009860823521507) q[9];
rz(1.4293379962167247) q[9];
ry(-1.5773029452402363) q[10];
rz(0.0006971408524091595) q[10];
ry(1.5707862955157765) q[11];
rz(0.0009758471625948031) q[11];
ry(-0.0003291551839082761) q[12];
rz(-1.492294029130651) q[12];
ry(9.515680814021959e-05) q[13];
rz(1.2804083703630242) q[13];
ry(2.8437192409553917) q[14];
rz(-0.7178042075104368) q[14];
ry(2.4128073681144353) q[15];
rz(0.38681477792501434) q[15];
ry(1.5710669612874189) q[16];
rz(-1.571187041270357) q[16];
ry(-1.5709959460099787) q[17];
rz(0.00024660935903273324) q[17];
ry(-1.271869011323602) q[18];
rz(-1.5645178753203905) q[18];
ry(-0.1403339894647626) q[19];
rz(3.073296837047426) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.1034125127496144) q[0];
rz(-0.11628810658069888) q[0];
ry(1.5856048340460207) q[1];
rz(-3.0959663458175295) q[1];
ry(0.006134033641849097) q[2];
rz(-1.4152217724123233) q[2];
ry(-1.5743628042103053) q[3];
rz(-1.5201841076700053) q[3];
ry(-0.000177556375666299) q[4];
rz(-2.7798709355934914) q[4];
ry(0.00667588380964812) q[5];
rz(2.4535221612204983) q[5];
ry(3.1413134138321) q[6];
rz(0.1843205650186709) q[6];
ry(0.00041246362692106444) q[7];
rz(1.529628465191133) q[7];
ry(1.5712603233419467) q[8];
rz(-1.7555171118555721) q[8];
ry(-1.5712544363807472) q[9];
rz(3.0871593578884937) q[9];
ry(1.5707510147802208) q[10];
rz(3.0112591562414046) q[10];
ry(1.570633122899193) q[11];
rz(0.313229718861228) q[11];
ry(-7.566204100584173e-05) q[12];
rz(-2.618578630835732) q[12];
ry(-0.0012468762746719758) q[13];
rz(1.881195957360343) q[13];
ry(1.5682957751311848) q[14];
rz(-0.03902863380108368) q[14];
ry(-1.5706657058875253) q[15];
rz(3.0604317807641834) q[15];
ry(1.5716311722108058) q[16];
rz(2.832300466982974) q[16];
ry(-1.5704052940567106) q[17];
rz(-0.0002771057986699077) q[17];
ry(3.139801328127232) q[18];
rz(2.9926350985923578) q[18];
ry(3.1279334399586824) q[19];
rz(2.1803094138431716) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5695201667464795) q[0];
rz(3.0079185925268086) q[0];
ry(-1.4787701503924688) q[1];
rz(-1.691298957298106) q[1];
ry(0.026140753007153267) q[2];
rz(-1.722135971723504) q[2];
ry(-1.4797326228813539) q[3];
rz(-1.6933500742214322) q[3];
ry(3.1351111067103927) q[4];
rz(2.7604402099180114) q[4];
ry(-1.629776633990077) q[5];
rz(2.984093787607679) q[5];
ry(-1.5848662054250848) q[6];
rz(3.010485768877206) q[6];
ry(-1.6365864342286887) q[7];
rz(2.960421240601788) q[7];
ry(-0.14614986864012547) q[8];
rz(1.6192734478591406) q[8];
ry(-1.7788448663687584) q[9];
rz(1.4657176097812972) q[9];
ry(0.2060221309328695) q[10];
rz(-1.5776553105376547) q[10];
ry(-2.926848540456932) q[11];
rz(-1.4186143169753587) q[11];
ry(-1.5343198206308912) q[12];
rz(2.988109743365678) q[12];
ry(1.5730927551584362) q[13];
rz(-1.7247882589122403) q[13];
ry(-1.58377958868832) q[14];
rz(-1.7284133013975593) q[14];
ry(-0.003008583604036552) q[15];
rz(3.070093666350519) q[15];
ry(-3.1054771024156125) q[16];
rz(2.6725699251132555) q[16];
ry(1.5679123488769973) q[17];
rz(1.375193071575083) q[17];
ry(3.107428350637294) q[18];
rz(-0.34070304215955366) q[18];
ry(0.0036828051134998323) q[19];
rz(-2.5267273586191297) q[19];