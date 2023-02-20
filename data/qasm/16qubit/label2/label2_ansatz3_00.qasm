OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.9886436733152104) q[0];
rz(1.4841710518386293) q[0];
ry(-0.12603378884171157) q[1];
rz(-1.6143812072614727) q[1];
ry(-3.058947206993795) q[2];
rz(-2.578055425566116) q[2];
ry(0.05598355282330498) q[3];
rz(-2.400928114167423) q[3];
ry(2.3518120711658828) q[4];
rz(-3.0057243460457945) q[4];
ry(1.5941724451114696) q[5];
rz(1.5926771631750274) q[5];
ry(-1.565932825492536) q[6];
rz(1.5673399855845007) q[6];
ry(3.1411462095796985) q[7];
rz(1.954813959405402) q[7];
ry(0.32203668386619905) q[8];
rz(-1.5687578049591995) q[8];
ry(-1.5709804926403352) q[9];
rz(1.5711493918766906) q[9];
ry(1.571197745417754) q[10];
rz(1.005095020687837) q[10];
ry(3.102019195481846) q[11];
rz(0.97838672472859) q[11];
ry(-0.10675233070075096) q[12];
rz(-2.996447443289721) q[12];
ry(-0.06407905452267393) q[13];
rz(-1.3350038277743677) q[13];
ry(-1.4501831348924954) q[14];
rz(-2.1279023503420595) q[14];
ry(1.5354593990731356) q[15];
rz(1.9736690661592562) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.734652646364131) q[0];
rz(-2.162341253202311) q[0];
ry(-1.6875942502446568) q[1];
rz(0.098253247280462) q[1];
ry(3.053452639782373) q[2];
rz(-0.8218279579635901) q[2];
ry(-0.4297032593791451) q[3];
rz(-1.357239002479234) q[3];
ry(0.04849941494827625) q[4];
rz(0.8900359781624658) q[4];
ry(2.232299908084589) q[5];
rz(2.2229935343211658) q[5];
ry(2.8814496210496157) q[6];
rz(-1.5753560036163334) q[6];
ry(0.3705625849582339) q[7];
rz(-0.18084134736957264) q[7];
ry(3.6965673073652283e-06) q[8];
rz(0.6305300800182613) q[8];
ry(-2.571466238002891) q[9];
rz(1.2840506974325576) q[9];
ry(0.014746352069852975) q[10];
rz(-1.005768319053916) q[10];
ry(-1.0426775528281675) q[11];
rz(-1.629540571003761) q[11];
ry(-3.091651361345508) q[12];
rz(-3.007001760930849) q[12];
ry(0.26594923150745253) q[13];
rz(-1.2745473641188552) q[13];
ry(-3.085480747802001) q[14];
rz(-0.03865861766155724) q[14];
ry(0.5258003339207997) q[15];
rz(2.7183022458166524) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.7168057485287243) q[0];
rz(0.8463839548684042) q[0];
ry(-3.0932559249141685) q[1];
rz(-1.5444347608894469) q[1];
ry(3.0672763647522725) q[2];
rz(-0.3347214335243782) q[2];
ry(3.1385947488035786) q[3];
rz(1.7998037297049647) q[3];
ry(0.7016489272240616) q[4];
rz(-2.221837114421174) q[4];
ry(3.1168821754747866) q[5];
rz(-2.4897462023280093) q[5];
ry(1.575241231755107) q[6];
rz(-2.400250188329675) q[6];
ry(0.00039111506173661326) q[7];
rz(-2.0068101004365113) q[7];
ry(-0.31583032298358693) q[8];
rz(2.1521218938313034) q[8];
ry(-3.1415148449143793) q[9];
rz(-0.287083895703469) q[9];
ry(1.5690591224940809) q[10];
rz(2.7802069267449645) q[10];
ry(-3.1129455033964883) q[11];
rz(0.9225778617798784) q[11];
ry(-0.11066401817344751) q[12];
rz(0.2936422786038202) q[12];
ry(0.0002780627199872821) q[13];
rz(-2.602773003076619) q[13];
ry(-2.305185996796003) q[14];
rz(-2.070472735094457) q[14];
ry(0.7622256900933319) q[15];
rz(-1.530527713776905) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.004112545876382433) q[0];
rz(1.829210232540215) q[0];
ry(3.120255532002645) q[1];
rz(-2.0392454509843185) q[1];
ry(0.011414190367015742) q[2];
rz(3.0077992591375913) q[2];
ry(-2.7140012788345866) q[3];
rz(-0.46730001564083573) q[3];
ry(3.1354475503727466) q[4];
rz(-0.5020965079956277) q[4];
ry(1.5566040031020556) q[5];
rz(2.6554860232003756) q[5];
ry(-3.1405767633145216) q[6];
rz(1.7526149306332215) q[6];
ry(3.139849788013356) q[7];
rz(-1.1901862475067928) q[7];
ry(0.0013700721835485652) q[8];
rz(2.5106509566113133) q[8];
ry(1.5692557706463068) q[9];
rz(-0.6911281710326006) q[9];
ry(3.1367017609302676) q[10];
rz(0.525577179219022) q[10];
ry(0.0027952811559481816) q[11];
rz(1.476455365462228) q[11];
ry(3.13996055575585) q[12];
rz(-1.0031145372972854) q[12];
ry(2.88060930313154) q[13];
rz(-1.2629466280000676) q[13];
ry(3.1295272490543877) q[14];
rz(0.16421727984015977) q[14];
ry(-1.3707762977638076) q[15];
rz(-2.5405158950284026) q[15];