OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.5713398197778385) q[0];
rz(2.626165220256634) q[0];
ry(1.570795070119576) q[1];
rz(-1.0302260893019426) q[1];
ry(-3.140723674754028) q[2];
rz(2.512714124442222) q[2];
ry(3.141278092098691) q[3];
rz(-1.119774106595943) q[3];
ry(1.5727462441570945) q[4];
rz(1.5655263054402493) q[4];
ry(-1.4712040333492) q[5];
rz(1.336086898229638) q[5];
ry(-0.00033254343620825466) q[6];
rz(-2.5141652024304686) q[6];
ry(-7.060185168938915e-06) q[7];
rz(1.168444674006346) q[7];
ry(-3.041540042913256) q[8];
rz(-2.8068692737919116) q[8];
ry(0.03235638028276089) q[9];
rz(-1.2154733422408883) q[9];
ry(-0.2619094084382563) q[10];
rz(2.666250386816092) q[10];
ry(-2.8905909341576965) q[11];
rz(-1.7974547712356455) q[11];
ry(-3.120474577137238) q[12];
rz(-3.1154218224305237) q[12];
ry(1.4642006569893578) q[13];
rz(-2.9401690329074985) q[13];
ry(3.049398702925957) q[14];
rz(2.9154857761327158) q[14];
ry(3.1413077010105166) q[15];
rz(0.7803589235056262) q[15];
ry(3.1414329091575826) q[16];
rz(-1.6840648166327126) q[16];
ry(1.0679880269367132) q[17];
rz(-0.010528841792223633) q[17];
ry(-1.3986219714730534) q[18];
rz(1.7483921740253612) q[18];
ry(0.07681295891203024) q[19];
rz(-2.7177378707373006) q[19];
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
ry(-2.4565540251631606) q[0];
rz(0.6096713362283995) q[0];
ry(1.2472206016368574) q[1];
rz(0.9632526930286094) q[1];
ry(1.57990288317309) q[2];
rz(-3.0765201725359352) q[2];
ry(1.5700441589549285) q[3];
rz(-2.192202417365413) q[3];
ry(-1.6471154269187993) q[4];
rz(-0.006304097861435486) q[4];
ry(-1.555042094474988) q[5];
rz(1.6112401130355325) q[5];
ry(3.1413961377264794) q[6];
rz(3.0543574843429946) q[6];
ry(-3.753395308393771e-05) q[7];
rz(2.909136327713386) q[7];
ry(1.5185186381784899) q[8];
rz(1.5820526405651387) q[8];
ry(3.057022275500466) q[9];
rz(0.2429866286509055) q[9];
ry(0.4461391835809017) q[10];
rz(1.0794677468126854) q[10];
ry(-0.0015498530462272342) q[11];
rz(2.216703647562946) q[11];
ry(-0.2731008767914753) q[12];
rz(0.7486236170814055) q[12];
ry(3.1304251260378515) q[13];
rz(-1.363545564044574) q[13];
ry(1.611812211145482) q[14];
rz(-1.6427231341865427) q[14];
ry(0.04506091505251586) q[15];
rz(-0.9362783705660647) q[15];
ry(-2.772186354097324) q[16];
rz(3.078652816628471) q[16];
ry(-1.010327456933418) q[17];
rz(-0.9439290343753267) q[17];
ry(-0.7985791219570444) q[18];
rz(-2.786158251385101) q[18];
ry(-3.080156097238233) q[19];
rz(3.0896129447301117) q[19];
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
ry(-1.6051903786951836) q[0];
rz(-1.0843808484657247) q[0];
ry(1.5830312600259826) q[1];
rz(-0.5169288940148071) q[1];
ry(-0.00025013145493321784) q[2];
rz(-2.4814930975043143) q[2];
ry(3.139052247601587) q[3];
rz(-2.823993806950714) q[3];
ry(-1.6184243079895348) q[4];
rz(0.04616826003880832) q[4];
ry(-1.5673132071086289) q[5];
rz(0.003642469239222401) q[5];
ry(0.00011130691207320552) q[6];
rz(1.4322585350342596) q[6];
ry(-3.1415861015978352) q[7];
rz(-0.8701043536643372) q[7];
ry(0.9886459577319751) q[8];
rz(-1.4405025379396685) q[8];
ry(-2.2201639350277182) q[9];
rz(1.646658133020671) q[9];
ry(0.04157401034354357) q[10];
rz(2.4942420376308734) q[10];
ry(-2.129542004540893) q[11];
rz(-1.540669221718626) q[11];
ry(-3.140868404673323) q[12];
rz(-2.463798669179042) q[12];
ry(2.0898457427492936) q[13];
rz(-1.5309218905828865) q[13];
ry(-3.140763711162453) q[14];
rz(-1.6185589621067935) q[14];
ry(1.187928207999197) q[15];
rz(1.1871360555742054) q[15];
ry(-3.083646330565863) q[16];
rz(3.104541961665251) q[16];
ry(3.0884664583861428) q[17];
rz(-1.984378487483868) q[17];
ry(0.015075835326494591) q[18];
rz(-0.3824163744476783) q[18];
ry(-0.1592790024162371) q[19];
rz(2.2761399962546065) q[19];
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
ry(1.548709702091666) q[0];
rz(1.9879813307520475) q[0];
ry(1.5260238333567824) q[1];
rz(1.070163365602931) q[1];
ry(-3.141531674659206) q[2];
rz(-1.4056141937018687) q[2];
ry(1.103855522887235e-05) q[3];
rz(-1.9878155929009722) q[3];
ry(1.5694022453454195) q[4];
rz(1.5680949300608409) q[4];
ry(-1.5796949195921717) q[5];
rz(-1.5730793754366694) q[5];
ry(-0.028977426649895218) q[6];
rz(0.23249750337712613) q[6];
ry(3.13531243933765) q[7];
rz(-2.449305248921514) q[7];
ry(-1.6398400907407529) q[8];
rz(-0.5755305855938605) q[8];
ry(-3.135971033385283) q[9];
rz(-0.007046558542921879) q[9];
ry(-2.5211531981824957) q[10];
rz(2.5964234351188367) q[10];
ry(-3.1414187569890175) q[11];
rz(-1.6464325703799174) q[11];
ry(-0.21526349639134457) q[12];
rz(1.5105725745369902) q[12];
ry(3.1339953690202527) q[13];
rz(3.0730988000242068) q[13];
ry(1.584459953073063) q[14];
rz(-1.2403143174336284) q[14];
ry(3.095175196904648) q[15];
rz(-1.9461126321572277) q[15];
ry(-0.3675678839160631) q[16];
rz(3.029742968205387) q[16];
ry(-3.1415483964102338) q[17];
rz(-2.543283127148468) q[17];
ry(-0.00691116412403936) q[18];
rz(-0.19854882622597003) q[18];
ry(1.6073642194887743) q[19];
rz(-1.5980250014052968) q[19];
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
ry(-1.10114699363265) q[0];
rz(-2.9053155991772486) q[0];
ry(1.1906779063767825) q[1];
rz(1.9858785518575601) q[1];
ry(-3.0645878346786546) q[2];
rz(-1.2204686241404565) q[2];
ry(1.6475535476913155) q[3];
rz(-1.1393939748221902) q[3];
ry(-1.5727211954720701) q[4];
rz(-0.9466495517315394) q[4];
ry(1.5733085826491253) q[5];
rz(2.353569693558678) q[5];
ry(3.141376012122197) q[6];
rz(2.4275079205106644) q[6];
ry(-3.1412837295710343) q[7];
rz(-0.2681032972925959) q[7];
ry(-3.1131355457895933) q[8];
rz(1.044319604054908) q[8];
ry(3.0770809035995033) q[9];
rz(3.1096559596640887) q[9];
ry(3.1401034119242714) q[10];
rz(2.531930844600457) q[10];
ry(1.1706555695491423) q[11];
rz(-3.0332606684591124) q[11];
ry(3.1391887036869583) q[12];
rz(-2.789148406169562) q[12];
ry(1.6290368722899782) q[13];
rz(0.1481754451162208) q[13];
ry(3.1354395460209665) q[14];
rz(0.48092635070254897) q[14];
ry(3.096361298009879) q[15];
rz(-1.411467227939876) q[15];
ry(-0.0011880559938588445) q[16];
rz(-1.277927854431746) q[16];
ry(0.015211499166191793) q[17];
rz(0.770558164052313) q[17];
ry(-1.5570785630046018) q[18];
rz(-0.18488617287830333) q[18];
ry(-1.591260449093993) q[19];
rz(-1.6852966927832915) q[19];