OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.3569161043209421) q[0];
rz(-0.8873007386688149) q[0];
ry(-1.376817946379717) q[1];
rz(-0.20511135464754737) q[1];
ry(1.7874671658885433) q[2];
rz(-1.7420770651238688) q[2];
ry(-2.145760466647178) q[3];
rz(-1.0426940346449411) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.3478163739360813) q[0];
rz(-2.322637515247589) q[0];
ry(-2.265293951682982) q[1];
rz(3.0966944286898386) q[1];
ry(0.01446803314640821) q[2];
rz(-2.316406621848227) q[2];
ry(0.25535756729966064) q[3];
rz(0.1968585255276638) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.053557922354898) q[0];
rz(-0.24517347066710823) q[0];
ry(-1.0531462659349717) q[1];
rz(1.8326666974692931) q[1];
ry(1.8546581502368786) q[2];
rz(-1.7078279392353082) q[2];
ry(2.881788948894399) q[3];
rz(-0.5914097153422525) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.3358291892915377) q[0];
rz(2.0257616544735506) q[0];
ry(1.5806263814273376) q[1];
rz(1.3224672919815823) q[1];
ry(1.133054205432063) q[2];
rz(0.6110488748479987) q[2];
ry(-0.5860159284377174) q[3];
rz(0.639447467894614) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.064816540813889) q[0];
rz(-0.7134998618714937) q[0];
ry(2.503679264307578) q[1];
rz(0.057345639439627914) q[1];
ry(0.10065276208984651) q[2];
rz(2.7318979282513642) q[2];
ry(2.4609735732438835) q[3];
rz(0.6340479645361743) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.530858979035803) q[0];
rz(-2.286263761444823) q[0];
ry(-2.565526631057534) q[1];
rz(1.6700445573432756) q[1];
ry(1.5904753855494622) q[2];
rz(-1.245871401800795) q[2];
ry(-0.6087042288578504) q[3];
rz(-1.6778105666881284) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.4075241378654857) q[0];
rz(2.955985311279919) q[0];
ry(-1.5405359947666502) q[1];
rz(3.12156661901836) q[1];
ry(-3.037589096286409) q[2];
rz(-1.878300564987649) q[2];
ry(2.017847918164592) q[3];
rz(0.7208014827596783) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.7748302931904574) q[0];
rz(2.4884336865514944) q[0];
ry(2.60717093976299) q[1];
rz(0.655312405936294) q[1];
ry(2.7135531294942745) q[2];
rz(2.058179986040451) q[2];
ry(-3.118237694152188) q[3];
rz(-0.08254690023135856) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.7184276175033724) q[0];
rz(0.4850509786462311) q[0];
ry(-0.9667640101874201) q[1];
rz(-2.095876999465262) q[1];
ry(-1.7396330225334653) q[2];
rz(1.7580519844519449) q[2];
ry(2.3898848764781464) q[3];
rz(1.7719110275614778) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.011199765205604645) q[0];
rz(1.898545758918515) q[0];
ry(0.06250418155625859) q[1];
rz(-1.2676306308135639) q[1];
ry(-1.6975386308646634) q[2];
rz(-1.789354577635125) q[2];
ry(3.0636230389793164) q[3];
rz(-2.7833843537129104) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.3522424716641428) q[0];
rz(1.4168784054321468) q[0];
ry(3.0834226350661984) q[1];
rz(1.7290661348484244) q[1];
ry(-0.9802010997768962) q[2];
rz(-0.1300617205245569) q[2];
ry(2.126590774392172) q[3];
rz(0.5353944180258497) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.7716731190158717) q[0];
rz(-2.986328133217628) q[0];
ry(1.6613136278241531) q[1];
rz(-0.639154017971654) q[1];
ry(-0.3440619999844246) q[2];
rz(2.6712788189346957) q[2];
ry(1.3136813580334188) q[3];
rz(-0.5328728248890959) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.1075921061301717) q[0];
rz(2.028466059606263) q[0];
ry(-0.8396691260710707) q[1];
rz(-2.487744337658351) q[1];
ry(-0.2268086197906087) q[2];
rz(0.5792415752298907) q[2];
ry(1.0803657809848732) q[3];
rz(2.7939191711363276) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.38437670541926) q[0];
rz(-1.5899583467838683) q[0];
ry(-0.1007247792313164) q[1];
rz(-1.9756018656057648) q[1];
ry(1.1701240832506066) q[2];
rz(-0.9051816548525382) q[2];
ry(-1.395741629339164) q[3];
rz(-2.676088534830495) q[3];