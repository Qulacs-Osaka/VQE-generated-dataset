OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.982726123926266) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-2.1028799116916574) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.008379690995799587) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.5225907915010808) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.8894247205038102) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.6914859869989952) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-2.5916150747593076) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(2.165802001591083) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.4413388984056721) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(2.087447089537688) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.12319721916674564) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.3364106694026626) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-1.575504089328511) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(1.6447906157296543) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.19621376913294333) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(1.5862875299907497) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-1.5757368263746694) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-1.4613707115777004) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.06596396716082449) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.12797622995562513) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.4312180738228016) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.2200244221509854) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.21668193655483015) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.07719027978818302) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.6610802157779206) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(2.1719642281186013) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.02066219583106487) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-0.28797258284487864) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.21516862443280496) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.06462894178900766) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-1.79739008686242) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(2.9519840398250694) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.5176137721584958) q[11];
cx q[10],q[11];
rz(-0.7447385553901246) q[0];
rz(0.7464469651016515) q[1];
rz(1.435111075246502) q[2];
rz(2.022486328831871) q[3];
rz(-0.4505188968415714) q[4];
rz(0.45764227090575915) q[5];
rz(-1.8528808707180835) q[6];
rz(1.102729002483111) q[7];
rz(0.12304320576441267) q[8];
rz(0.040108141189469565) q[9];
rz(-0.14068577284136777) q[10];
rz(-0.09481229091521927) q[11];
rx(-1.8246913842335504) q[0];
rx(0.5644787328711149) q[1];
rx(2.9986423640389503) q[2];
rx(0.36703843585067963) q[3];
rx(-3.040444793701716) q[4];
rx(-1.5858319532660325) q[5];
rx(-3.019769806291765) q[6];
rx(-1.3563841220670088) q[7];
rx(-2.4067933294628485) q[8];
rx(2.825926946057302) q[9];
rx(-1.4485329080076967) q[10];
rx(-2.013649588903049) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(1.8618906712738548) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-1.4687308852960286) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(1.286612205849229) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.04196820951518482) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.23337650999443726) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.11650230661672602) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-1.432049724543013) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.7153304378131705) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.17058727894217748) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(1.4949246097728304) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-1.4883238945948416) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.03521509619220373) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-1.6108061879478317) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(2.5169802499197456) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-1.4943908424063792) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(2.957013242239786) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(3.0199799382658656) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.08111884543212243) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.08471663035038236) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-2.229366760304783) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.0019349780391775097) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.014921374031535683) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.012229834333457437) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.030022089436891226) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-1.0269357799136565) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(2.3269101497813045) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.3127860660556838) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(1.6425351799855543) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-1.6217829292928525) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.11806985096949885) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-2.4451727955171143) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(2.9595549047327148) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.21721499203083663) q[11];
cx q[10],q[11];
rz(0.3269371708490767) q[0];
rz(-0.9940709008892259) q[1];
rz(1.0548342742816212) q[2];
rz(0.4603279967889844) q[3];
rz(0.16991650999116156) q[4];
rz(-0.27697971822268563) q[5];
rz(-1.823281172489255) q[6];
rz(0.9595320939692321) q[7];
rz(0.2359834353096165) q[8];
rz(0.16906232818518657) q[9];
rz(1.917586669387141) q[10];
rz(0.15283428652975864) q[11];
rx(-1.2323169186589957) q[0];
rx(0.7293896861870887) q[1];
rx(2.6532541644872945) q[2];
rx(-0.09135163383694676) q[3];
rx(-1.3375127376264984) q[4];
rx(-1.4521667290209743) q[5];
rx(1.6574649813712998) q[6];
rx(-0.33787190678554063) q[7];
rx(-2.1805094833986995) q[8];
rx(-1.5767217618114246) q[9];
rx(-1.6489004874585118) q[10];
rx(-0.5767682415854509) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-2.977256405056701) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-2.4266471381706736) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.34725848875976006) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-1.256964311672447) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.28750322651676335) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0680156687816862) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-3.115623662993235) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(3.083791030352818) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.42962003458730075) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(1.305151925355011) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-1.3074933645112081) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(1.7494065549051303) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-1.6531938073821) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(1.395574562666116) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.9185611606262581) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(1.7908183567083114) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-2.1152601720319244) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(2.186695590411237) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.7454448035557663) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.5893565068532123) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.5798007375471614) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-8.560212226599203e-05) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.015732347055609875) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.037296199710333024) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.024927885933200177) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.003851316964751572) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.004155860761640725) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(1.6506652653448513) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.17373373109313064) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-1.6893481286527372) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(2.9205501870991566) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-3.0196216165559515) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(1.6109866715943675) q[11];
cx q[10],q[11];
rz(-0.8242832099258544) q[0];
rz(0.8558423118945042) q[1];
rz(1.4344339235586234) q[2];
rz(0.6172321028530128) q[3];
rz(0.44628916439883776) q[4];
rz(-0.5017996123048583) q[5];
rz(2.395750327818118) q[6];
rz(-0.629024885270419) q[7];
rz(-0.09511088512396251) q[8];
rz(0.8646504869231371) q[9];
rz(0.7860329723486468) q[10];
rz(-0.40612105366880724) q[11];
rx(-2.161969571852522) q[0];
rx(-0.9158799360927679) q[1];
rx(3.077746622175089) q[2];
rx(0.2846417751445643) q[3];
rx(-2.7640199719163387) q[4];
rx(3.002262601329594) q[5];
rx(-0.3201712062309927) q[6];
rx(-2.3526594601337387) q[7];
rx(-1.8314815882934044) q[8];
rx(-0.553924654755283) q[9];
rx(-1.5065569840980964) q[10];
rx(3.0823201839424583) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(2.9887958626026863) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-2.9826399178707788) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.23209931967784103) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-2.206927555166632) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(1.0655816803710816) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.0215823969555543) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-1.583068898384856) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(2.2724241444366826) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0845813939712935) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(3.0894262071869965) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-1.7670685387620113) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.04440350391474073) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(2.9605623330691313) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(2.426658906472974) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.3873801961236657) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(1.568379825472245) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-1.8864700582728773) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(1.372731721118283) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.07063862204373716) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.9940809710352766) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.36162286489322687) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.03324873807985812) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.09767948878229002) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.011382491640833157) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.2513354250831204) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(1.6621903100559157) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.07765530833302674) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.27269455306614526) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.5113503581810586) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.0423333704506603) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-2.7238948746727787) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(3.139566518433791) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.032617576663489634) q[11];
cx q[10],q[11];
rz(0.4891037797439473) q[0];
rz(-0.17735488717521838) q[1];
rz(1.1656266148930439) q[2];
rz(-0.9190089768783799) q[3];
rz(-0.11606000922399623) q[4];
rz(0.17960231148658645) q[5];
rz(-0.411190817308627) q[6];
rz(-0.03355059966698147) q[7];
rz(1.4763231916984743) q[8];
rz(-0.14692387271483048) q[9];
rz(-1.7542618874516287) q[10];
rz(-0.23606135854039237) q[11];
rx(-1.7799199136372095) q[0];
rx(-2.0354536899825977) q[1];
rx(-2.7358105142434113) q[2];
rx(0.1685332802979004) q[3];
rx(-0.8173634185524831) q[4];
rx(1.8398513532677854) q[5];
rx(1.4706383602229034) q[6];
rx(0.8916836973830691) q[7];
rx(2.3717367131183296) q[8];
rx(-1.5385940085340855) q[9];
rx(0.34531120795074033) q[10];
rx(-2.6504225261737293) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(2.912427820257967) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-1.7622591141566322) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.05165777776902124) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-2.065525235803169) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(2.5880561054181124) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.08569966974516303) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-2.7921014812696643) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(1.2977436768995223) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.13048512611514007) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-3.1166633161693964) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-3.0699099734670323) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.030016462386904058) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-3.1202549773112116) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(2.955290138205702) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.09233243978496299) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(2.079078290312046) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-1.3133350779147888) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(1.21702019434954) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(1.9517973850888857) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-2.067376679189646) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.1436548058552727) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(2.714247175962777) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.7570998824326592) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.7627608031588862) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.575383228466202) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(-0.08670134367995734) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.05884838360346817) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(2.5949792561188016) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.3685186667554753) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(1.3400336672574942) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(2.964299184541624) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(3.1221467390324125) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.014799788085081547) q[11];
cx q[10],q[11];
rz(0.3422570548425574) q[0];
rz(-0.8140800479596508) q[1];
rz(0.08087391234996491) q[2];
rz(-0.026971818797082714) q[3];
rz(0.6833938043227006) q[4];
rz(-1.053291579062758) q[5];
rz(1.3283368577372219) q[6];
rz(-1.5496368341515567) q[7];
rz(-0.6066521798554406) q[8];
rz(-0.6702274578672499) q[9];
rz(-2.5019681152670956) q[10];
rz(1.5438206747353689) q[11];
rx(-1.9020234959550157) q[0];
rx(-0.18317917969588993) q[1];
rx(-3.0768567080918108) q[2];
rx(-1.0843679227230576) q[3];
rx(3.00099745524775) q[4];
rx(0.13582737550768106) q[5];
rx(-2.715037388809226) q[6];
rx(1.3055490677039245) q[7];
rx(-1.1121761287631222) q[8];
rx(1.9972114133750152) q[9];
rx(-2.6719326183830905) q[10];
rx(-1.4560697496235884) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(2.301208606458404) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-1.390608772496381) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.40063262580797815) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.1776290109512397) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(2.418680001365421) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.13584235845851686) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-2.338998292963708) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(2.8194154256189066) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.4608659601423524) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(2.978460104485563) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(3.141552230564528) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.07525280695840779) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-2.532537607557806) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(-2.754245020516317) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.5252558361093819) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(-0.08519661443954987) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.02334149796028293) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.03706375480123253) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.10590566280088162) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(0.11360374399620225) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.08166131449737293) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.014617176094565651) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.015209038580903009) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.06409201244198515) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.5555396538286194) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.2326014463358955) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.0025507922471784514) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.9088210463146771) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(0.8786472492760262) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(0.8755026458553551) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(2.901548053575348) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-2.915076779862179) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.10472788827631312) q[11];
cx q[10],q[11];
rz(-0.3880412258567393) q[0];
rz(0.3377489264068549) q[1];
rz(-1.144613575920026) q[2];
rz(-0.01592568694588193) q[3];
rz(0.6045277539814403) q[4];
rz(0.292460833940614) q[5];
rz(-0.5786601910767305) q[6];
rz(-0.022303408223672338) q[7];
rz(1.8283410960330533) q[8];
rz(-1.2803764886006608) q[9];
rz(-0.05636161105170443) q[10];
rz(1.5182359680533417) q[11];
rx(-1.2446157824076798) q[0];
rx(-1.0042303570570632) q[1];
rx(-1.4032622883521597) q[2];
rx(-0.3569454103789924) q[3];
rx(-2.222410536586787) q[4];
rx(-1.7562417171979696) q[5];
rx(2.930432276806275) q[6];
rx(0.2369497663250686) q[7];
rx(-1.9793248361654647) q[8];
rx(-2.9590376483493572) q[9];
rx(-0.016269540943399503) q[10];
rx(-0.019678217718934885) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(2.4881790767385503) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(-2.702265473401177) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(0.5616491331399647) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.9918767186483092) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-0.03305056235811566) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.03514592511393679) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-2.1766520788591115) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.0033707537655054674) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.25404836825337446) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(3.0663231347362987) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(3.097302639094688) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.022445761907019215) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-2.7544224138457545) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(2.541940498431637) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.4944975499450419) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.47115908022532793) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.46289357815240584) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.4244839115336203) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(0.2637841596711007) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.25738869831360717) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.23156300806185265) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.7179438482223968) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.7320615188067976) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.18884091555789784) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.5285961058333973) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.48552320523631387) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-0.3951346320320482) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(0.7181731649562217) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-0.7487363066559032) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-2.5673908155065237) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-2.5282367023787473) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(2.5295522700853317) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.4878822759182628) q[11];
cx q[10],q[11];
rz(-0.05012015173774247) q[0];
rz(0.19953697724596692) q[1];
rz(0.09233184807386603) q[2];
rz(0.0678545559506866) q[3];
rz(0.16697377710917552) q[4];
rz(-0.7900884138989904) q[5];
rz(-1.3394978325892752) q[6];
rz(-2.515808188416608) q[7];
rz(-0.4441200300095752) q[8];
rz(0.12225093423696246) q[9];
rz(-0.35035596432275645) q[10];
rz(-1.5326418000369884) q[11];
rx(-1.3110281096033585) q[0];
rx(-0.5975818486765565) q[1];
rx(1.4919288873235672) q[2];
rx(-1.5294839816510413) q[3];
rx(-2.0314804574597147) q[4];
rx(0.039426178465204575) q[5];
rx(3.0941648126309853) q[6];
rx(-0.026545570859491774) q[7];
rx(3.1382544978664657) q[8];
rx(3.138293013665079) q[9];
rx(-0.0031474216360184255) q[10];
rx(-3.1368522481952956) q[11];