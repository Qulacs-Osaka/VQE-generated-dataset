OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.31759563033637583) q[0];
rz(3.051777248266486) q[0];
ry(0.7239059370509933) q[1];
rz(-0.5920329665627454) q[1];
ry(-0.0079041322135458) q[2];
rz(-0.8465460007519559) q[2];
ry(-1.9778900711953413) q[3];
rz(2.868142507560262) q[3];
ry(-3.1415815097607536) q[4];
rz(1.6782553501375987) q[4];
ry(1.8415878591519421) q[5];
rz(-0.0736444966894867) q[5];
ry(1.5553493050639666) q[6];
rz(2.234568001788624) q[6];
ry(-3.141579880051073) q[7];
rz(1.6347557886456998) q[7];
ry(1.593654268372644) q[8];
rz(-1.467747698421789) q[8];
ry(3.0831929431125484) q[9];
rz(-1.2542231534161383) q[9];
ry(1.9750652766096772) q[10];
rz(1.4217545280548178) q[10];
ry(-0.259673542470952) q[11];
rz(-2.94209145801506) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.22269701753875995) q[0];
rz(-2.4232749616176914) q[0];
ry(2.754742788244888) q[1];
rz(2.5789188440179243) q[1];
ry(-0.001856551110213998) q[2];
rz(2.3403609666206524) q[2];
ry(1.4279944830848068) q[3];
rz(-2.7556173286363213) q[3];
ry(-0.00020629108232661508) q[4];
rz(1.3728899018389116) q[4];
ry(-1.6261891306452645) q[5];
rz(0.45271585321202545) q[5];
ry(3.0481843072368955) q[6];
rz(-1.5299255342566647) q[6];
ry(3.410327615394881e-05) q[7];
rz(-0.8919551979051094) q[7];
ry(1.074374559409839) q[8];
rz(-2.256598990359951) q[8];
ry(-3.1411024030737633) q[9];
rz(1.7396236413475217) q[9];
ry(-2.0988194635369886) q[10];
rz(1.6626672078589446) q[10];
ry(-2.417946901466899) q[11];
rz(2.2506527069049334) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.10820535124019641) q[0];
rz(-0.6340785104939579) q[0];
ry(-1.5793126191021527) q[1];
rz(-0.37636542110991306) q[1];
ry(0.866037375582203) q[2];
rz(-1.7767535284864657) q[2];
ry(0.5988046925634329) q[3];
rz(-1.8958346296184418) q[3];
ry(1.5707980266450075) q[4];
rz(2.455656381818648) q[4];
ry(-1.7248260677114202) q[5];
rz(-0.8380933038398887) q[5];
ry(0.46486558801861033) q[6];
rz(-0.15837730242852732) q[6];
ry(-1.5707639962982245) q[7];
rz(0.5208062129287283) q[7];
ry(2.510853445993977) q[8];
rz(2.4240874429207366) q[8];
ry(-0.01270337840594582) q[9];
rz(0.03363624039624824) q[9];
ry(-1.2207086866631227) q[10];
rz(-1.4538129329509362) q[10];
ry(-0.6922219586574149) q[11];
rz(1.1418069273152807) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.3200292909146787) q[0];
rz(-2.035097292594659) q[0];
ry(-3.132453703626549) q[1];
rz(2.6032183103905013) q[1];
ry(0.003965712670492463) q[2];
rz(-1.3670655047591167) q[2];
ry(-1.5707965053503963) q[3];
rz(0.4415034252448673) q[3];
ry(2.550138473328492) q[4];
rz(-2.634937572363833) q[4];
ry(-1.947936321511854) q[5];
rz(-1.4279427808738623) q[5];
ry(2.619822254513363) q[6];
rz(2.791089842699773) q[6];
ry(-3.120611564037832) q[7];
rz(0.45157978294338824) q[7];
ry(-1.570805025025475) q[8];
rz(0.09247039876768604) q[8];
ry(8.506280905842161e-05) q[9];
rz(0.632627594331041) q[9];
ry(-1.2520127783113009) q[10];
rz(-0.06069644367375471) q[10];
ry(-1.5933182619410884) q[11];
rz(-1.4408773640392534) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.1119751164480203) q[0];
rz(2.28262844911679) q[0];
ry(-2.463907982289025) q[1];
rz(-0.6174070881209021) q[1];
ry(1.5708018243240947) q[2];
rz(1.5194488362245204) q[2];
ry(-3.0968183546463752) q[3];
rz(2.215216060298185) q[3];
ry(-3.100789097763694) q[4];
rz(-2.649564213813453) q[4];
ry(-0.04907611885395182) q[5];
rz(-2.8821791753389423) q[5];
ry(-0.12021784410090534) q[6];
rz(-0.3247738651347216) q[6];
ry(2.308491383979978) q[7];
rz(-1.4737486846979826) q[7];
ry(-1.0509111091370436) q[8];
rz(0.22391894914029642) q[8];
ry(1.570878374586319) q[9];
rz(1.401865557297354) q[9];
ry(2.378743947326974) q[10];
rz(-3.1016142190832077) q[10];
ry(-1.8589808003103023) q[11];
rz(-0.431263928382939) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.0245364356083781) q[0];
rz(-0.23513140876257133) q[0];
ry(-1.5708297982662651) q[1];
rz(-1.4033475879180939) q[1];
ry(0.22422531112121644) q[2];
rz(-2.515603784308854) q[2];
ry(0.29607002101180907) q[3];
rz(0.7463427041922769) q[3];
ry(1.594581980851487) q[4];
rz(0.18510963668639313) q[4];
ry(-1.5893546142990012) q[5];
rz(2.76427681103916) q[5];
ry(-3.077454913143944) q[6];
rz(1.7803690093688502) q[6];
ry(-3.074365384098303) q[7];
rz(1.6820571175332857) q[7];
ry(0.2551802267619254) q[8];
rz(-1.5194483452214578) q[8];
ry(-0.0008172097691243607) q[9];
rz(0.13149608458447895) q[9];
ry(0.6880409343500727) q[10];
rz(1.9326853963823298) q[10];
ry(-2.1902238224612325) q[11];
rz(-1.4942686524780466) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5707886115233505) q[0];
rz(-2.368709138894971) q[0];
ry(2.5433531799785625) q[1];
rz(-0.3069248741946398) q[1];
ry(-2.981785589935233) q[2];
rz(1.21952887033543) q[2];
ry(0.05956972140303041) q[3];
rz(-0.3374236350390832) q[3];
ry(-3.126202531237804) q[4];
rz(2.966231298014583) q[4];
ry(2.9928023694415247) q[5];
rz(2.4749150036805743) q[5];
ry(0.008707118239433598) q[6];
rz(-2.07130848446381) q[6];
ry(0.9561565675444269) q[7];
rz(-0.06856952203595856) q[7];
ry(-3.052002592506898) q[8];
rz(1.7967688496904541) q[8];
ry(3.1414163589290123) q[9];
rz(1.7674674520785223) q[9];
ry(1.6942934325338523) q[10];
rz(1.1129892983522647) q[10];
ry(-2.660409139983108) q[11];
rz(0.7857269031955907) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.3586911424761818) q[0];
rz(0.23198670102356633) q[0];
ry(-2.2245776986340657) q[1];
rz(1.2222945038740844) q[1];
ry(3.1380792262771164) q[2];
rz(0.8349326906403727) q[2];
ry(-0.7618376428584446) q[3];
rz(0.9161551182737897) q[3];
ry(2.4013731719174833) q[4];
rz(0.9323153275852327) q[4];
ry(-0.9023188771182644) q[5];
rz(-0.34122970906060995) q[5];
ry(0.22142097729973464) q[6];
rz(2.1980773876657977) q[6];
ry(2.0720027031101473) q[7];
rz(0.03916130819190844) q[7];
ry(1.3019136074363162) q[8];
rz(-1.2708323368141696) q[8];
ry(0.09281707584506163) q[9];
rz(-2.919648431664039) q[9];
ry(-1.4947623945516062) q[10];
rz(1.7251241393931123) q[10];
ry(-2.746186773667951) q[11];
rz(1.8479661202701414) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.141509693869558) q[0];
rz(-1.1734278032547063) q[0];
ry(-1.6867004678510227) q[1];
rz(2.893595742317486) q[1];
ry(0.0038356023973324938) q[2];
rz(2.8861103550846003) q[2];
ry(3.0761742103036354) q[3];
rz(0.46370129592616655) q[3];
ry(-0.04057137910598296) q[4];
rz(-0.7149146368980537) q[4];
ry(-3.0797346766861335) q[5];
rz(-0.09385980512770452) q[5];
ry(-0.05911218849632949) q[6];
rz(-1.2368879413678258) q[6];
ry(2.927874984260983) q[7];
rz(1.8955745618732225) q[7];
ry(-3.136131806866305) q[8];
rz(2.114996966422898) q[8];
ry(-3.124970146959065) q[9];
rz(-0.705677417386803) q[9];
ry(1.2565430801041524) q[10];
rz(2.326840960794428) q[10];
ry(-1.4319809391831235) q[11];
rz(-1.5075505564389367) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.9394957341239518) q[0];
rz(2.121005106007903) q[0];
ry(0.588601859526925) q[1];
rz(-1.8212942915400103) q[1];
ry(-0.892291462514713) q[2];
rz(3.072099795736969) q[2];
ry(-1.09126570701948) q[3];
rz(-2.8548933188184216) q[3];
ry(-1.246620128084903) q[4];
rz(1.6947306693870434) q[4];
ry(-1.0635401005061436) q[5];
rz(1.9946873033519301) q[5];
ry(2.4923934817114017) q[6];
rz(-1.5939262621231807) q[6];
ry(-0.5869814872048114) q[7];
rz(-1.3490156843127794) q[7];
ry(0.3156300692094556) q[8];
rz(1.8322765620367285) q[8];
ry(-2.675602908933964) q[9];
rz(1.9779866808916013) q[9];
ry(-1.7560727798472628) q[10];
rz(-0.17146581947220607) q[10];
ry(-2.973162207374671) q[11];
rz(-1.7630905068543035) q[11];