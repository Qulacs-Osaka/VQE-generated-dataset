OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.1589531603728784) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.0162386661671974) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.0341991391347261) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.2668279502400215) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.38941113154149) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.22299692613763597) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-1.6731036154113155) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-1.8452491485820655) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-1.3992488851173053) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.013933017546394913) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.24935828382543085) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-1.0539964742606216) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.8347326911591043) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(2.306396345628823) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.5555498976411418) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.18559585798289166) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.18572337206288475) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(-0.5230801297251864) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(2.0837733650834616) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(1.0630527925666278) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-0.6854061131003774) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.04298777839837004) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.043091600041475116) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-1.366704552405193) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.1401640228404559) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.1418255595261892) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(2.096566294541378) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(3.1219268785865686) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-3.123461069540866) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-1.214179188581711) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(3.1321258413748434) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(3.13231648329623) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.34965630198650555) q[11];
cx q[10],q[11];
rz(-0.22364453777707852) q[0];
rz(1.1540520185177867) q[1];
rz(-0.004918287508175395) q[2];
rz(1.1364148080103211) q[3];
rz(0.9805684208779707) q[4];
rz(0.3245782219458908) q[5];
rz(0.3817046123923929) q[6];
rz(-0.35102705697089365) q[7];
rz(0.9973837127696827) q[8];
rz(-0.027663488964166493) q[9];
rz(1.2532690868828207) q[10];
rz(-2.4085355024905373) q[11];
rx(1.4525401975655177) q[0];
rx(-1.7034890150069801) q[1];
rx(1.3501777537785964) q[2];
rx(-3.1250571546373904) q[3];
rx(3.1320346901212246) q[4];
rx(0.0020882708948851) q[5];
rx(-3.1411095281689487) q[6];
rx(2.18918012009157) q[7];
rx(3.135218126039455) q[8];
rx(-3.1395619132812276) q[9];
rx(-3.1379265429233407) q[10];
rx(0.02446030670376942) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-0.9244853779474244) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(2.3513766895898667) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.9457556319371234) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(-0.012565075708105295) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.02571267057173) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(1.638792251717123) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(1.1005781668801902) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(-0.22810644897617297) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(1.9783126082260312) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-1.9615896123594907) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(1.2077099989067095) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(1.2940448540166485) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.6155423912844112) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(2.51830959117809) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.020648470985417444) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(1.1456596165111197) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-1.1398683692745242) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(1.859936903859799) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-1.5804634084667608) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-1.5704389504204663) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-1.569955667992754) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.06809354038433844) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-3.0715343251767804) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.028193293928462614) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(0.3735568046954317) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.36817140080137456) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-1.595805323168592) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(2.8653074890782744) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-2.866491910023033) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.10440833615489678) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(3.068045528085764) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-3.0680748901029857) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(0.050201226158058214) q[11];
cx q[10],q[11];
rz(1.6179407781675978) q[0];
rz(-0.23988054868996467) q[1];
rz(1.0343649939268136) q[2];
rz(-0.26092275813167487) q[3];
rz(-2.105030207061889) q[4];
rz(-0.8500422300645448) q[5];
rz(-0.00033271757760543425) q[6];
rz(-1.120833484175468) q[7];
rz(-1.544159818780803) q[8];
rz(2.420023752851715) q[9];
rz(-0.777852827894244) q[10];
rz(0.11210132894457399) q[11];
rx(2.63358305836218) q[0];
rx(-1.4277137615158544) q[1];
rx(-3.1360022868466335) q[2];
rx(-3.132367467484623) q[3];
rx(3.140121243345988) q[4];
rx(0.0010871128054355515) q[5];
rx(-2.189049859004417) q[6];
rx(-3.1414885251641564) q[7];
rx(3.134439950210671) q[8];
rx(0.0017149753187209907) q[9];
rx(3.137917118533068) q[10];
rx(-3.1171007901310785) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.7527551469838196) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(1.8750337893035276) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.8114314094729767) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.0565151837646936) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(-1.5894723381893021) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(-0.08166408492680767) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.5707862709669289) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.5694194384428725) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.547289189194315) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(0.025548031581286365) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(-0.028632596740378005) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(-0.09151881954867291) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.6200454524742154) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.6188382807073071) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(-0.06173070988041953) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(2.535012838898393) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(0.6108377172392829) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.05630962481390701) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.8536388433116658) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-2.294736540541531) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(-2.9831829459065577) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(0.7314603750134989) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(-0.730852422157368) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(0.16960503189799322) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.40936078146391064) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(0.40895556380342873) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(-2.9398822884137257) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(-2.1589722249204097) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-2.162124972533666) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(2.8048098193219992) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(2.9101464659522573) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-2.9116425473398393) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-1.6564091700101784) q[11];
cx q[10],q[11];
rz(-0.6492027591434075) q[0];
rz(-1.1341473416321448) q[1];
rz(-1.0675623453320024) q[2];
rz(-1.0822247613611724) q[3];
rz(-2.7426080593861686) q[4];
rz(0.9578239147100039) q[5];
rz(0.32934588216330496) q[6];
rz(3.0520623783118457) q[7];
rz(-1.1764887087882665) q[8];
rz(-1.7969743308283608) q[9];
rz(-1.8474756701562347) q[10];
rz(-2.6634441370135207) q[11];
rx(2.011578275043707) q[0];
rx(-1.8756573454601022) q[1];
rx(3.1314137004721876) q[2];
rx(-3.1350584309825766) q[3];
rx(-3.1411068588993274) q[4];
rx(9.660011317662846e-05) q[5];
rx(3.1394684535940716) q[6];
rx(-5.7782544742757125e-05) q[7];
rx(0.0005086038578421661) q[8];
rx(3.1414703582376777) q[9];
rx(-3.1404211005846854) q[10];
rx(-3.1386024904675414) q[11];
h q[0];
h q[1];
cx q[0],q[1];
rz(-1.5876284905664588) q[1];
cx q[0],q[1];
h q[0];
h q[1];
sdg q[0];
h q[0];
sdg q[1];
h q[1];
cx q[0],q[1];
rz(2.9226649907838023) q[1];
cx q[0],q[1];
h q[0];
s q[0];
h q[1];
s q[1];
cx q[0],q[1];
rz(-0.008109714464555384) q[1];
cx q[0],q[1];
h q[1];
h q[2];
cx q[1],q[2];
rz(0.0173890637994414) q[2];
cx q[1],q[2];
h q[1];
h q[2];
sdg q[1];
h q[1];
sdg q[2];
h q[2];
cx q[1],q[2];
rz(0.000667134290656931) q[2];
cx q[1],q[2];
h q[1];
s q[1];
h q[2];
s q[2];
cx q[1],q[2];
rz(0.0021694640940775043) q[2];
cx q[1],q[2];
h q[2];
h q[3];
cx q[2],q[3];
rz(-0.6619239135063701) q[3];
cx q[2],q[3];
h q[2];
h q[3];
sdg q[2];
h q[2];
sdg q[3];
h q[3];
cx q[2],q[3];
rz(0.6682406180192776) q[3];
cx q[2],q[3];
h q[2];
s q[2];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.5323210241836387) q[3];
cx q[2],q[3];
h q[3];
h q[4];
cx q[3],q[4];
rz(-0.11820574400795712) q[4];
cx q[3],q[4];
h q[3];
h q[4];
sdg q[3];
h q[3];
sdg q[4];
h q[4];
cx q[3],q[4];
rz(0.12060249205457942) q[4];
cx q[3],q[4];
h q[3];
s q[3];
h q[4];
s q[4];
cx q[3],q[4];
rz(0.11729082279331407) q[4];
cx q[3],q[4];
h q[4];
h q[5];
cx q[4],q[5];
rz(-0.6860252827577156) q[5];
cx q[4],q[5];
h q[4];
h q[5];
sdg q[4];
h q[4];
sdg q[5];
h q[5];
cx q[4],q[5];
rz(0.6866183523045803) q[5];
cx q[4],q[5];
h q[4];
s q[4];
h q[5];
s q[5];
cx q[4],q[5];
rz(0.20453720113039126) q[5];
cx q[4],q[5];
h q[5];
h q[6];
cx q[5],q[6];
rz(0.37439180835187974) q[6];
cx q[5],q[6];
h q[5];
h q[6];
sdg q[5];
h q[5];
sdg q[6];
h q[6];
cx q[5],q[6];
rz(-0.3753006203603362) q[6];
cx q[5],q[6];
h q[5];
s q[5];
h q[6];
s q[6];
cx q[5],q[6];
rz(0.2106615290918101) q[6];
cx q[5],q[6];
h q[6];
h q[7];
cx q[6],q[7];
rz(-0.2882687933311994) q[7];
cx q[6],q[7];
h q[6];
h q[7];
sdg q[6];
h q[6];
sdg q[7];
h q[7];
cx q[6],q[7];
rz(-0.286520143756049) q[7];
cx q[6],q[7];
h q[6];
s q[6];
h q[7];
s q[7];
cx q[6],q[7];
rz(0.13094936101522872) q[7];
cx q[6],q[7];
h q[7];
h q[8];
cx q[7],q[8];
rz(-0.761482387976404) q[8];
cx q[7],q[8];
h q[7];
h q[8];
sdg q[7];
h q[7];
sdg q[8];
h q[8];
cx q[7],q[8];
rz(0.7643309289833929) q[8];
cx q[7],q[8];
h q[7];
s q[7];
h q[8];
s q[8];
cx q[7],q[8];
rz(-0.35471089976484005) q[8];
cx q[7],q[8];
h q[8];
h q[9];
cx q[8],q[9];
rz(-0.5082205970052267) q[9];
cx q[8],q[9];
h q[8];
h q[9];
sdg q[8];
h q[8];
sdg q[9];
h q[9];
cx q[8],q[9];
rz(2.6335594644308657) q[9];
cx q[8],q[9];
h q[8];
s q[8];
h q[9];
s q[9];
cx q[8],q[9];
rz(0.356400438376849) q[9];
cx q[8],q[9];
h q[9];
h q[10];
cx q[9],q[10];
rz(2.4072801726406667) q[10];
cx q[9],q[10];
h q[9];
h q[10];
sdg q[9];
h q[9];
sdg q[10];
h q[10];
cx q[9],q[10];
rz(-2.4047125595599046) q[10];
cx q[9],q[10];
h q[9];
s q[9];
h q[10];
s q[10];
cx q[9],q[10];
rz(-0.4115239253789729) q[10];
cx q[9],q[10];
h q[10];
h q[11];
cx q[10],q[11];
rz(-2.5093004833344614) q[11];
cx q[10],q[11];
h q[10];
h q[11];
sdg q[10];
h q[10];
sdg q[11];
h q[11];
cx q[10],q[11];
rz(-0.6329250408151449) q[11];
cx q[10],q[11];
h q[10];
s q[10];
h q[11];
s q[11];
cx q[10],q[11];
rz(-0.3835712955965175) q[11];
cx q[10],q[11];
rz(0.10422856337558231) q[0];
rz(0.02333893705648529) q[1];
rz(-2.7134683533457755) q[2];
rz(-0.2281991746093849) q[3];
rz(0.26833828487386874) q[4];
rz(-0.43631785572552584) q[5];
rz(3.0970626600262507) q[6];
rz(-0.38865128030237406) q[7];
rz(-0.5823587666856689) q[8];
rz(-1.3493462080361964) q[9];
rz(1.6408725785723683) q[10];
rz(-0.40963360558003453) q[11];
rx(1.5055425604114365) q[0];
rx(-1.9186571490910718) q[1];
rx(3.1390642054172413) q[2];
rx(-0.00044917569534395815) q[3];
rx(3.140708408938875) q[4];
rx(-0.0024015060421133244) q[5];
rx(3.1393941703117894) q[6];
rx(3.1409148466738106) q[7];
rx(3.141256844527122) q[8];
rx(3.140996251545046) q[9];
rx(-3.141523579595815) q[10];
rx(-0.0033656347962566493) q[11];