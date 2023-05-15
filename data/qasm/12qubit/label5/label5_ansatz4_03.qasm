OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5708050453775635) q[0];
rz(2.569901663355608) q[0];
ry(-1.947094586003385) q[1];
rz(1.8070077313994588) q[1];
ry(-2.642678044590231) q[2];
rz(1.569778645530723) q[2];
ry(1.5708239347971384) q[3];
rz(-1.5116922611666173) q[3];
ry(-1.0075642964765593e-06) q[4];
rz(-1.7517916466153416) q[4];
ry(0.0003403784615771954) q[5];
rz(0.5288237563760165) q[5];
ry(-1.57086954930373) q[6];
rz(1.5707898174022312) q[6];
ry(-1.5707845917036631) q[7];
rz(1.2027844370090177) q[7];
ry(-3.0834177461741192) q[8];
rz(1.5706670618245946) q[8];
ry(-1.8685446247831976) q[9];
rz(1.5707410459457893) q[9];
ry(-0.7781498137586029) q[10];
rz(2.8761843500956106) q[10];
ry(-1.5707978114167291) q[11];
rz(0.4462603949019668) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(3.1313551980228067) q[0];
rz(-1.7899986092066928) q[0];
ry(-1.707295637896257) q[1];
rz(-1.0567803696018228) q[1];
ry(-1.5707950160943922) q[2];
rz(-1.7664812769597016) q[2];
ry(-1.579827710070343) q[3];
rz(-3.0920335009536624) q[3];
ry(-1.5707744791747762) q[4];
rz(2.5689536120252154) q[4];
ry(-4.857947224936121e-06) q[5];
rz(1.67577932353122) q[5];
ry(0.2689267295912972) q[6];
rz(1.5708049137974862) q[6];
ry(1.7062919135888622e-05) q[7];
rz(0.5372526561059582) q[7];
ry(1.570795346946623) q[8];
rz(3.1415912141043805) q[8];
ry(1.570794652020697) q[9];
rz(3.141590172102013) q[9];
ry(-3.141589031969547) q[10];
rz(2.876325802167529) q[10];
ry(1.038430278388347e-05) q[11];
rz(-2.017060252057691) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-3.139975030009025) q[0];
rz(1.9241543023954362) q[0];
ry(3.1411986804932863) q[1];
rz(-2.660455158026755) q[1];
ry(-5.95705580526129e-05) q[2];
rz(0.37680985015029833) q[2];
ry(-3.137028978560846) q[3];
rz(0.2067814846509268) q[3];
ry(1.023244248334313e-06) q[4];
rz(-1.003023238769754) q[4];
ry(-1.5707995676594588) q[5];
rz(-3.1415580055199674) q[5];
ry(-1.5707963204380178) q[6];
rz(2.348711379441718) q[6];
ry(-1.5607097883713958) q[7];
rz(2.4046116160451287) q[7];
ry(-1.5707988589324586) q[8];
rz(2.487236147930168) q[8];
ry(-1.5707942796031582) q[9];
rz(3.1379172370676423) q[9];
ry(0.11351630493619158) q[10];
rz(-1.2463264840223602) q[10];
ry(1.570804385266803) q[11];
rz(-1.5708304182757642) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.731816292996825) q[0];
rz(-3.035583167951233) q[0];
ry(1.80480767902385) q[1];
rz(3.100034747461093) q[1];
ry(3.1415368712161746) q[2];
rz(-1.3965770928773367) q[2];
ry(3.1316669102918566) q[3];
rz(-1.5658356442960035) q[3];
ry(-1.57079824608767) q[4];
rz(-1.1177976230844158) q[4];
ry(1.570808730004487) q[5];
rz(1.5616244772809216) q[5];
ry(3.141592532747096) q[6];
rz(2.337984975455976) q[6];
ry(3.1415893068648337) q[7];
rz(-0.8992369164466961) q[7];
ry(1.5707849606817301) q[8];
rz(-1.5707803956930828) q[8];
ry(1.570781331716185) q[9];
rz(-2.5481802038436765) q[9];
ry(1.570794487552135) q[10];
rz(-1.7548399563423374) q[10];
ry(-1.570898634052955) q[11];
rz(1.2494404627193085) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.7042119011545042) q[0];
rz(-3.0992776014336) q[0];
ry(3.1258224054138157) q[1];
rz(-3.042754491448183) q[1];
ry(-1.3040244731870394e-06) q[2];
rz(1.5777268962207882) q[2];
ry(-1.5707848407786724) q[3];
rz(-1.6845797312662203) q[3];
ry(-2.1659030113418964e-05) q[4];
rz(-0.45299661526437696) q[4];
ry(-1.5707985116339982) q[5];
rz(1.8372814933237196) q[5];
ry(-1.1982989029470401) q[6];
rz(-1.5698859395821199) q[6];
ry(-3.8139698948156746e-06) q[7];
rz(-1.4072806759649972) q[7];
ry(-1.5707940686378272) q[8];
rz(-3.1415925346564064) q[8];
ry(-1.2131644321181374e-06) q[9];
rz(0.9773642450539929) q[9];
ry(-3.141569459889834) q[10];
rz(0.444359785200964) q[10];
ry(-3.123819296005084) q[11];
rz(1.5740400838719015) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(-1.5703213170343637) q[0];
rz(-1.571165678496679) q[0];
ry(-1.570794903512555) q[1];
rz(-0.025332679286031015) q[1];
ry(1.5708497069346974) q[2];
rz(1.570809139962347) q[2];
ry(-3.138608186874927) q[3];
rz(1.6280529523475689) q[3];
ry(-1.7004710354385608) q[4];
rz(-0.46106320533754896) q[4];
ry(3.141580611192903) q[5];
rz(-1.3042358745437677) q[5];
ry(1.573978519176988) q[6];
rz(3.0987789326167556) q[6];
ry(1.5707980749981396) q[7];
rz(1.5707955304430854) q[7];
ry(-1.0440582132063216) q[8];
rz(1.5059512454840272e-06) q[8];
ry(-1.5707962257852213) q[9];
rz(-1.7679169622632172) q[9];
ry(-4.147162236530455e-06) q[10];
rz(-0.7366676421733231) q[10];
ry(1.5707901388149264) q[11];
rz(2.448429923695983e-06) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
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
ry(0.7603409601401472) q[0];
rz(-1.5959565396185316) q[0];
ry(-1.5525544228012222e-06) q[1];
rz(2.1494414563360227) q[1];
ry(1.57079797713289) q[2];
rz(1.545290807826043) q[2];
ry(-1.7487383374212719e-07) q[3];
rz(0.382749830664344) q[3];
ry(7.01315787834389e-05) q[4];
rz(-1.1350250487843325) q[4];
ry(1.5707966516288008) q[5];
rz(-1.0174790289416944) q[5];
ry(1.570796566058934) q[6];
rz(-0.025475211754863128) q[6];
ry(1.570795494306366) q[7];
rz(0.5533174608963926) q[7];
ry(-1.5694132621514143) q[8];
rz(1.5454548786407685) q[8];
ry(1.3597971824097499e-06) q[9];
rz(2.3212338421268126) q[9];
ry(-3.762692792429581e-06) q[10];
rz(-1.48801139808914) q[10];
ry(-1.5707988208673953) q[11];
rz(-1.0174840952972533) q[11];