OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-1.8707136727831404) q[0];
rz(1.5765270114519385) q[0];
ry(-2.5153430553278815) q[1];
rz(1.4691815414216247) q[1];
ry(0.49792557320164654) q[2];
rz(2.6108581144526393) q[2];
ry(-3.0446459573882474) q[3];
rz(2.0172032680313707) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.639396702254671) q[0];
rz(-0.08855969090245885) q[0];
ry(1.1062081056748223) q[1];
rz(2.8517144492019355) q[1];
ry(1.9668469489954032) q[2];
rz(-0.3546895827680432) q[2];
ry(-1.3997409826354668) q[3];
rz(0.5570419305878964) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-3.0731723426816107) q[0];
rz(2.528187368443127) q[0];
ry(-2.0509974488268568) q[1];
rz(-1.4468011277432369) q[1];
ry(1.36958215631668) q[2];
rz(-0.4250766026624537) q[2];
ry(2.778089673472354) q[3];
rz(-2.071760175704049) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7960889783253735) q[0];
rz(1.6724820350179326) q[0];
ry(0.9940555261093315) q[1];
rz(-1.3864700824083283) q[1];
ry(-1.4308485901718235) q[2];
rz(1.7627624590657) q[2];
ry(0.47092362804397114) q[3];
rz(0.21867503301820168) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.7873651676491287) q[0];
rz(1.6905300499315572) q[0];
ry(2.0387300571033777) q[1];
rz(2.24571762493506) q[1];
ry(3.1113190848816004) q[2];
rz(1.5651301215468982) q[2];
ry(1.999674507820342) q[3];
rz(1.4007231714853674) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.0035716271350257) q[0];
rz(1.1038537282699492) q[0];
ry(2.699321547049386) q[1];
rz(-2.973078703626704) q[1];
ry(2.234906795197194) q[2];
rz(2.6513713881038554) q[2];
ry(1.7269078152002155) q[3];
rz(-2.832591633764181) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.2095053246978864) q[0];
rz(-2.1687576097491204) q[0];
ry(-0.6937502490687377) q[1];
rz(2.812458243223992) q[1];
ry(-2.712087947943635) q[2];
rz(2.0715295830295455) q[2];
ry(1.0736377040998057) q[3];
rz(-2.5466237172087465) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9968383299339258) q[0];
rz(-2.1254742652601886) q[0];
ry(2.0918528005105177) q[1];
rz(0.6591518494711665) q[1];
ry(0.3180884571662288) q[2];
rz(2.6683454707533363) q[2];
ry(-1.5538900376666156) q[3];
rz(2.9646461432222635) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.3507695487967852) q[0];
rz(2.959780363266416) q[0];
ry(-0.41197142636062983) q[1];
rz(-2.2821392541193486) q[1];
ry(-2.2189054533843455) q[2];
rz(-0.05947239687164309) q[2];
ry(-3.025049498702928) q[3];
rz(-0.6435444804750545) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.23168023088907508) q[0];
rz(0.6357825690656592) q[0];
ry(0.8144662961974252) q[1];
rz(1.6447585061847971) q[1];
ry(1.0617959082796684) q[2];
rz(2.4345117043123117) q[2];
ry(2.67429181558704) q[3];
rz(1.3570311523144003) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.9345411606096938) q[0];
rz(1.6691599340922734) q[0];
ry(-0.8891646351270914) q[1];
rz(1.1958294570286097) q[1];
ry(1.2184291709169706) q[2];
rz(0.4110384726927822) q[2];
ry(-1.3387255457505667) q[3];
rz(2.417438276022541) q[3];