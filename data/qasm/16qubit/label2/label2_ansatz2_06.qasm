OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.5707963260634363) q[0];
rz(-3.1415925741900512) q[0];
ry(-1.5707963182829312) q[1];
rz(0.5952265819932956) q[1];
ry(1.570796324086085) q[2];
rz(-1.5707964010957858) q[2];
ry(1.5707963938819054) q[3];
rz(1.5707962831786306) q[3];
ry(1.5707960303425639) q[4];
rz(-0.2688439677515236) q[4];
ry(1.5708149006422367) q[5];
rz(-1.6770068063167726e-09) q[5];
ry(-1.5709024547315458) q[6];
rz(-0.0006288284752156825) q[6];
ry(3.101547281407641) q[7];
rz(1.464612241035347) q[7];
ry(-3.0917987134599043) q[8];
rz(0.6057723741373272) q[8];
ry(-9.486120677411212e-06) q[9];
rz(-0.18840995792613766) q[9];
ry(1.5707964862963495) q[10];
rz(-1.5093620067047662) q[10];
ry(-1.629536602779133) q[11];
rz(-1.3058833895780645e-06) q[11];
ry(3.0916021581455113) q[12];
rz(3.141579338287875) q[12];
ry(4.524812071693418e-07) q[13];
rz(-1.5311598122266974) q[13];
ry(1.5707957380606372) q[14];
rz(-0.5363477372248386) q[14];
ry(-1.5239836470790664) q[15];
rz(2.124782803036851e-07) q[15];
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
ry(0.7177487116751767) q[0];
rz(2.948442332845197) q[0];
ry(8.289671971084545e-08) q[1];
rz(-0.14400148161791546) q[1];
ry(-1.5707963383878178) q[2];
rz(1.3817509965157138) q[2];
ry(1.5707963756427052) q[3];
rz(-1.2894273366485978) q[3];
ry(-0.18919361417553748) q[4];
rz(1.8350652242865682) q[4];
ry(-1.801354143370438) q[5];
rz(2.8982494023309355) q[5];
ry(-0.2464243688174066) q[6];
rz(1.5714079276536064) q[6];
ry(3.14159229249125) q[7];
rz(1.7066843753999885) q[7];
ry(4.6664638060356374e-05) q[8];
rz(1.3205349315187358) q[8];
ry(-3.1415925852316295) q[9];
rz(-1.409126142199237) q[9];
ry(-3.1415926406352437) q[10];
rz(1.8825350209571923) q[10];
ry(0.6452946029081683) q[11];
rz(2.07201188505916) q[11];
ry(-1.5748551242941176) q[12];
rz(-0.1859755739262146) q[12];
ry(1.5706611891494937) q[13];
rz(0.011367100899369889) q[13];
ry(-0.0001350206555201794) q[14];
rz(-1.0344514577492163) q[14];
ry(1.570413490475384) q[15];
rz(-1.06014307254643) q[15];
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
ry(-1.5651330009802678e-08) q[0];
rz(-2.948529784087371) q[0];
ry(-2.2298013213628565e-08) q[1];
rz(2.6903003279174342) q[1];
ry(-1.5707963465879304) q[2];
rz(-2.990436193538413) q[2];
ry(-1.570796351855237) q[3];
rz(0.0007957614979225981) q[3];
ry(1.5708071388290508) q[4];
rz(-1.2767616461547937) q[4];
ry(9.849620674629023e-09) q[5];
rz(-0.3976021883354875) q[5];
ry(0.19907674603359246) q[6];
rz(-3.141329512927998) q[6];
ry(-3.14159261102077) q[7];
rz(1.8132446865220464) q[7];
ry(-6.530179000450225e-08) q[8];
rz(-0.3513412319321059) q[8];
ry(-3.1415926262185647) q[9];
rz(-1.6053064713441545) q[9];
ry(3.14159260653149) q[10];
rz(-2.8912882925794254) q[10];
ry(3.141592419352051) q[11];
rz(1.0789003689300183) q[11];
ry(3.141586582665136) q[12];
rz(1.19023716403707) q[12];
ry(-3.141586423798784) q[13];
rz(1.5819590488494126) q[13];
ry(-0.19906541509957698) q[14];
rz(1.5708000307509158) q[14];
ry(-3.141592426699477) q[15];
rz(2.0651461993143254) q[15];
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
ry(2.4710835877395216) q[0];
rz(-2.7247941407706464) q[0];
ry(2.7169426449290928) q[1];
rz(-1.5708570062741627) q[1];
ry(0.005300970184300092) q[2];
rz(1.3554701341994904) q[2];
ry(-1.5655553701629739) q[3];
rz(-1.521547609168745) q[3];
ry(3.141585748570681) q[4];
rz(-1.2791368182606035) q[4];
ry(-0.08355441197675706) q[5];
rz(-1.1007152869800194) q[5];
ry(1.5709315267471506) q[6];
rz(-2.150195903931305) q[6];
ry(-1.6206488272755741) q[7];
rz(-3.137432905089338) q[7];
ry(0.04986297530089434) q[8];
rz(3.1374353018234995) q[8];
ry(-2.738209481378817e-08) q[9];
rz(-2.649009974010277) q[9];
ry(-1.321119464388036) q[10];
rz(1.5707963219502445) q[10];
ry(-3.140043311637972) q[11];
rz(0.5776864161352728) q[11];
ry(-3.1405331365103657) q[12];
rz(2.9470569752724116) q[12];
ry(-1.5718545264008634) q[13];
rz(-5.82297125950872e-05) q[13];
ry(0.18180783563105596) q[14];
rz(-2.119238503180382) q[14];
ry(3.1415769616157543) q[15];
rz(-0.704906231189586) q[15];
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
ry(3.141592630401229) q[0];
rz(1.9876633303291174) q[0];
ry(-0.63572385794836) q[1];
rz(3.141592094945874) q[1];
ry(0.0002248890888773944) q[2];
rz(0.16409040764991253) q[2];
ry(-0.00022476959662487414) q[3];
rz(3.1281610412722065) q[3];
ry(-3.1415581100874586) q[4];
rz(-0.0023509030557994372) q[4];
ry(3.1415908522189566) q[5];
rz(-0.16919031496553139) q[5];
ry(3.101391850407045) q[6];
rz(3.0379294721760566) q[6];
ry(-1.570769669592103) q[7];
rz(1.747075553396176) q[7];
ry(-1.573939929486501) q[8];
rz(-1.5429657295088748) q[8];
ry(-1.5707963242520973) q[9];
rz(-3.1415673797060113) q[9];
ry(1.401989893429656) q[10];
rz(1.6530818876192148) q[10];
ry(-2.3760623171604025) q[11];
rz(-1.6066088694252807) q[11];
ry(-2.882923284575047) q[12];
rz(2.889845411539331) q[12];
ry(2.883702664426665) q[13];
rz(0.17895588679634652) q[13];
ry(-3.1411684592586173) q[14];
rz(-0.5497114226674027) q[14];
ry(-3.1415463615449735) q[15];
rz(-2.259399358742992) q[15];
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
ry(-1.5707963333453627) q[0];
rz(1.9046871353631427) q[0];
ry(1.5707963527997366) q[1];
rz(0.04264256398892594) q[1];
ry(1.341528817622307e-06) q[2];
rz(-1.1843856463162616) q[2];
ry(-1.4165159614863465e-06) q[3];
rz(-1.4126474688687032) q[3];
ry(1.5707963807374226) q[4];
rz(-1.3290996603237515) q[4];
ry(-1.5707962915869738) q[5];
rz(1.3164842335375904) q[5];
ry(1.5707964219294168) q[6];
rz(1.6663557759821548) q[6];
ry(-7.840523874591554e-08) q[7];
rz(-0.665107885672031) q[7];
ry(-2.5328107913101895e-05) q[8];
rz(1.0896388538179493) q[8];
ry(-1.5707955236717248) q[9];
rz(2.6174434208194945) q[9];
ry(-3.8580939349941445e-06) q[10];
rz(1.48849544465428) q[10];
ry(-7.816670371418866e-06) q[11];
rz(2.6602847744620095) q[11];
ry(-2.3601146681428986e-06) q[12];
rz(2.8543068562695075) q[12];
ry(2.336635609884785e-06) q[13];
rz(-2.8471178078728343) q[13];
ry(1.572934341867828) q[14];
rz(-0.536093861419337) q[14];
ry(-1.5707962612215631) q[15];
rz(1.5707861660568925) q[15];
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
ry(3.14159265223857) q[0];
rz(1.6410551435588028) q[0];
ry(-1.0526155946877225e-08) q[1];
rz(2.826374685094897) q[1];
ry(3.1415925603564916) q[2];
rz(2.2243730934845125) q[2];
ry(5.869356576787414e-08) q[3];
rz(0.1290495706911723) q[3];
ry(3.1415925717601287) q[4];
rz(-2.6392803706393924) q[4];
ry(-3.1415926367098415) q[5];
rz(0.4077301265224868) q[5];
ry(-8.458862999721532e-08) q[6];
rz(2.9661570324940643) q[6];
ry(1.916882334573801e-07) q[7];
rz(2.1821814426849624) q[7];
ry(2.55596596379045e-06) q[8];
rz(0.7355568665319293) q[8];
ry(2.501851439931907e-06) q[9];
rz(0.40151848845577565) q[9];
ry(-1.5707877199909235) q[10];
rz(3.131564881983139) q[10];
ry(3.141575188607424) q[11];
rz(2.6469831024219284) q[11];
ry(-8.078046955775476e-06) q[12];
rz(2.3361795203142077) q[12];
ry(-8.090079722578025e-06) q[13];
rz(-2.288118216523375) q[13];
ry(1.57079638968758) q[14];
rz(-1.5452894892547473) q[14];
ry(-1.5707963957035582) q[15];
rz(-1.5707999317167436) q[15];
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
ry(1.5707963555753155) q[0];
rz(-0.1283921509152728) q[0];
ry(-1.5707964938311247) q[1];
rz(3.000048287489394) q[1];
ry(-2.0923425569918663e-07) q[2];
rz(-0.16713333368823943) q[2];
ry(-2.576654555666254e-07) q[3];
rz(-1.8937024333569519) q[3];
ry(-1.4889261912287566e-07) q[4];
rz(-0.26053930008604453) q[4];
ry(0.0001204700233108369) q[5];
rz(0.9088146859125485) q[5];
ry(3.8927419937292464e-06) q[6];
rz(-1.392920439284887) q[6];
ry(3.1415923278778854) q[7];
rz(0.8276988012452948) q[7];
ry(-2.83580234014321e-07) q[8];
rz(-1.6444104506690806) q[8];
ry(3.141592363375548) q[9];
rz(-1.181160765535017) q[9];
ry(-3.1415926483592727) q[10];
rz(-0.5044892216517233) q[10];
ry(3.141592650101537) q[11];
rz(0.6053816138567463) q[11];
ry(-3.1415873636725) q[12];
rz(0.0733675554494848) q[12];
ry(3.141587320202283) q[13];
rz(3.1170762657606654) q[13];
ry(-1.5707996570322094) q[14];
rz(0.043887650158160485) q[14];
ry(1.5963031455854697) q[15];
rz(0.6647752231700733) q[15];
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
ry(-0.8387882019304113) q[0];
rz(-0.3757681449233834) q[0];
ry(2.400470354838235) q[1];
rz(-0.5665795315617236) q[1];
ry(0.9092778069605262) q[2];
rz(0.29666843828994166) q[2];
ry(-2.2323148796972023) q[3];
rz(2.9987013206981286) q[3];
ry(-1.5707963114698962) q[4];
rz(2.3099747164990037) q[4];
ry(-1.570792705692024) q[5];
rz(3.1414536749440574) q[5];
ry(3.735628757439713e-06) q[6];
rz(3.0437944894274453) q[6];
ry(3.141592316734711) q[7];
rz(2.2627058978557244) q[7];
ry(3.1415926103511143) q[8];
rz(-0.24423036677333698) q[8];
ry(1.4533732439758751e-08) q[9];
rz(0.18761429019940312) q[9];
ry(2.6495470032748555e-08) q[10];
rz(-0.6607388074205964) q[10];
ry(3.1415926482926637) q[11];
rz(-2.3823307041366566) q[11];
ry(-1.8640186895643e-06) q[12];
rz(-3.021793942647973) q[12];
ry(1.8656260465377272e-06) q[13];
rz(-0.13946955233126793) q[13];
ry(-1.3769986928480193e-08) q[14];
rz(0.45394276544536527) q[14];
ry(3.1415926491384423) q[15];
rz(-1.3457834414625152) q[15];
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
ry(2.035110329882808) q[0];
rz(-0.8422028633672429) q[0];
ry(-2.0351103080172086) q[1];
rz(2.2993897255113747) q[1];
ry(3.1415646326356645) q[2];
rz(-0.8920078272294553) q[2];
ry(-2.994043032966687e-05) q[3];
rz(-1.0457853764786333) q[3];
ry(3.1414048067289744) q[4];
rz(-2.020469976427017) q[4];
ry(-1.5706695913188748) q[5];
rz(1.9527451342546318) q[5];
ry(1.570793689373149) q[6];
rz(-3.0068232657292033) q[6];
ry(-3.1413910281457467) q[7];
rz(1.6924230490001597) q[7];
ry(-2.104891730487181e-07) q[8];
rz(-0.9830696886783193) q[8];
ry(2.852588129947088e-07) q[9];
rz(-2.1357964631196817) q[9];
ry(4.370083847250762e-06) q[10];
rz(-1.851503837542797) q[10];
ry(4.365403283411815e-06) q[11];
rz(-1.612298959632401) q[11];
ry(3.1415767242662493) q[12];
rz(1.6725472297895037) q[12];
ry(1.585706943199081e-05) q[13];
rz(-1.5157560969931334) q[13];
ry(-3.1415662983166075) q[14];
rz(-0.9380647392117795) q[14];
ry(-3.477857723055422e-05) q[15];
rz(2.1454614014838302) q[15];