OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.5853353378953887) q[0];
rz(2.2099948713043105) q[0];
ry(-2.3698574629877824) q[1];
rz(-0.8361944821088496) q[1];
ry(-1.0045962064649476) q[2];
rz(1.3336541693610748) q[2];
ry(1.2213275030655206) q[3];
rz(-1.3245089221370143) q[3];
ry(1.4998317788897175) q[4];
rz(0.1476658132941199) q[4];
ry(-2.586062066020431) q[5];
rz(-2.626320006977674) q[5];
ry(-1.5632556281040764) q[6];
rz(-0.7311826298888152) q[6];
ry(1.642258379168152) q[7];
rz(-3.063912364029498) q[7];
ry(0.8197993004482053) q[8];
rz(0.8117032978215449) q[8];
ry(2.9333899088510607) q[9];
rz(2.1538877122167666) q[9];
ry(-0.44178261857488277) q[10];
rz(-2.8703293643680428) q[10];
ry(-1.6101782135848501) q[11];
rz(0.29075081693852217) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.2224779997904087) q[0];
rz(2.237137454836123) q[0];
ry(0.7780420873423006) q[1];
rz(-0.6716359706650088) q[1];
ry(2.964006527330302) q[2];
rz(1.453204247699587) q[2];
ry(1.037075917668963) q[3];
rz(-2.783740841322811) q[3];
ry(-2.2639424034641364) q[4];
rz(-1.3333134007945757) q[4];
ry(2.3988161519810585) q[5];
rz(0.24820023784476594) q[5];
ry(0.035897892370068796) q[6];
rz(1.8689090550147327) q[6];
ry(2.1543892013900448) q[7];
rz(-1.5068297714444294) q[7];
ry(-2.1277859089813207) q[8];
rz(-2.2427837775581256) q[8];
ry(-1.0678160060874042) q[9];
rz(-1.9590552365035947) q[9];
ry(-2.6424480155646988) q[10];
rz(2.869338112766271) q[10];
ry(-0.016720382790334776) q[11];
rz(-1.8069226375149192) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.716101133055064) q[0];
rz(-1.900822733171899) q[0];
ry(1.799048770835583) q[1];
rz(-0.9488638169938737) q[1];
ry(-1.6936267598291004) q[2];
rz(-2.1099785781478237) q[2];
ry(-1.7828975224543209) q[3];
rz(2.353772404994429) q[3];
ry(3.111193018742499) q[4];
rz(2.015571780640844) q[4];
ry(-0.17390073880176438) q[5];
rz(1.762979465392073) q[5];
ry(0.02020072617961155) q[6];
rz(-1.7089257767878632) q[6];
ry(2.444736697045466) q[7];
rz(1.523380707701457) q[7];
ry(1.3486097968818809) q[8];
rz(-1.8951379751862794) q[8];
ry(-1.603416972310323) q[9];
rz(-2.4299038407207156) q[9];
ry(-2.2034460066254242) q[10];
rz(-0.11649133153674196) q[10];
ry(0.2473613299145612) q[11];
rz(1.4953755364384538) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.7093083124903073) q[0];
rz(-0.7727391195294168) q[0];
ry(-0.5747012800564368) q[1];
rz(2.937993408876522) q[1];
ry(-0.9221624650003317) q[2];
rz(2.212682641206417) q[2];
ry(1.6772353664481192) q[3];
rz(2.250103917015508) q[3];
ry(0.3090042015606542) q[4];
rz(-1.777681488473938) q[4];
ry(-1.9792022532215061) q[5];
rz(1.3061622532879777) q[5];
ry(3.1287891892262585) q[6];
rz(0.9884265638828879) q[6];
ry(-2.1814554867762586) q[7];
rz(-1.5579538675117695) q[7];
ry(1.9641896198775022) q[8];
rz(1.614444324508855) q[8];
ry(2.9792524997779672) q[9];
rz(2.0961059305953746) q[9];
ry(-2.189199531272278) q[10];
rz(2.1009836074520862) q[10];
ry(-0.6925013629464125) q[11];
rz(1.587440689790938) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.3949909122526614) q[0];
rz(0.006272227551606324) q[0];
ry(-2.6045154447718875) q[1];
rz(0.9315708664625504) q[1];
ry(2.061532905866236) q[2];
rz(-0.419386384209945) q[2];
ry(0.31718439405103105) q[3];
rz(2.72312137969791) q[3];
ry(-2.930772748476117) q[4];
rz(-1.3608072516041976) q[4];
ry(2.140848397120842) q[5];
rz(0.38243582154716993) q[5];
ry(-1.4272999418849803) q[6];
rz(1.5238926851825116) q[6];
ry(-1.893310499190914) q[7];
rz(-1.5132211136594294) q[7];
ry(-0.270156767058622) q[8];
rz(0.5988495123851454) q[8];
ry(-0.8902204048858247) q[9];
rz(2.6546860833213306) q[9];
ry(-3.1055882842788263) q[10];
rz(-0.2555393203983227) q[10];
ry(-0.542644801430853) q[11];
rz(1.60759598889581) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.0631798974805158) q[0];
rz(2.9833640878891856) q[0];
ry(1.216307372002666) q[1];
rz(-3.0745852699528275) q[1];
ry(-0.4781796231924247) q[2];
rz(2.626849821751494) q[2];
ry(0.6411165555185959) q[3];
rz(-0.9667017000198187) q[3];
ry(0.2484946956638323) q[4];
rz(-1.7143309306364047) q[4];
ry(1.6563947957894953) q[5];
rz(-1.2622918086463) q[5];
ry(-0.7669144233626861) q[6];
rz(1.6206944755530783) q[6];
ry(-0.5944716624070364) q[7];
rz(1.5254345575912356) q[7];
ry(-1.0060611642814692) q[8];
rz(-2.1085283390721914) q[8];
ry(1.404984281633583) q[9];
rz(-1.0915969003674693) q[9];
ry(-1.3724617990471923) q[10];
rz(2.3269325598385944) q[10];
ry(-2.5411399432313595) q[11];
rz(1.6620499433964042) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.8710596007473892) q[0];
rz(3.1105361866654255) q[0];
ry(-2.4795571369352705) q[1];
rz(1.445316495581913) q[1];
ry(-1.7629717975475518) q[2];
rz(2.2181927601453446) q[2];
ry(0.23752713515353896) q[3];
rz(0.793429885946848) q[3];
ry(1.62079859943126) q[4];
rz(0.006029246443973498) q[4];
ry(1.5260454515430733) q[5];
rz(-0.04357974469665394) q[5];
ry(1.4164985552205547) q[6];
rz(-1.5880831998689662) q[6];
ry(-1.8519199832837119) q[7];
rz(-1.5599718461203584) q[7];
ry(1.8045704257722148) q[8];
rz(-0.4199720757874613) q[8];
ry(2.018114552853426) q[9];
rz(1.842004621859302) q[9];
ry(2.131411949467275) q[10];
rz(1.4773764084050232) q[10];
ry(-1.2644261905262484) q[11];
rz(1.5616013642511586) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.3474587998571743) q[0];
rz(-2.334326524141608) q[0];
ry(-1.752117444104793) q[1];
rz(-1.067515743254983) q[1];
ry(-2.6800275609119435) q[2];
rz(-0.2514374096092697) q[2];
ry(2.54839592245522) q[3];
rz(2.7509454392344037) q[3];
ry(-0.020042844345498533) q[4];
rz(-0.5130839857837267) q[4];
ry(0.8613772771110639) q[5];
rz(3.0568738715188855) q[5];
ry(2.1850495600154387) q[6];
rz(-1.96826383312433) q[6];
ry(-1.908158982525109) q[7];
rz(1.5790183547814518) q[7];
ry(0.9709789085263404) q[8];
rz(-0.598147255342121) q[8];
ry(1.1490307718543367) q[9];
rz(-2.0180200151341214) q[9];
ry(-2.7037506015534682) q[10];
rz(-0.2984308431909761) q[10];
ry(2.547027230900787) q[11];
rz(1.5808027976410122) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.239361384119154) q[0];
rz(1.3106235809532922) q[0];
ry(-1.6708082361401524) q[1];
rz(-3.03152939225477) q[1];
ry(0.7370303687127964) q[2];
rz(1.3370159557157368) q[2];
ry(-1.1204022120839183) q[3];
rz(-1.141194407103442) q[3];
ry(3.0724765762716344) q[4];
rz(1.053741895359023) q[4];
ry(-2.4164006688337794) q[5];
rz(-0.5669431284136709) q[5];
ry(-3.111002109853465) q[6];
rz(-1.993519174774528) q[6];
ry(1.3876823415796238) q[7];
rz(-1.6049187637853717) q[7];
ry(-1.6533537024023088) q[8];
rz(-1.7157130057085905) q[8];
ry(-1.3760766299507683) q[9];
rz(-0.0585795120953185) q[9];
ry(1.9771601092904765) q[10];
rz(-1.9949153895592335) q[10];
ry(0.2515586688150076) q[11];
rz(2.396101636163424) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.6036328435178673) q[0];
rz(0.12042321958908267) q[0];
ry(-1.550727745197776) q[1];
rz(-0.4376907233792432) q[1];
ry(1.6939790459687671) q[2];
rz(2.4254946098073678) q[2];
ry(1.3127647990967157) q[3];
rz(-2.5483911591986868) q[3];
ry(0.47509924577347806) q[4];
rz(1.6143737236184093) q[4];
ry(-0.5479178845209364) q[5];
rz(-0.5886321854083603) q[5];
ry(1.5172211555609039) q[6];
rz(0.7206006615527576) q[6];
ry(2.199528180526761) q[7];
rz(1.532256048054749) q[7];
ry(-1.386366569468465) q[8];
rz(0.24001117268258607) q[8];
ry(-2.116454574055135) q[9];
rz(-1.7674233228846892) q[9];
ry(1.2668389757671454) q[10];
rz(-1.1207019782720442) q[10];
ry(-0.035829266846160626) q[11];
rz(-2.420659048451269) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.669735063864453) q[0];
rz(-1.0088850827602984) q[0];
ry(-0.14983301278904373) q[1];
rz(-1.4495246004167655) q[1];
ry(0.9958394802114103) q[2];
rz(-2.247953036052646) q[2];
ry(0.6159341111539117) q[3];
rz(2.3128441166453615) q[3];
ry(0.6661450966569458) q[4];
rz(2.079234342303929) q[4];
ry(-0.3025487792606541) q[5];
rz(0.640851811918255) q[5];
ry(0.04606558553315334) q[6];
rz(-0.6864235481552617) q[6];
ry(1.5094027837053074) q[7];
rz(1.4235015365441628) q[7];
ry(1.9257534576026991) q[8];
rz(-1.9084645150671575) q[8];
ry(-0.8000284295965718) q[9];
rz(-3.091489506524876) q[9];
ry(-2.5494518268042303) q[10];
rz(-2.5473695512484227) q[10];
ry(-2.831635309190827) q[11];
rz(-1.5704667259534517) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.4643837164515916) q[0];
rz(1.8153208886951786) q[0];
ry(0.5100255590428302) q[1];
rz(-2.1867574326088963) q[1];
ry(1.1195888321386231) q[2];
rz(0.1371260268588905) q[2];
ry(-1.5351285241514452) q[3];
rz(0.9863217579384758) q[3];
ry(3.0813897747134456) q[4];
rz(2.0994820289967846) q[4];
ry(1.0368874467844986) q[5];
rz(2.7958035285516707) q[5];
ry(1.6416704056578837) q[6];
rz(1.5122260800523168) q[6];
ry(-0.16944261377897504) q[7];
rz(-2.112336672056557) q[7];
ry(0.7023215204776365) q[8];
rz(0.2747642277937148) q[8];
ry(1.9138698360180104) q[9];
rz(-2.9659169313184885) q[9];
ry(1.2285070938524836) q[10];
rz(-0.7190524430967785) q[10];
ry(2.2505988281069946) q[11];
rz(1.6039021289135842) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.4787023093015925) q[0];
rz(2.2437416445257963) q[0];
ry(1.6217760856195946) q[1];
rz(2.1525244804385357) q[1];
ry(1.0900426054784027) q[2];
rz(-1.4251413284991687) q[2];
ry(1.996919181180747) q[3];
rz(3.0259062148536593) q[3];
ry(-0.870509816632655) q[4];
rz(-1.6281093575103824) q[4];
ry(-1.905609557866807) q[5];
rz(2.769873589258269) q[5];
ry(2.6882016134727538) q[6];
rz(1.4899444284125336) q[6];
ry(3.1309897552340367) q[7];
rz(0.10066763415014178) q[7];
ry(1.1139744897371422) q[8];
rz(1.9699033552354743) q[8];
ry(1.9430645550805377) q[9];
rz(-1.5742684025278555) q[9];
ry(-0.8755253640339278) q[10];
rz(1.7057680523532237) q[10];
ry(2.529013128085019) q[11];
rz(-2.1285846589408797) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(2.2003580729248062) q[0];
rz(3.0681061362601483) q[0];
ry(-2.036766881159341) q[1];
rz(-2.534467997733166) q[1];
ry(1.314226582859627) q[2];
rz(2.265963911866157) q[2];
ry(2.1135988979396245) q[3];
rz(-0.7118153213152778) q[3];
ry(1.9464108950203725) q[4];
rz(3.1008560862875396) q[4];
ry(2.4618267661024884) q[5];
rz(-0.24123381734267346) q[5];
ry(2.0282757845442028) q[6];
rz(-0.013905564209165623) q[6];
ry(0.023500783420402366) q[7];
rz(-2.386817418995608) q[7];
ry(-0.2054208977575076) q[8];
rz(-0.19869706669303522) q[8];
ry(-1.465808760880115) q[9];
rz(2.970649836285649) q[9];
ry(1.7674259041145222) q[10];
rz(-0.40924233449125236) q[10];
ry(0.0198783843896031) q[11];
rz(0.5422380932366375) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.6912799259204325) q[0];
rz(0.06606727759373761) q[0];
ry(-2.7384823645357885) q[1];
rz(-1.1013519604966637) q[1];
ry(1.9649657983990645) q[2];
rz(2.136038181364272) q[2];
ry(-1.8638684925728333) q[3];
rz(0.9541489946127416) q[3];
ry(-1.56725658077898) q[4];
rz(0.702267667087245) q[4];
ry(-2.051241726954185) q[5];
rz(0.026277882079004836) q[5];
ry(-1.610002776194661) q[6];
rz(-0.605241560136542) q[6];
ry(-1.5538367944634288) q[7];
rz(0.33530882024604836) q[7];
ry(-0.5743575419794811) q[8];
rz(-1.3497600424701766) q[8];
ry(-0.3901433678307251) q[9];
rz(-0.4739194029973222) q[9];
ry(-0.7689721915016197) q[10];
rz(-1.2422622587205798) q[10];
ry(1.5829815756966426) q[11];
rz(-0.2504798541839275) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.2539140728567748) q[0];
rz(0.8733654398574887) q[0];
ry(-2.4686731807748057) q[1];
rz(1.914973524192205) q[1];
ry(-0.6565773046677865) q[2];
rz(1.6866493143411199) q[2];
ry(2.5622715412016084) q[3];
rz(-3.002867428680655) q[3];
ry(-3.099747984982705) q[4];
rz(-0.4367707917486329) q[4];
ry(2.407480028616402) q[5];
rz(2.845774037098979) q[5];
ry(0.010922546647467349) q[6];
rz(2.4715190877597215) q[6];
ry(-3.138044050958393) q[7];
rz(1.9281427308781076) q[7];
ry(-2.8085328646647514) q[8];
rz(-2.3367972396664163) q[8];
ry(1.2779519389922571) q[9];
rz(-0.14624065185560386) q[9];
ry(-0.3169746280499339) q[10];
rz(-0.441864842963251) q[10];
ry(0.0016733175716856505) q[11];
rz(-0.9866958131352271) q[11];
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
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.334108628852785) q[0];
rz(-1.373282824784569) q[0];
ry(2.1130403105953715) q[1];
rz(2.5104349508022508) q[1];
ry(2.2178303676297437) q[2];
rz(-2.969285401770493) q[2];
ry(0.24721029787833856) q[3];
rz(1.2133386868533869) q[3];
ry(0.059561572112689594) q[4];
rz(0.7285274120069349) q[4];
ry(1.2828710081876407) q[5];
rz(0.7202767961156169) q[5];
ry(-3.072553633836251) q[6];
rz(-0.3436926305909936) q[6];
ry(-3.016090323514889) q[7];
rz(-2.384134054032833) q[7];
ry(0.6028056499383752) q[8];
rz(1.9926708007218519) q[8];
ry(0.8552450673279202) q[9];
rz(2.072414178371038) q[9];
ry(-2.2194655808754815) q[10];
rz(-0.5453991950184646) q[10];
ry(1.6148591031422406) q[11];
rz(1.0131486855310736) q[11];