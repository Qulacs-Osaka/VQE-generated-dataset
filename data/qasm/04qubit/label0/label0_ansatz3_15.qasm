OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(3.10739191263123) q[0];
rz(2.2314843735784997) q[0];
ry(-0.6494658080208324) q[1];
rz(1.2881034972299983) q[1];
ry(-0.2486610614068377) q[2];
rz(-0.11460484383615643) q[2];
ry(-0.6726989954104896) q[3];
rz(0.9859895222809755) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.69901347493347) q[0];
rz(0.8100371150697048) q[0];
ry(-0.516970691314036) q[1];
rz(-1.2232463443308286) q[1];
ry(0.9725269217270773) q[2];
rz(-3.011575256428344) q[2];
ry(2.2423725349008876) q[3];
rz(0.5042523726687742) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.0044107283189425) q[0];
rz(-1.0271054202253003) q[0];
ry(0.29184933495166204) q[1];
rz(-2.708229968977854) q[1];
ry(1.4213509956162005) q[2];
rz(-2.2079629409268917) q[2];
ry(0.3793395949479453) q[3];
rz(1.144238504767972) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.5192891028630813) q[0];
rz(1.6835733036064067) q[0];
ry(1.7193494117246786) q[1];
rz(1.7170846717962487) q[1];
ry(0.7116193922949564) q[2];
rz(0.31987861553689173) q[2];
ry(0.9768111929472738) q[3];
rz(2.3400133614995786) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.6312615543477489) q[0];
rz(0.6896701881689258) q[0];
ry(2.127772620356559) q[1];
rz(0.3550555741318995) q[1];
ry(1.7075400297105165) q[2];
rz(-2.2746815677354437) q[2];
ry(0.8746432203601371) q[3];
rz(2.0682872662779275) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.625899111230104) q[0];
rz(1.077813166539699) q[0];
ry(2.50672711028141) q[1];
rz(0.576741196987081) q[1];
ry(-2.9934715994802077) q[2];
rz(2.5491709102900266) q[2];
ry(0.6706136046807941) q[3];
rz(-1.9920791775834532) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.5875700609363609) q[0];
rz(2.216763591632457) q[0];
ry(-0.8100740273376086) q[1];
rz(-0.03982282157620164) q[1];
ry(2.7266891822769423) q[2];
rz(-1.0239075907537378) q[2];
ry(-1.0177906054909087) q[3];
rz(2.5082614601322804) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.18011697026435328) q[0];
rz(2.8725427396523315) q[0];
ry(-0.9774253484469799) q[1];
rz(-2.9526497722866467) q[1];
ry(-1.7771675984422288) q[2];
rz(1.5293833867431017) q[2];
ry(-1.3673716932074216) q[3];
rz(0.9693496689256668) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.3954056906173717) q[0];
rz(2.4237708447990003) q[0];
ry(2.8146749318733426) q[1];
rz(1.3474061773525623) q[1];
ry(-1.166870308407888) q[2];
rz(1.854589982570296) q[2];
ry(-1.6372007309043168) q[3];
rz(-0.2545560527533173) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-3.0660576480159807) q[0];
rz(-2.6153691851229834) q[0];
ry(-0.7056274354288257) q[1];
rz(1.0636078815945886) q[1];
ry(0.6131297169056937) q[2];
rz(-0.3197095859799441) q[2];
ry(-0.1664070913198774) q[3];
rz(-2.447137857423649) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.5946221939113165) q[0];
rz(-2.433510440560663) q[0];
ry(0.49268305577611726) q[1];
rz(0.6144615204188123) q[1];
ry(1.090194652014774) q[2];
rz(0.1515064913583024) q[2];
ry(-2.743185236606735) q[3];
rz(0.17037903744012614) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.5243452584444594) q[0];
rz(1.9699682655389326) q[0];
ry(1.683115188737192) q[1];
rz(-0.12202657206730816) q[1];
ry(2.745318049812062) q[2];
rz(0.9577411373449394) q[2];
ry(2.5237345060882754) q[3];
rz(-0.5007799898394713) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.797812622009899) q[0];
rz(-0.6679948023374084) q[0];
ry(0.5637422426737597) q[1];
rz(-0.8289844960518175) q[1];
ry(1.1849794422733204) q[2];
rz(1.0054057908870968) q[2];
ry(1.3995210996299674) q[3];
rz(1.0994300398326882) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.34714335103416) q[0];
rz(-2.8873591505218013) q[0];
ry(-2.512855001700626) q[1];
rz(2.470203344882291) q[1];
ry(-2.9193473317511596) q[2];
rz(1.9658181316925507) q[2];
ry(-2.2869228385708418) q[3];
rz(-1.8780055845594354) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.10347523095253042) q[0];
rz(-2.1677493466664997) q[0];
ry(-1.2254153091094548) q[1];
rz(-1.2178036756851212) q[1];
ry(-0.698319877199909) q[2];
rz(-2.4584565505691875) q[2];
ry(-1.5822530991457988) q[3];
rz(-2.5202426909677644) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.9010415634036932) q[0];
rz(1.0017134467917008) q[0];
ry(-1.0883711961330103) q[1];
rz(2.1497538929481808) q[1];
ry(-0.21591020778901596) q[2];
rz(1.8859943379962285) q[2];
ry(-0.7554641691042444) q[3];
rz(0.308076709246792) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.227231168873846) q[0];
rz(1.2444312994851972) q[0];
ry(1.9912237095186436) q[1];
rz(-0.3214786845713873) q[1];
ry(2.5537022685279513) q[2];
rz(1.1934980329063727) q[2];
ry(-1.2890902618640057) q[3];
rz(2.804415037832937) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.08451770765312006) q[0];
rz(3.0679789552935066) q[0];
ry(1.0780524407990217) q[1];
rz(-2.1849933548764398) q[1];
ry(1.8585380897759407) q[2];
rz(2.367251076196479) q[2];
ry(-1.2138035312953503) q[3];
rz(-1.5290548175609038) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.21192793555380796) q[0];
rz(2.0964316123641877) q[0];
ry(2.4062979638618147) q[1];
rz(-1.4894361948626806) q[1];
ry(0.12654566234716183) q[2];
rz(1.3667766852083822) q[2];
ry(2.363489003702874) q[3];
rz(1.4452872782632984) q[3];