OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.4448158354646132) q[0];
rz(2.0475546154493847) q[0];
ry(1.4109302828396642) q[1];
rz(0.37368788740548015) q[1];
ry(-2.268524599792724) q[2];
rz(-2.9462148924463443) q[2];
ry(1.7309787826192462) q[3];
rz(-2.4335427801494998) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.838512094144886) q[0];
rz(-0.989843344900088) q[0];
ry(-0.8945607462481827) q[1];
rz(0.6622215929438435) q[1];
ry(0.9935557438936422) q[2];
rz(0.6012831684780815) q[2];
ry(1.7676896616847717) q[3];
rz(2.7758029328874665) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.5778155524811144) q[0];
rz(-1.403377209405629) q[0];
ry(-2.4462739554656836) q[1];
rz(-1.5045705658387885) q[1];
ry(-2.468720037454098) q[2];
rz(0.7544413201491799) q[2];
ry(0.19612233004707372) q[3];
rz(2.306712452074781) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.6523099643790036) q[0];
rz(0.8764322260244102) q[0];
ry(-2.2329695727130483) q[1];
rz(-1.6889222045729602) q[1];
ry(-0.37979378464667324) q[2];
rz(-1.7278280056386564) q[2];
ry(-0.31129271171577305) q[3];
rz(-1.3069460328987865) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.9196507053121056) q[0];
rz(0.40075149829108536) q[0];
ry(0.28249600123583585) q[1];
rz(2.127638084376203) q[1];
ry(-0.24650959300083033) q[2];
rz(-1.6660985732326952) q[2];
ry(1.1959158932061253) q[3];
rz(0.8971979892386034) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.0298516416449197) q[0];
rz(-2.991123929122202) q[0];
ry(1.8589138137034116) q[1];
rz(-0.8793884946018335) q[1];
ry(2.5106992421766625) q[2];
rz(0.3783999754096994) q[2];
ry(-2.4346785507383455) q[3];
rz(0.05541267924548432) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.900003522986901) q[0];
rz(-2.824261091137709) q[0];
ry(0.9831212981810227) q[1];
rz(-0.48435338782903564) q[1];
ry(-2.7670167403812798) q[2];
rz(2.466654052214878) q[2];
ry(-1.9078304583072851) q[3];
rz(2.22093452196673) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.043132929080313) q[0];
rz(-2.1991003833725458) q[0];
ry(2.921173007928446) q[1];
rz(1.691517509557496) q[1];
ry(-1.1020862238228568) q[2];
rz(-1.4844755091572424) q[2];
ry(-3.0390655583379353) q[3];
rz(-0.9783953637794108) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.5053110012215978) q[0];
rz(1.7174859086297938) q[0];
ry(-2.865538614095215) q[1];
rz(-1.877866884365478) q[1];
ry(-0.14363109584686562) q[2];
rz(-1.5535969674443861) q[2];
ry(2.1424501487791785) q[3];
rz(-0.28982682260965575) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.12327735551355623) q[0];
rz(-2.8685352016719152) q[0];
ry(1.2762651146001645) q[1];
rz(0.14816351083824691) q[1];
ry(-2.0495119154265944) q[2];
rz(2.830545538173314) q[2];
ry(0.8584984501401874) q[3];
rz(0.38055310140105286) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.0186572568946508) q[0];
rz(1.875982039672091) q[0];
ry(2.522373256619384) q[1];
rz(-2.9635404389486415) q[1];
ry(-2.947691563059873) q[2];
rz(1.967892236659) q[2];
ry(-0.6512725903639445) q[3];
rz(1.2818796756882918) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.04471222874798148) q[0];
rz(-0.13791217457545987) q[0];
ry(-0.7466830941569063) q[1];
rz(0.817460240233251) q[1];
ry(-2.2447288731131865) q[2];
rz(0.6764985578337969) q[2];
ry(-0.4749971871215619) q[3];
rz(-1.3505547415526993) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.7501674462977169) q[0];
rz(-2.8739498395983114) q[0];
ry(2.475249686765776) q[1];
rz(0.5112503357774028) q[1];
ry(3.115600850790646) q[2];
rz(-1.0767797721953458) q[2];
ry(-0.0020545535269249626) q[3];
rz(-2.817412468252543) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.2143992359296711) q[0];
rz(-2.3490178156089896) q[0];
ry(0.6846354383715515) q[1];
rz(1.1271801210511037) q[1];
ry(2.288814935743456) q[2];
rz(1.7038789863242085) q[2];
ry(1.6443587118222789) q[3];
rz(1.4813803161706236) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.3658770144870291) q[0];
rz(0.11727518301382833) q[0];
ry(0.6823304768336405) q[1];
rz(2.280253719857297) q[1];
ry(0.1392855633948145) q[2];
rz(0.8163675486980496) q[2];
ry(2.598222394691322) q[3];
rz(0.20364963175074816) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.3918391689782315) q[0];
rz(-2.3470061869998546) q[0];
ry(0.41226568772567257) q[1];
rz(1.856693184500286) q[1];
ry(2.810397805129114) q[2];
rz(1.0917135292077418) q[2];
ry(2.477566536626902) q[3];
rz(0.6019592362112132) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.9833885881560698) q[0];
rz(2.538295537530544) q[0];
ry(0.5773529036775317) q[1];
rz(-2.5094007229539446) q[1];
ry(0.6991942118203407) q[2];
rz(2.179152629787855) q[2];
ry(-0.6806998159708608) q[3];
rz(-0.05212384996094421) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.6497796296632794) q[0];
rz(-1.0778832493956179) q[0];
ry(-1.3471797663214993) q[1];
rz(2.0028593031558697) q[1];
ry(0.12203827731360305) q[2];
rz(2.496641686563324) q[2];
ry(-0.7062741130140688) q[3];
rz(-3.0061191005833146) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.0983084761419986) q[0];
rz(1.6122564147121994) q[0];
ry(-0.8530845065904116) q[1];
rz(0.29433262656309195) q[1];
ry(1.0427274650668414) q[2];
rz(2.1841387857909114) q[2];
ry(2.9466745665000205) q[3];
rz(1.3876156809838782) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-1.2677904832039664) q[0];
rz(1.6312317773027196) q[0];
ry(-1.82706632440055) q[1];
rz(-0.09320328284472978) q[1];
ry(-3.0629124490848327) q[2];
rz(-2.578232758167651) q[2];
ry(-0.5435983955309148) q[3];
rz(0.6120772803261483) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.7028872607230918) q[0];
rz(-0.24708026785885814) q[0];
ry(-0.3413025624743647) q[1];
rz(-2.1820063258721474) q[1];
ry(-0.5542620526450026) q[2];
rz(-2.5079008158149017) q[2];
ry(0.1772417097688131) q[3];
rz(0.3542534533538992) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.2655521937829861) q[0];
rz(0.5536527986758418) q[0];
ry(-2.3865330382181758) q[1];
rz(-0.27828218697912993) q[1];
ry(-2.0763733493627727) q[2];
rz(1.941655433041273) q[2];
ry(2.048328958433263) q[3];
rz(0.013742281074239138) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-3.1309122818145747) q[0];
rz(0.5781041895745113) q[0];
ry(2.29285492527369) q[1];
rz(0.02479763711952288) q[1];
ry(-1.6326206447609248) q[2];
rz(-0.6579257034957777) q[2];
ry(0.34265708249916926) q[3];
rz(-0.5899757259594649) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.5034393417419345) q[0];
rz(1.249334817206447) q[0];
ry(1.4491777270563098) q[1];
rz(1.278645132524887) q[1];
ry(-2.4283559244790354) q[2];
rz(-2.756901118572522) q[2];
ry(-2.266178373298969) q[3];
rz(2.234528013313869) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-2.76172686125652) q[0];
rz(2.0715927725987635) q[0];
ry(-2.288260107486295) q[1];
rz(-2.0823437759869146) q[1];
ry(0.07165510455242519) q[2];
rz(-1.9412562209402244) q[2];
ry(-1.4423134808333429) q[3];
rz(-0.8742432290109922) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.6027610987042635) q[0];
rz(-2.7993457033997395) q[0];
ry(-1.6313280902488572) q[1];
rz(-2.795866629258888) q[1];
ry(2.658340917462492) q[2];
rz(0.6132530691156545) q[2];
ry(-0.7646857439720064) q[3];
rz(1.7224006764571813) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(-0.943679085329155) q[0];
rz(1.3245453422533586) q[0];
ry(-0.42690274752099877) q[1];
rz(0.04673338269620286) q[1];
ry(-1.7765380749311293) q[2];
rz(1.3739277835387442) q[2];
ry(1.9330968243710636) q[3];
rz(-0.042825973723453294) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(1.9396443041469238) q[0];
rz(1.5798410164735048) q[0];
ry(-1.5135087210684852) q[1];
rz(2.760488602599672) q[1];
ry(0.06087609309139364) q[2];
rz(-3.045988559973335) q[2];
ry(1.2836860271149815) q[3];
rz(-2.575108259667469) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(2.141293775705) q[0];
rz(2.5836270240807804) q[0];
ry(-1.1969049602996558) q[1];
rz(-1.3711092994687615) q[1];
ry(0.2624972800512593) q[2];
rz(2.3521716625088547) q[2];
ry(2.3583793746953243) q[3];
rz(-0.6998101768904598) q[3];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[1],q[2];
cz q[1],q[3];
cz q[2],q[3];
ry(0.2684455202429326) q[0];
rz(1.803967889191082) q[0];
ry(2.6951594351584456) q[1];
rz(1.2642844460994103) q[1];
ry(-2.41527246498947) q[2];
rz(-1.51800053252389) q[2];
ry(2.398347028036712) q[3];
rz(-1.719171081553366) q[3];