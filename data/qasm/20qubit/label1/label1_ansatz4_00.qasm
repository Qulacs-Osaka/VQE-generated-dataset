OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(1.7066244250897866) q[0];
rz(-0.11652870350413083) q[0];
ry(-0.3576558495943676) q[1];
rz(0.0551770218578688) q[1];
ry(0.29284961150662525) q[2];
rz(-3.036302205649162) q[2];
ry(0.0025469435996496865) q[3];
rz(2.4465568890081237) q[3];
ry(0.005900425705435742) q[4];
rz(-1.215115297085438) q[4];
ry(0.020378311031366003) q[5];
rz(1.7140893584185175) q[5];
ry(1.3875852284038375) q[6];
rz(-2.5657118841490933) q[6];
ry(1.4581030495260496) q[7];
rz(-1.1532266323230171) q[7];
ry(-2.9588494954426636) q[8];
rz(-0.3167976968844664) q[8];
ry(3.0823895883198853) q[9];
rz(-2.0327407077013615) q[9];
ry(0.003944716370891221) q[10];
rz(-2.563929843568773) q[10];
ry(-3.1128224709278216) q[11];
rz(-1.4799456178612207) q[11];
ry(1.1921915932205733) q[12];
rz(-0.007131208851875569) q[12];
ry(-1.089470953662886) q[13];
rz(3.137168412829173) q[13];
ry(0.024699403207948393) q[14];
rz(-0.8670740109250478) q[14];
ry(3.111982450897644) q[15];
rz(-1.4206134426417512) q[15];
ry(2.7062451498590527) q[16];
rz(-1.5461960081952077) q[16];
ry(-0.08009645078538279) q[17];
rz(-1.8624456710694464) q[17];
ry(1.7457147092694885) q[18];
rz(0.8031332907463531) q[18];
ry(-2.4641154352278627) q[19];
rz(1.1479654589258486) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.598417548472808) q[0];
rz(0.004135126095340947) q[0];
ry(-1.513072660662606) q[1];
rz(-3.136371073207959) q[1];
ry(-2.44997512978947) q[2];
rz(-0.09382053410743475) q[2];
ry(0.002156836888156377) q[3];
rz(-2.358948960343572) q[3];
ry(-1.6310464806770126) q[4];
rz(-2.6765957682737906) q[4];
ry(-0.2781170739868587) q[5];
rz(-3.0723227834407463) q[5];
ry(-1.3101738137361414) q[6];
rz(-1.4033755810048847) q[6];
ry(-2.563175715621743) q[7];
rz(-1.492244771514154) q[7];
ry(0.496809550497189) q[8];
rz(2.1560485722206817) q[8];
ry(-2.1231029396329477) q[9];
rz(-2.209625303721615) q[9];
ry(2.146124104536943) q[10];
rz(-2.0183368626281073) q[10];
ry(-2.7028465579313967) q[11];
rz(1.7116022124273247) q[11];
ry(-0.8707977086339288) q[12];
rz(-1.4508249091905245) q[12];
ry(2.533998953071863) q[13];
rz(-1.2458013714467184) q[13];
ry(-2.6439751181763484) q[14];
rz(-0.29119587942646413) q[14];
ry(1.5181083213965174) q[15];
rz(0.029857468758225814) q[15];
ry(-2.2598947236619784) q[16];
rz(-3.0582874939406333) q[16];
ry(-0.17535242368544512) q[17];
rz(2.9601492350807144) q[17];
ry(-1.5994613926958605) q[18];
rz(1.6376303104278298) q[18];
ry(-0.2669699197037172) q[19];
rz(0.1911500220141118) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.0522102582053945) q[0];
rz(3.035867648776114) q[0];
ry(2.0466799109887157) q[1];
rz(0.04468596409077197) q[1];
ry(2.8564563099785985) q[2];
rz(-0.013356236756424111) q[2];
ry(-1.5142781097669675) q[3];
rz(-3.140867595432309) q[3];
ry(0.14202293861966456) q[4];
rz(-0.4141374161527338) q[4];
ry(2.953149181698081) q[5];
rz(-0.03409695647864564) q[5];
ry(-3.1415765217933256) q[6];
rz(-1.0030989501418714) q[6];
ry(-3.1357337722634275) q[7];
rz(3.024552319035016) q[7];
ry(0.36477283544033146) q[8];
rz(0.2902315189737923) q[8];
ry(-0.9400589041466247) q[9];
rz(-2.4496859165488964) q[9];
ry(0.715377767502383) q[10];
rz(0.6145313424405137) q[10];
ry(0.26282301367367555) q[11];
rz(0.5036808148952527) q[11];
ry(3.0985514699687147) q[12];
rz(3.0476962377228602) q[12];
ry(-3.115422444389292) q[13];
rz(-3.022229326803198) q[13];
ry(-3.117911941578674) q[14];
rz(-0.31147987336565874) q[14];
ry(1.2152224665965177) q[15];
rz(3.084170110449538) q[15];
ry(0.17598741333268997) q[16];
rz(-0.0735137733668898) q[16];
ry(1.3459702171763392) q[17];
rz(3.1302400506137653) q[17];
ry(0.7891204469186978) q[18];
rz(1.5729723426968558) q[18];
ry(-2.9543997733501963) q[19];
rz(2.595168632311583) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.1314093994974503) q[0];
rz(0.48907858336731014) q[0];
ry(-3.122655739464558) q[1];
rz(0.48658118658065697) q[1];
ry(3.1279962492459745) q[2];
rz(-2.6663958653287865) q[2];
ry(-1.598989713106782) q[3];
rz(-2.651071651396429) q[3];
ry(3.1337463209061553) q[4];
rz(0.5422750157947905) q[4];
ry(-0.06915797680109285) q[5];
rz(0.5410492764639976) q[5];
ry(0.10557332560162512) q[6];
rz(-2.505160644984282) q[6];
ry(3.059956799888423) q[7];
rz(1.1929307345810438) q[7];
ry(-0.42916744993738565) q[8];
rz(-1.9308443700640812) q[8];
ry(-2.355694037818362) q[9];
rz(-1.2825919615112724) q[9];
ry(0.7684417741743261) q[10];
rz(-0.38485324317181124) q[10];
ry(2.5478772786554753) q[11];
rz(0.20278399232690258) q[11];
ry(-1.3915393342398314) q[12];
rz(-2.619666440417872) q[12];
ry(1.7530500211533933) q[13];
rz(0.5187041209043013) q[13];
ry(-2.764941635631797) q[14];
rz(0.5091663413085594) q[14];
ry(-0.5313865961291937) q[15];
rz(-2.538499612666761) q[15];
ry(2.749577929839718) q[16];
rz(-2.603959512040109) q[16];
ry(-1.6832277170452903) q[17];
rz(-2.5146673704392466) q[17];
ry(0.02383163688804416) q[18];
rz(2.3694382493450132) q[18];
ry(-3.102713885552656) q[19];
rz(-2.7281374696819407) q[19];