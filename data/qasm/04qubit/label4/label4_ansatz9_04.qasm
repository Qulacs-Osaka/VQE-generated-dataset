OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.8774756725395974) q[0];
ry(0.5930010729915428) q[1];
cx q[0],q[1];
ry(-2.1399374289247897) q[0];
ry(0.40724144534821427) q[1];
cx q[0],q[1];
ry(2.4939047375802836) q[2];
ry(1.23429108898022) q[3];
cx q[2],q[3];
ry(-0.5359930277907976) q[2];
ry(0.7634778256408277) q[3];
cx q[2],q[3];
ry(-2.0759305710323535) q[0];
ry(-1.7895140644387224) q[2];
cx q[0],q[2];
ry(0.5225904440555578) q[0];
ry(-0.3929184602937134) q[2];
cx q[0],q[2];
ry(-2.8250335663589183) q[1];
ry(-2.089127950467639) q[3];
cx q[1],q[3];
ry(0.29100125541668925) q[1];
ry(0.41764495664321366) q[3];
cx q[1],q[3];
ry(-1.5623946411635556) q[0];
ry(-0.5304335100195089) q[3];
cx q[0],q[3];
ry(1.6238251073046897) q[0];
ry(1.3952072973735208) q[3];
cx q[0],q[3];
ry(1.435337370326032) q[1];
ry(2.3603584515453053) q[2];
cx q[1],q[2];
ry(-0.708600908176675) q[1];
ry(-0.7254857642076541) q[2];
cx q[1],q[2];
ry(-0.6076029441014698) q[0];
ry(2.66655671656466) q[1];
cx q[0],q[1];
ry(0.39014686230219287) q[0];
ry(1.0973574192624858) q[1];
cx q[0],q[1];
ry(0.8588461351935877) q[2];
ry(1.845714210685887) q[3];
cx q[2],q[3];
ry(1.0110897802109644) q[2];
ry(2.623450002999288) q[3];
cx q[2],q[3];
ry(3.01750597173602) q[0];
ry(1.6684892537056497) q[2];
cx q[0],q[2];
ry(0.7048056832930039) q[0];
ry(0.017425347590976692) q[2];
cx q[0],q[2];
ry(-0.042172497488706995) q[1];
ry(-0.8936009946615234) q[3];
cx q[1],q[3];
ry(1.6042219272404936) q[1];
ry(0.5575013734023804) q[3];
cx q[1],q[3];
ry(0.6586795871495399) q[0];
ry(2.691005661734384) q[3];
cx q[0],q[3];
ry(-1.5950882637679251) q[0];
ry(1.783718664973572) q[3];
cx q[0],q[3];
ry(0.7094435769351272) q[1];
ry(0.6004110287907158) q[2];
cx q[1],q[2];
ry(-1.3403857864715674) q[1];
ry(1.8631912430409128) q[2];
cx q[1],q[2];
ry(-0.45246373037601395) q[0];
ry(1.6736713809145805) q[1];
cx q[0],q[1];
ry(2.507804316592714) q[0];
ry(-1.4989657237152625) q[1];
cx q[0],q[1];
ry(1.1378775603947333) q[2];
ry(-2.3301568529419443) q[3];
cx q[2],q[3];
ry(-2.3981447221519) q[2];
ry(-2.044776750376629) q[3];
cx q[2],q[3];
ry(-2.8883479023638574) q[0];
ry(2.3713886506229174) q[2];
cx q[0],q[2];
ry(0.4380049545384965) q[0];
ry(-1.1372978825512865) q[2];
cx q[0],q[2];
ry(2.9179916777568784) q[1];
ry(1.1387059029429756) q[3];
cx q[1],q[3];
ry(0.637171957390505) q[1];
ry(3.017821100305203) q[3];
cx q[1],q[3];
ry(-2.155577852597256) q[0];
ry(-3.1202960970679365) q[3];
cx q[0],q[3];
ry(-0.9277604628767617) q[0];
ry(2.5398608131986875) q[3];
cx q[0],q[3];
ry(0.8628860021249114) q[1];
ry(-0.34840734156574715) q[2];
cx q[1],q[2];
ry(-1.9898150163415984) q[1];
ry(-0.7906890206248098) q[2];
cx q[1],q[2];
ry(0.001444309682459565) q[0];
ry(-2.232917071265982) q[1];
cx q[0],q[1];
ry(2.7897235322353438) q[0];
ry(2.5137341835614913) q[1];
cx q[0],q[1];
ry(-0.049786145113581703) q[2];
ry(-1.4345068274451949) q[3];
cx q[2],q[3];
ry(-0.5959028383985012) q[2];
ry(-2.077627436290859) q[3];
cx q[2],q[3];
ry(0.33858917053236703) q[0];
ry(-0.5678658602573996) q[2];
cx q[0],q[2];
ry(-0.34971009708941914) q[0];
ry(-2.8470432390993334) q[2];
cx q[0],q[2];
ry(0.9835077031009565) q[1];
ry(1.227159650237551) q[3];
cx q[1],q[3];
ry(-1.0272729222336339) q[1];
ry(1.2461973935220452) q[3];
cx q[1],q[3];
ry(-0.1393636291837055) q[0];
ry(2.163653197137725) q[3];
cx q[0],q[3];
ry(-1.5555775612586267) q[0];
ry(-0.2626730334790033) q[3];
cx q[0],q[3];
ry(3.1259532630290336) q[1];
ry(1.1357587735346604) q[2];
cx q[1],q[2];
ry(-0.9342253741771636) q[1];
ry(-0.4022989050459885) q[2];
cx q[1],q[2];
ry(-0.9795164144332158) q[0];
ry(2.1021830299437694) q[1];
cx q[0],q[1];
ry(-2.8639544396895102) q[0];
ry(-2.3632347182906135) q[1];
cx q[0],q[1];
ry(-0.8826350746291532) q[2];
ry(-2.5117010049068433) q[3];
cx q[2],q[3];
ry(1.7305238886042726) q[2];
ry(-1.1056630809007406) q[3];
cx q[2],q[3];
ry(-2.2958837029746615) q[0];
ry(-1.4991452525816609) q[2];
cx q[0],q[2];
ry(-2.570069841761134) q[0];
ry(2.595596810372733) q[2];
cx q[0],q[2];
ry(0.14511736812768394) q[1];
ry(2.1540340168216927) q[3];
cx q[1],q[3];
ry(2.043452761544267) q[1];
ry(-0.7050662490908319) q[3];
cx q[1],q[3];
ry(0.8797103325707795) q[0];
ry(1.2262255283980472) q[3];
cx q[0],q[3];
ry(2.0257847442189085) q[0];
ry(-0.293289878548344) q[3];
cx q[0],q[3];
ry(2.914959082824591) q[1];
ry(-1.3782835296591571) q[2];
cx q[1],q[2];
ry(2.5335167667171623) q[1];
ry(2.3718565353900547) q[2];
cx q[1],q[2];
ry(1.1352023322849796) q[0];
ry(-2.1581426985921817) q[1];
cx q[0],q[1];
ry(2.015060623211211) q[0];
ry(1.2294995867729) q[1];
cx q[0],q[1];
ry(-2.0137380086009102) q[2];
ry(-2.6246846308153375) q[3];
cx q[2],q[3];
ry(-0.92706897949029) q[2];
ry(1.8349739561176823) q[3];
cx q[2],q[3];
ry(-1.916324786976376) q[0];
ry(1.9777561837124449) q[2];
cx q[0],q[2];
ry(-2.631202551382032) q[0];
ry(-2.2415403738089026) q[2];
cx q[0],q[2];
ry(1.1601289310274674) q[1];
ry(2.4212183983902333) q[3];
cx q[1],q[3];
ry(-0.06376165495565723) q[1];
ry(-2.7751085882563666) q[3];
cx q[1],q[3];
ry(2.2554393362635405) q[0];
ry(-2.46960530151783) q[3];
cx q[0],q[3];
ry(2.97832242121814) q[0];
ry(0.32950228563501405) q[3];
cx q[0],q[3];
ry(1.052436734058639) q[1];
ry(0.9628057081625028) q[2];
cx q[1],q[2];
ry(0.7448075045892475) q[1];
ry(-2.832438780505238) q[2];
cx q[1],q[2];
ry(-0.31231843707270274) q[0];
ry(-2.494799598136137) q[1];
cx q[0],q[1];
ry(-1.6533581906472536) q[0];
ry(0.45155588774127936) q[1];
cx q[0],q[1];
ry(-0.36915287731045715) q[2];
ry(2.256897188598285) q[3];
cx q[2],q[3];
ry(-1.5800507750264075) q[2];
ry(1.4032750976636645) q[3];
cx q[2],q[3];
ry(-2.1442344829039683) q[0];
ry(-2.9597250271251965) q[2];
cx q[0],q[2];
ry(0.008495242902053057) q[0];
ry(-2.7581036245316395) q[2];
cx q[0],q[2];
ry(-2.0359701488058453) q[1];
ry(1.8562351792936225) q[3];
cx q[1],q[3];
ry(-0.8381457604417444) q[1];
ry(1.8815901934532606) q[3];
cx q[1],q[3];
ry(-0.5228297028359536) q[0];
ry(2.5446711805940763) q[3];
cx q[0],q[3];
ry(-1.5318270187788743) q[0];
ry(2.5728626236828576) q[3];
cx q[0],q[3];
ry(1.6799106281473675) q[1];
ry(0.9191767008813899) q[2];
cx q[1],q[2];
ry(-2.082038214973628) q[1];
ry(-2.716569598234826) q[2];
cx q[1],q[2];
ry(-1.3604734320435055) q[0];
ry(-0.010725240987603963) q[1];
ry(2.6204291467071266) q[2];
ry(-3.082142420432344) q[3];