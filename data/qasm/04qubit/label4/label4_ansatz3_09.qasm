OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(2.5553514036737504) q[0];
rz(-0.028047743457054766) q[0];
ry(0.39139542235100677) q[1];
rz(-2.8945380486343675) q[1];
ry(2.1114842608001148) q[2];
rz(0.6466329521950209) q[2];
ry(1.1270932029048804) q[3];
rz(1.927937140127935) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.508131065039971) q[0];
rz(-1.4766992728876138) q[0];
ry(-1.3676633055374428) q[1];
rz(-0.47994966473925216) q[1];
ry(-2.222569670349756) q[2];
rz(-1.5170354557519943) q[2];
ry(-2.4192782945553954) q[3];
rz(-1.112633916954777) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.842458652184947) q[0];
rz(2.6966237517853893) q[0];
ry(-1.3759131030148795) q[1];
rz(-0.5859505415856932) q[1];
ry(1.564293507738309) q[2];
rz(-2.707038832443232) q[2];
ry(-1.7949519913640588) q[3];
rz(-2.444442342801437) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.1265131623433247) q[0];
rz(-2.7013540522537536) q[0];
ry(-2.7609905393822367) q[1];
rz(-2.9467602108226467) q[1];
ry(2.8341906817596487) q[2];
rz(2.714163323098458) q[2];
ry(-2.6695012616005527) q[3];
rz(-2.5447046278004883) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.0850345140847364) q[0];
rz(-1.9680486353808497) q[0];
ry(1.3749690953840625) q[1];
rz(0.3120013638446393) q[1];
ry(-0.002199232212880986) q[2];
rz(1.4812927136103413) q[2];
ry(-3.029855287200478) q[3];
rz(-0.6291627945160964) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.1222976512673641) q[0];
rz(2.0918975914467497) q[0];
ry(-0.019282394263244237) q[1];
rz(-0.6326881400497947) q[1];
ry(0.9646943583685799) q[2];
rz(-1.9644076257796508) q[2];
ry(2.329327586253515) q[3];
rz(0.714200308276452) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.988276630579357) q[0];
rz(2.9441228588967276) q[0];
ry(-2.7529795604223515) q[1];
rz(1.6566424618755073) q[1];
ry(2.427311499392099) q[2];
rz(-1.5681265576931442) q[2];
ry(2.982434106451796) q[3];
rz(-1.1698335518609326) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.539603499210526) q[0];
rz(1.972155164376498) q[0];
ry(-2.530334206252881) q[1];
rz(0.5270700905444382) q[1];
ry(-2.802537221030959) q[2];
rz(2.2150585896601784) q[2];
ry(0.7277381639056106) q[3];
rz(1.716766342073967) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.0736351144350813) q[0];
rz(-2.5005806869188834) q[0];
ry(2.156386596206344) q[1];
rz(-2.4595742451379863) q[1];
ry(1.9214233137450172) q[2];
rz(0.7316020509560556) q[2];
ry(-2.557222654126182) q[3];
rz(-1.7404318534376964) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.22502537233457967) q[0];
rz(0.2306896371630412) q[0];
ry(-2.927230484126929) q[1];
rz(0.8284414656372424) q[1];
ry(0.34335630177200915) q[2];
rz(-2.302311661402151) q[2];
ry(0.021083072256343094) q[3];
rz(0.36431877745758623) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.1009896375369412) q[0];
rz(-1.002048598797756) q[0];
ry(-1.6580526643066746) q[1];
rz(-0.9033712085207141) q[1];
ry(2.7306418151198404) q[2];
rz(1.0293827526286647) q[2];
ry(-0.8420858082837359) q[3];
rz(-0.6992573537921309) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(3.030657418005725) q[0];
rz(-0.6078223761758964) q[0];
ry(2.2451417333139285) q[1];
rz(-1.3059251796827933) q[1];
ry(0.0017558719191595774) q[2];
rz(2.529250088530544) q[2];
ry(-0.8459352248158004) q[3];
rz(0.4148955557315404) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.44607818517262654) q[0];
rz(0.6638719082957509) q[0];
ry(0.14116788430657312) q[1];
rz(1.672723432689102) q[1];
ry(-1.5193536349888472) q[2];
rz(-2.2088812128042346) q[2];
ry(-3.0774232807936714) q[3];
rz(-1.9493748418712507) q[3];