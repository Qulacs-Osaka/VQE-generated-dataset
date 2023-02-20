OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.9595218885499408) q[0];
rz(-0.5404892508405942) q[0];
ry(1.9509316751666537) q[1];
rz(0.19307841968569295) q[1];
ry(2.301528602660683) q[2];
rz(-1.1552602086650925) q[2];
ry(-2.994374834978498) q[3];
rz(0.07778605155863083) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.695082401279274) q[0];
rz(-0.17212414002770426) q[0];
ry(2.38518972299566) q[1];
rz(-1.7543508651565745) q[1];
ry(1.7274250534638034) q[2];
rz(-2.8089994422296347) q[2];
ry(1.8554053894670075) q[3];
rz(0.7817067145501655) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.218955994021217) q[0];
rz(-0.05769095775029776) q[0];
ry(0.39767894321492836) q[1];
rz(1.1901981364935206) q[1];
ry(-2.1089253805662462) q[2];
rz(-0.027009389452874767) q[2];
ry(-3.1367797717687718) q[3];
rz(2.278256927183249) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.6331617829071226) q[0];
rz(2.754775590502366) q[0];
ry(-0.6313476788546308) q[1];
rz(0.36115293931492237) q[1];
ry(0.2650927401379759) q[2];
rz(0.8452060885404152) q[2];
ry(-0.12541048036122238) q[3];
rz(2.9423479705629303) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.7479834513181043) q[0];
rz(-0.3581757781182446) q[0];
ry(2.0893578097113608) q[1];
rz(0.6438219859239993) q[1];
ry(-0.3732236545917109) q[2];
rz(1.1145620379846894) q[2];
ry(1.2279897549059573) q[3];
rz(1.2129443308577321) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.24050531505712) q[0];
rz(1.358055755893987) q[0];
ry(2.5162384643363755) q[1];
rz(-1.9064448188925125) q[1];
ry(2.41063231642148) q[2];
rz(-0.2881918454248821) q[2];
ry(-2.043540630778338) q[3];
rz(-2.610847324090087) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.4011440401708195) q[0];
rz(2.222807194166833) q[0];
ry(-0.4623783180552333) q[1];
rz(-1.0633055895208416) q[1];
ry(1.2153980612372068) q[2];
rz(1.4991201901050426) q[2];
ry(1.322438934827662) q[3];
rz(-1.2743494021975994) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.724913920305012) q[0];
rz(-0.5021587013117628) q[0];
ry(-2.5456290190142177) q[1];
rz(2.5076795818790405) q[1];
ry(0.5538800489769805) q[2];
rz(0.5003411955223771) q[2];
ry(1.4752965733950933) q[3];
rz(-1.4955300004426733) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.937728001455929) q[0];
rz(-0.1115640037635668) q[0];
ry(-1.4847239578547304) q[1];
rz(0.2575599924271632) q[1];
ry(0.7801832778986375) q[2];
rz(0.8914665157364761) q[2];
ry(-1.2332021974410852) q[3];
rz(0.8858328960692462) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.21552997457967304) q[0];
rz(-0.33824311958160075) q[0];
ry(-2.1398752035635082) q[1];
rz(0.7318784955017508) q[1];
ry(-2.879742968002936) q[2];
rz(1.987187260827004) q[2];
ry(2.719095688549617) q[3];
rz(-1.0393499175688483) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.4349385084698057) q[0];
rz(1.239880118863792) q[0];
ry(-2.750445081281485) q[1];
rz(-2.9742535762478255) q[1];
ry(-2.215148975596696) q[2];
rz(-0.15594871974045435) q[2];
ry(-0.6897990975516359) q[3];
rz(-2.0102623115017555) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.5651109339882208) q[0];
rz(-2.9098265873417164) q[0];
ry(-1.574287647657296) q[1];
rz(1.5945541004929509) q[1];
ry(-0.9474600747553055) q[2];
rz(-2.117921215949722) q[2];
ry(2.2814806872574276) q[3];
rz(1.82292411182332) q[3];