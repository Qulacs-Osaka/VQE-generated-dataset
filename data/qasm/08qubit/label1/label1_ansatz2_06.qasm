OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.47524341079292753) q[0];
rz(2.963389151832437) q[0];
ry(3.0595434997553865) q[1];
rz(-1.2839156085040535) q[1];
ry(-2.5021929608070854) q[2];
rz(2.8016162739064416) q[2];
ry(-1.1149663723970906) q[3];
rz(-2.6292273233745322) q[3];
ry(-6.449533440967734e-05) q[4];
rz(-0.9340755142736621) q[4];
ry(-2.325165307148986) q[5];
rz(-0.7767389546067915) q[5];
ry(-1.5679640749644363) q[6];
rz(0.13653988083708501) q[6];
ry(1.5660120337445977) q[7];
rz(3.1310061579243498) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.8032811999976157) q[0];
rz(-0.736129369428313) q[0];
ry(-1.5059075896739977) q[1];
rz(1.639609542496845) q[1];
ry(-1.8990130035396604) q[2];
rz(-1.793951357436943) q[2];
ry(3.068838621942758) q[3];
rz(-0.15229923782039495) q[3];
ry(-1.9715292902233728e-05) q[4];
rz(2.2822350857999743) q[4];
ry(3.1321496946170244) q[5];
rz(1.6648367643276962) q[5];
ry(0.005699125908921044) q[6];
rz(-0.8753686165500262) q[6];
ry(2.3151101808634267) q[7];
rz(-1.5630040736024235) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.5053014000719256) q[0];
rz(0.07575865922137393) q[0];
ry(-2.0350360862296104) q[1];
rz(-1.496148964187671) q[1];
ry(-1.3422626637172752) q[2];
rz(0.04374263549015911) q[2];
ry(-2.3718056844659) q[3];
rz(0.7706148873923132) q[3];
ry(8.637143447085005e-05) q[4];
rz(0.895877440968638) q[4];
ry(-3.1346003803577758) q[5];
rz(-1.9593938015540429) q[5];
ry(0.004469286958686425) q[6];
rz(-0.9218220858822903) q[6];
ry(-2.824995969832507) q[7];
rz(1.5967745919548992) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.5007722985681573) q[0];
rz(0.1891963382145594) q[0];
ry(-1.6213219767901599) q[1];
rz(2.8663771345413958) q[1];
ry(-1.6594954794085695) q[2];
rz(-0.03341850082525077) q[2];
ry(1.561897293169259) q[3];
rz(-1.5782572291021564) q[3];
ry(1.4668537966948225e-05) q[4];
rz(-1.4231575582146847) q[4];
ry(3.130930213735179) q[5];
rz(2.273853445621075) q[5];
ry(3.1405962707182127) q[6];
rz(-2.5114710689517294) q[6];
ry(-0.7896676320583875) q[7];
rz(1.5655750148177578) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(-1.832494478209769) q[0];
rz(-1.8749629542808814) q[0];
ry(1.806912469225376) q[1];
rz(-1.3085312828992395) q[1];
ry(-0.8884384044716596) q[2];
rz(1.152663353032088) q[2];
ry(-1.757284655944606) q[3];
rz(0.24015139127233046) q[3];
ry(1.1100807462582907e-05) q[4];
rz(0.3985611778241704) q[4];
ry(-0.002334909490888215) q[5];
rz(-1.9755005845459273) q[5];
ry(-3.1386544778824494) q[6];
rz(1.8856005959181807) q[6];
ry(1.549499419791645) q[7];
rz(1.5375672584226192) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(1.9021057008869962) q[0];
rz(0.5153790786497764) q[0];
ry(0.7319972475395121) q[1];
rz(-2.925651148212202) q[1];
ry(-0.569317303909239) q[2];
rz(-1.5192750784716973) q[2];
ry(-0.7478489396593208) q[3];
rz(0.6645881733156278) q[3];
ry(3.141453887955481) q[4];
rz(2.2018234215234163) q[4];
ry(1.4942467476951353) q[5];
rz(-1.46290121754949) q[5];
ry(-0.1298867481603727) q[6];
rz(2.4963564592662273) q[6];
ry(0.8125638262757052) q[7];
rz(-1.720576746853111) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.138260848844258) q[0];
rz(2.309893179405737) q[0];
ry(3.1357125116106266) q[1];
rz(2.8015667421821266) q[1];
ry(3.124551077864562) q[2];
rz(0.1889771136056373) q[2];
ry(3.0710636732430983) q[3];
rz(-1.6155245467039796) q[3];
ry(-3.141576223912188) q[4];
rz(1.5080835572392874) q[4];
ry(-2.9249689832175734) q[5];
rz(2.2866602329107852) q[5];
ry(-0.32311331426233725) q[6];
rz(-2.031792527411769) q[6];
ry(-2.5366314133711376) q[7];
rz(-1.8796228379893039) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(0.024075497795310596) q[0];
rz(0.48006093395289345) q[0];
ry(-3.078905227039206) q[1];
rz(-2.90945694602382) q[1];
ry(3.1269808189688026) q[2];
rz(-2.1207445274033168) q[2];
ry(2.9859697752979875) q[3];
rz(2.6823755271167413) q[3];
ry(1.5708908595504831) q[4];
rz(-0.00012543885849058967) q[4];
ry(1.5119424703248354) q[5];
rz(1.9831564114729474) q[5];
ry(-0.27614996157993676) q[6];
rz(-1.1238011297774813) q[6];
ry(-0.6845099294652375) q[7];
rz(2.7317380986319244) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(8.210369557239943e-05) q[0];
rz(-0.9497412990210468) q[0];
ry(-3.141376159355687) q[1];
rz(-2.3883200353087717) q[1];
ry(-4.768698918589109e-05) q[2];
rz(1.8751484763307755) q[2];
ry(-3.8477571436601465e-05) q[3];
rz(-1.854389212256433) q[3];
ry(1.5708711006894929) q[4];
rz(1.6386413909993376) q[4];
ry(7.069085127290982e-05) q[5];
rz(1.8288979788884712) q[5];
ry(1.573742351393961) q[6];
rz(-1.5692800814306715) q[6];
ry(-1.56935490725313) q[7];
rz(1.5678104248052218) q[7];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[5],q[6];
cz q[5],q[7];
cz q[6],q[7];
ry(3.1415595656014053) q[0];
rz(0.7521616594284816) q[0];
ry(-3.1415552131756073) q[1];
rz(-0.6893471085406293) q[1];
ry(1.492642606670813e-05) q[2];
rz(-1.4105135984957613) q[2];
ry(2.378250334205915e-05) q[3];
rz(1.7247436246325305) q[3];
ry(-3.1415785860542327) q[4];
rz(0.8614481765921201) q[4];
ry(0.0001616400610942037) q[5];
rz(0.0559823003770053) q[5];
ry(1.5706782732456162) q[6];
rz(2.344678572067239) q[6];
ry(1.5708634061321585) q[7];
rz(-0.7910972955844289) q[7];