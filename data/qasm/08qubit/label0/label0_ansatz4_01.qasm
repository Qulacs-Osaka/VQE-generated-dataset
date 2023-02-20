OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.0032326143403098584) q[0];
rz(-1.5172013118474297) q[0];
ry(-3.1360470895203556) q[1];
rz(1.664456738800416) q[1];
ry(1.5666018747597281) q[2];
rz(0.03714966947258747) q[2];
ry(2.8982873299257035) q[3];
rz(1.6458643928369874) q[3];
ry(-0.06189907145957996) q[4];
rz(1.521595460248549) q[4];
ry(-3.129164884774858) q[5];
rz(1.547805391412474) q[5];
ry(-1.21300741354022) q[6];
rz(-3.1338765305379535) q[6];
ry(-2.8749478364672587) q[7];
rz(-0.02584309330819146) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.892370807458461) q[0];
rz(-0.009906806868168694) q[0];
ry(-1.6374300211001316) q[1];
rz(0.1258757759595997) q[1];
ry(1.566653599742663) q[2];
rz(1.8416252267604132) q[2];
ry(3.1283123384087492) q[3];
rz(0.16346519363419604) q[3];
ry(-3.1310396992574496) q[4];
rz(3.0675599406809315) q[4];
ry(1.5204154136651686) q[5];
rz(3.134403144338464) q[5];
ry(-0.9721040007449279) q[6];
rz(-0.008958464114442585) q[6];
ry(0.0699084781418694) q[7];
rz(0.021757125287588868) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-3.1266934736493988) q[0];
rz(-0.08686319792226715) q[0];
ry(3.077956876740197) q[1];
rz(-1.4451019411854198) q[1];
ry(-3.114218467451182) q[2];
rz(0.27407673687061257) q[2];
ry(-0.24796568814493153) q[3];
rz(3.0518376705928696) q[3];
ry(1.4070960443594638) q[4];
rz(0.000984498437407666) q[4];
ry(1.2502844498851722) q[5];
rz(-3.1308124134411073) q[5];
ry(0.10927666915901038) q[6];
rz(-3.051655647982147) q[6];
ry(-3.1014776453875434) q[7];
rz(2.8812500730030277) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(3.141515438739739) q[0];
rz(1.5006555838585862) q[0];
ry(-1.4849524743891553) q[1];
rz(-3.13487103606603) q[1];
ry(0.22572281216353662) q[2];
rz(-1.5696832000795264) q[2];
ry(3.066686422152938) q[3];
rz(-1.5711360508707388) q[3];
ry(-1.4054564807356025) q[4];
rz(-3.1404438355477797) q[4];
ry(-0.1758261390959417) q[5];
rz(0.014829030830992239) q[5];
ry(-0.0014719021668012644) q[6];
rz(3.0544794507949193) q[6];
ry(3.138261614288123) q[7];
rz(-0.26004142392836727) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5711604692678973) q[0];
rz(-0.0003069529112886826) q[0];
ry(1.5696432432588483) q[1];
rz(1.5701835130495425) q[1];
ry(1.5665492317977494) q[2];
rz(-0.0002923200132665116) q[2];
ry(1.5984587089659639) q[3];
rz(-0.0003100900095311942) q[3];
ry(-1.6109313842342388) q[4];
rz(3.1398064675315074) q[4];
ry(-1.4168487746948206) q[5];
rz(3.1390186463366074) q[5];
ry(1.5431369127011891) q[6];
rz(3.140639317666852) q[6];
ry(1.5651275175442272) q[7];
rz(-0.0008447888504417682) q[7];