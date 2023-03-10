OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5707971933481601) q[0];
rz(-1.5708000557391708) q[0];
ry(-1.5707966885706075) q[1];
rz(-1.6168210234841718) q[1];
ry(2.9050318327168974) q[2];
rz(-3.87748871611704e-06) q[2];
ry(-2.184419748003621) q[3];
rz(1.285106902863541) q[3];
ry(2.8556371626644665) q[4];
rz(1.5707917142353656) q[4];
ry(4.5464312314891224e-08) q[5];
rz(0.6376605746605097) q[5];
ry(9.427594754336497e-07) q[6];
rz(-0.11723872423573732) q[6];
ry(-6.398683250403423e-07) q[7];
rz(-2.8030156398138284) q[7];
ry(-3.141592480909705) q[8];
rz(1.8028733977851354) q[8];
ry(-3.141592586533402) q[9];
rz(-1.4631009323501318) q[9];
ry(-3.90206507740128e-07) q[10];
rz(0.8754899598336587) q[10];
ry(-3.1415922922611843) q[11];
rz(0.23663711670824256) q[11];
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
ry(-1.570800828433903) q[0];
rz(0.7706669354282605) q[0];
ry(-3.1415840989483885) q[1];
rz(-0.42795454123114995) q[1];
ry(-1.5707946326759075) q[2];
rz(2.4724400175469) q[2];
ry(-8.70510541275419e-07) q[3];
rz(0.1597201963140007) q[3];
ry(-1.5707963831016734) q[4];
rz(3.1115723926055723) q[4];
ry(-1.5707953788930649) q[5];
rz(2.307543391398131) q[5];
ry(-1.5707945949152686) q[6];
rz(2.040777200562616e-07) q[6];
ry(-0.12674300254589596) q[7];
rz(0.6521990553641976) q[7];
ry(3.6042085692714345e-07) q[8];
rz(-2.509101849662373) q[8];
ry(3.141592585387418) q[9];
rz(2.070422737615112) q[9];
ry(-3.141592415588863) q[10];
rz(-0.48702436695035095) q[10];
ry(4.915330125143669e-07) q[11];
rz(-2.2793892224314902) q[11];
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
ry(-2.8508527279793725e-07) q[0];
rz(-0.7706669849062733) q[0];
ry(2.012541815265551e-06) q[1];
rz(-1.1888621919374336) q[1];
ry(-7.041696967746702e-07) q[2];
rz(0.6691534368455043) q[2];
ry(3.1415907568334784) q[3];
rz(1.383492393547656) q[3];
ry(-3.1415874655507396) q[4];
rz(-1.6008171361886419) q[4];
ry(-3.141592511188302) q[5];
rz(2.534181110241153) q[5];
ry(-1.5707997498303623) q[6];
rz(1.8676084607541557) q[6];
ry(2.185008077968727e-07) q[7];
rz(-2.1082130297248964) q[7];
ry(1.5707964979164264) q[8];
rz(3.141591536781468) q[8];
ry(-1.5707957707860587) q[9];
rz(2.0680389824061987) q[9];
ry(1.5707951784289413) q[10];
rz(9.75143706464668e-07) q[10];
ry(2.9751506869290303) q[11];
rz(-0.35282053449758655) q[11];
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
ry(1.5707954386982586) q[0];
rz(1.4015604112235371) q[0];
ry(1.570795591417211) q[1];
rz(-3.1415846606144755) q[1];
ry(-1.5707948610728732) q[2];
rz(3.1415764157292267) q[2];
ry(-3.1415925432125196) q[3];
rz(3.080251802402369) q[3];
ry(-1.5708053247341862) q[4];
rz(-3.141592512295048) q[4];
ry(3.1415914027938414) q[5];
rz(-0.7862740669286881) q[5];
ry(-3.1415850381859953) q[6];
rz(-2.844779082664843) q[6];
ry(-3.1415924203205945) q[7];
rz(0.11479639931702047) q[7];
ry(2.418654693280918) q[8];
rz(-3.1415925760517958) q[8];
ry(-3.1415925528355717) q[9];
rz(0.36428298214012467) q[9];
ry(-2.2937339681075244) q[10];
rz(-2.198744031289834e-07) q[10];
ry(-3.1415924583328767) q[11];
rz(-0.3528040308884343) q[11];
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
ry(3.1415922560564904) q[0];
rz(1.6186767679638405) q[0];
ry(1.5270584218122656) q[1];
rz(2.953951059111215) q[1];
ry(1.5707951747049833) q[2];
rz(-1.3536750594248017) q[2];
ry(1.5707947206716169) q[3];
rz(2.95395061477786) q[3];
ry(-1.5707946639133814) q[4];
rz(-2.924458231533016) q[4];
ry(0.16128747632971485) q[5];
rz(-0.7514108550390537) q[5];
ry(-1.5707950402323112) q[6];
rz(1.8738501047251654) q[6];
ry(1.570795865152001) q[7];
rz(-0.18764228995690943) q[7];
ry(-1.5707958704699692) q[8];
rz(-2.8220082707264065) q[8];
ry(-1.897302092402512) q[9];
rz(-1.8013105314083742) q[9];
ry(1.5707963060903753) q[10];
rz(2.2195889255428654) q[10];
ry(-1.5707958136772016) q[11];
rz(-0.1876430696417897) q[11];