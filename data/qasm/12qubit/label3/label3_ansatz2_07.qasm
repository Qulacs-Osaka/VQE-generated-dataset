OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5708097094908577) q[0];
rz(0.42055102965156355) q[0];
ry(1.916940071400941) q[1];
rz(1.4519945795303926) q[1];
ry(-1.5707989789861114) q[2];
rz(-3.0789427346897544) q[2];
ry(-1.570779587235047) q[3];
rz(1.2595368993516014) q[3];
ry(-2.1314813917676934) q[4];
rz(3.1415922307228215) q[4];
ry(-1.5708000106735929) q[5];
rz(-0.4721044545274684) q[5];
ry(-8.033805634344304e-05) q[6];
rz(2.855228668354128) q[6];
ry(5.396363711440699e-08) q[7];
rz(2.4264101844892494) q[7];
ry(-1.9182520758356463) q[8];
rz(8.511224977221151e-07) q[8];
ry(-0.22874489177076335) q[9];
rz(-1.6729142380996664) q[9];
ry(1.6082848740802111) q[10];
rz(-1.4376439943252635) q[10];
ry(-3.141519166042134) q[11];
rz(-2.910036774424067) q[11];
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
ry(-3.1415786198286413) q[0];
rz(2.739138935490264) q[0];
ry(8.998801712678528e-07) q[1];
rz(0.11881020108706493) q[1];
ry(3.141373285713775) q[2];
rz(1.636227886434132) q[2];
ry(-0.00045869382374099803) q[3];
rz(0.4114507743062335) q[3];
ry(1.5707481199846702) q[4];
rz(1.5707942761422753) q[4];
ry(-0.00019175414471828134) q[5];
rz(-0.9331740598570927) q[5];
ry(-1.5708014467589668) q[6];
rz(1.5722192113469304) q[6];
ry(-4.240183216938402e-08) q[7];
rz(0.19828156898482785) q[7];
ry(1.5708438369152211) q[8];
rz(-1.57079669890203) q[8];
ry(-3.141587568732137) q[9];
rz(1.4686370039215222) q[9];
ry(-5.778890889007018e-06) q[10];
rz(-0.13314882693622693) q[10];
ry(3.139479947735802) q[11];
rz(-0.010962180883264413) q[11];
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
ry(-0.01367119752342037) q[0];
rz(-0.802154940458198) q[0];
ry(2.3631195274957553) q[1];
rz(-1.9427499052579584) q[1];
ry(-0.3223965565247036) q[2];
rz(3.136457299075673) q[2];
ry(-0.0001388587437416433) q[3];
rz(-2.0827695575898297) q[3];
ry(1.5699928115052781) q[4];
rz(6.603058688448016e-05) q[4];
ry(0.7908836874978808) q[5];
rz(2.7127522916528433) q[5];
ry(1.5707971015432731) q[6];
rz(1.5707019986686843) q[6];
ry(4.846993902845349e-07) q[7];
rz(2.490086997257571) q[7];
ry(-1.5699359482291813) q[8];
rz(2.3511047365867873e-05) q[8];
ry(-1.5708005193675776) q[9];
rz(0.8792646721356432) q[9];
ry(-1.517498130515623) q[10];
rz(-3.14159198367694) q[10];
ry(-1.57079248685719) q[11];
rz(1.57090008707339) q[11];
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
ry(1.5710579752283258) q[0];
rz(-3.1366921036670354) q[0];
ry(-2.29386344026139e-05) q[1];
rz(2.180803622485894) q[1];
ry(-1.5708223063914903) q[2];
rz(0.31661904753459513) q[2];
ry(2.6670299922315618e-05) q[3];
rz(1.9825530919163212) q[3];
ry(1.5242736867651399) q[4];
rz(1.5704850964424877) q[4];
ry(1.5708022997729065) q[5];
rz(-3.1312532662536134) q[5];
ry(1.5704762841720505) q[6];
rz(1.5713570446382912) q[6];
ry(3.141592390220365) q[7];
rz(1.1519630621891424) q[7];
ry(1.5731173942652004) q[8];
rz(1.570386346872562) q[8];
ry(-1.5708727867034362) q[9];
rz(-0.004953024440447253) q[9];
ry(1.5706284422484142) q[10];
rz(1.4643784374661335) q[10];
ry(-0.7042364647542962) q[11];
rz(-1.57462584274782) q[11];
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
ry(0.41800858984582323) q[0];
rz(9.151021536180082e-05) q[0];
ry(3.3472686589414914e-05) q[1];
rz(-1.7916656636578123) q[1];
ry(-3.1411958376156703) q[2];
rz(0.2893360602585247) q[2];
ry(-1.5707503480428953) q[3];
rz(-1.5889339443451345) q[3];
ry(-1.1135699369126955) q[4];
rz(-3.1283135686732964) q[4];
ry(0.0022941591429690433) q[5];
rz(-0.009885386672277736) q[5];
ry(-1.6394936328111165) q[6];
rz(0.0003724716332142649) q[6];
ry(-3.1415924117359446) q[7];
rz(-0.5747230980349469) q[7];
ry(0.42292516400436475) q[8];
rz(-3.1412892274642545) q[8];
ry(3.1278299183218343) q[9];
rz(1.6418627774375223) q[9];
ry(1.366357558207909e-05) q[10];
rz(1.6771671812541458) q[10];
ry(-0.019251366826842715) q[11];
rz(-1.1358846244721124) q[11];
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
ry(2.7866643546888805) q[0];
rz(-5.1884586498256096e-05) q[0];
ry(0.018098597449721837) q[1];
rz(2.288855562289102) q[1];
ry(-0.000997112371373682) q[2];
rz(0.022216944236461363) q[2];
ry(0.017555090640652336) q[3];
rz(-1.2698996855022324) q[3];
ry(0.02616882314067226) q[4];
rz(3.1282519985309816) q[4];
ry(0.022165498330938505) q[5];
rz(-1.4683467232076914) q[5];
ry(-0.8351686386471365) q[6];
rz(-0.0001619435398287905) q[6];
ry(2.3870154177488386e-08) q[7];
rz(-0.30988967744303686) q[7];
ry(-1.7449473186548237) q[8];
rz(-3.1414665176454815) q[8];
ry(-3.1415707209368224) q[9];
rz(-0.9393802265086155) q[9];
ry(1.5707998108606582) q[10];
rz(-2.949685848521699) q[10];
ry(-2.1475966803130575e-05) q[11];
rz(-2.64850589040479) q[11];
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
ry(1.5708026623934703) q[0];
rz(-1.68507836015202) q[0];
ry(-3.1415195614069344) q[1];
rz(-2.406363010375603) q[1];
ry(-1.570797762566225) q[2];
rz(-3.1167910457877026) q[2];
ry(-0.0002623385354852266) q[3];
rz(-0.28274963912396806) q[3];
ry(1.5708082547557387) q[4];
rz(-2.4820870190404243) q[4];
ry(3.140986493862302) q[5];
rz(1.6987223725865475) q[5];
ry(1.5708106974250482) q[6];
rz(-2.1489756591581495) q[6];
ry(3.1415916628300997) q[7];
rz(-0.5406321645038332) q[7];
ry(-1.570810297709907) q[8];
rz(2.7314583090330293) q[8];
ry(-3.141582962207502) q[9];
rz(-2.5851758409868166) q[9];
ry(-1.2587965932300047e-05) q[10];
rz(-1.2209579813239202) q[10];
ry(-1.7571884193046117e-05) q[11];
rz(-2.4960792660950926) q[11];
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
ry(-0.024942738389248653) q[0];
rz(3.135822955307207) q[0];
ry(1.6371829859003197) q[1];
rz(0.10425737617911904) q[1];
ry(1.5711180359789978) q[2];
rz(-1.7634455047995097) q[2];
ry(-1.8243949659644798) q[3];
rz(1.2872281474276015) q[3];
ry(3.101013931149204) q[4];
rz(-0.38393490503634453) q[4];
ry(-3.141568051174306) q[5];
rz(2.9462133975734917) q[5];
ry(-0.029770585331404444) q[6];
rz(-0.6359948922369603) q[6];
ry(-1.57079450596318) q[7];
rz(1.5707967490202677) q[7];
ry(-0.062161036963258955) q[8];
rz(1.8020334247787684) q[8];
ry(-2.4481523793795756) q[9];
rz(-1.6576259439889867) q[9];
ry(0.028954119561246723) q[10];
rz(-0.46235761014220483) q[10];
ry(0.9186342041078094) q[11];
rz(0.7440702717022727) q[11];
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
ry(-3.1415923341967105) q[0];
rz(-0.11941132144826304) q[0];
ry(3.141592286211265) q[1];
rz(-1.4665449761336273) q[1];
ry(3.1415917823617314) q[2];
rz(2.9495365579822956) q[2];
ry(-3.1415923799375705) q[3];
rz(-0.283567612182007) q[3];
ry(-3.141592518021824) q[4];
rz(-2.6152652763160438) q[4];
ry(3.141592640995899) q[5];
rz(0.19584081737235073) q[5];
ry(-3.141592144423908) q[6];
rz(-1.2146090199865514) q[6];
ry(-1.5708031250768402) q[7];
rz(3.1415925553860724) q[7];
ry(-3.141592630083033) q[8];
rz(-0.17897821711997608) q[8];
ry(1.0397368166437104e-06) q[9];
rz(-3.054671975360259) q[9];
ry(8.72271459506635e-07) q[10];
rz(3.0616992001342007) q[10];
ry(-1.204101074079249e-06) q[11];
rz(0.8269199700736394) q[11];
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
ry(1.5706264600065045) q[0];
rz(-2.0389153629773156) q[0];
ry(-1.5708339710011892) q[1];
rz(-1.8139752840003345) q[1];
ry(1.5706267318578373) q[2];
rz(1.9924910121921116) q[2];
ry(-1.5707467217475495) q[3];
rz(1.430834632295595) q[3];
ry(-1.5709658214755589) q[4];
rz(-1.646623980278898) q[4];
ry(3.1841799156495654e-05) q[5];
rz(1.090477509341901) q[5];
ry(1.5709659091019939) q[6];
rz(-2.600149842174441) q[6];
ry(2.2441710880102415) q[7];
rz(-1.5707920148822385) q[7];
ry(1.5706317603392106) q[8];
rz(1.7054774015638543) q[8];
ry(-1.5707796483260337) q[9];
rz(0.3546699734922987) q[9];
ry(-1.570971870531614) q[10];
rz(-1.0101369805846188) q[10];
ry(1.5708100347745573) q[11];
rz(-0.5018730017247315) q[11];
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
ry(3.1415924234617107) q[0];
rz(3.0619444097430932) q[0];
ry(3.141589110108977) q[1];
rz(1.715798370812521) q[1];
ry(-3.1415925649209027) q[2];
rz(0.8101577645571173) q[2];
ry(3.1415894648911507) q[3];
rz(1.8190161101050097) q[3];
ry(-3.141592191114326) q[4];
rz(-2.82902538453906) q[4];
ry(-4.298873612640364e-06) q[5];
rz(2.023015368945064) q[5];
ry(-3.141592296803004) q[6];
rz(-0.6409677332266734) q[6];
ry(-1.5707946577530592) q[7];
rz(-2.753404474014688) q[7];
ry(3.1415923808105597) q[8];
rz(0.5230839547136307) q[8];
ry(-5.178634227234779e-06) q[9];
rz(1.604325260053486) q[9];
ry(-3.1415924863103557) q[10];
rz(-2.1925222639797073) q[10];
ry(-3.1415876454386567) q[11];
rz(-1.684473791203852) q[11];