OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(0.013608265509390094) q[0];
rz(1.6778491473914565) q[0];
ry(-2.68757853725442) q[1];
rz(-0.2943586233690567) q[1];
ry(0.9741895737690183) q[2];
rz(2.572433252977301) q[2];
ry(0.3047413574259316) q[3];
rz(-0.59398293656227) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.6388330589252281) q[0];
rz(-1.8699099723549253) q[0];
ry(2.5926023171036783) q[1];
rz(0.9742311975611547) q[1];
ry(0.573360144127808) q[2];
rz(3.0875257522859463) q[2];
ry(-2.7259519663792786) q[3];
rz(1.1851140008763723) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.9740539759656412) q[0];
rz(-2.051410608658202) q[0];
ry(-3.1175368050524486) q[1];
rz(-1.3756643350685795) q[1];
ry(-1.5435953486988614) q[2];
rz(2.3447072649905314) q[2];
ry(-1.9663351695179803) q[3];
rz(1.0016132482355302) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-2.508717221076507) q[0];
rz(2.6008994365246694) q[0];
ry(0.8400719115836675) q[1];
rz(1.1423150163262967) q[1];
ry(-0.7561893680796175) q[2];
rz(-2.80011092425663) q[2];
ry(2.179401148684084) q[3];
rz(-2.218140736813991) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(1.616917824924225) q[0];
rz(0.6472495368738883) q[0];
ry(-2.0114125206600253) q[1];
rz(-0.9735744152279174) q[1];
ry(0.056202599202670456) q[2];
rz(-0.025499659064553773) q[2];
ry(-2.568474643117218) q[3];
rz(0.726575872095168) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(0.5780797527956247) q[0];
rz(-2.2245379708617614) q[0];
ry(0.046286619683415076) q[1];
rz(1.099595572675871) q[1];
ry(-2.9631846109596056) q[2];
rz(-0.7360079787031282) q[2];
ry(-1.8011554244634724) q[3];
rz(-0.6151080590838998) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(2.9199668361592455) q[0];
rz(0.8167355569289203) q[0];
ry(-0.9413756806843746) q[1];
rz(-0.48475469535455407) q[1];
ry(2.7104101442933537) q[2];
rz(-0.5958143951751024) q[2];
ry(2.3528505392351895) q[3];
rz(-1.4695781260774623) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.03168027269236677) q[0];
rz(2.313718453064292) q[0];
ry(-1.1340331927817768) q[1];
rz(0.3238612542816101) q[1];
ry(0.8705256309049679) q[2];
rz(1.2227237286778996) q[2];
ry(0.40063100972326104) q[3];
rz(1.780077280523151) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.7144604730809763) q[0];
rz(-2.108742137457559) q[0];
ry(-2.8610553676647146) q[1];
rz(-1.705817322041745) q[1];
ry(-1.2207562904259293) q[2];
rz(0.22466068124517768) q[2];
ry(-0.06331041223148404) q[3];
rz(3.0069007227183877) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-1.7745891883350848) q[0];
rz(1.5865753769312816) q[0];
ry(0.8310344735093206) q[1];
rz(1.056784747108339) q[1];
ry(-2.5269227646197714) q[2];
rz(-1.5511144973994564) q[2];
ry(0.030761377780977307) q[3];
rz(0.3039212274223759) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.013295325577214355) q[0];
rz(1.825836842227142) q[0];
ry(2.3436852425270414) q[1];
rz(2.348113768279738) q[1];
ry(-2.160976547437423) q[2];
rz(1.7569323775893355) q[2];
ry(2.8152127172005206) q[3];
rz(0.13443082233660728) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
ry(-0.13699531992877298) q[0];
rz(-2.5906570783728164) q[0];
ry(1.4519006587321683) q[1];
rz(-1.769440080049306) q[1];
ry(1.4326354434606252) q[2];
rz(-0.6137932979353176) q[2];
ry(-2.3818411561037767) q[3];
rz(1.7559428689438314) q[3];