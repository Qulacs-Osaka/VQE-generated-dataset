OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.5702165312579472) q[0];
rz(3.141576706542649) q[0];
ry(0.0011058618076136284) q[1];
rz(-0.27624545344813356) q[1];
ry(-2.976118673980322) q[2];
rz(0.0633292759705526) q[2];
ry(3.141564246063482) q[3];
rz(2.714934350355777) q[3];
ry(-3.1415221202665977) q[4];
rz(-0.5538458940030511) q[4];
ry(1.5831435037195176) q[5];
rz(-3.1415770351331074) q[5];
ry(1.6374123967789718) q[6];
rz(-3.140862323523228) q[6];
ry(-3.126559188804794) q[7];
rz(0.8764787001219044) q[7];
ry(-1.570774413290245) q[8];
rz(0.9639543725984832) q[8];
ry(6.7633139666511966e-06) q[9];
rz(-0.8300675908523159) q[9];
ry(-0.7516275650419165) q[10];
rz(2.5093117989237625) q[10];
ry(-3.86980835269668e-05) q[11];
rz(2.61887985938782) q[11];
ry(-1.5707444973089464) q[12];
rz(-3.141580422220131) q[12];
ry(2.6086876898076254) q[13];
rz(-2.9961763944620294) q[13];
ry(-1.5579066284572165) q[14];
rz(-0.003601930331408098) q[14];
ry(-1.5707952304511759) q[15];
rz(2.1642523556557896) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(2.414903746013401) q[0];
rz(1.5315864186565236) q[0];
ry(-1.570746630937968) q[1];
rz(1.4782571251727799) q[1];
ry(0.0011785925769363104) q[2];
rz(2.9516307065645164) q[2];
ry(-3.141586936757276) q[3];
rz(0.8660327829419333) q[3];
ry(1.570814689663698) q[4];
rz(0.007219874583482484) q[4];
ry(-1.5842361675379568) q[5];
rz(-0.10450130371640545) q[5];
ry(1.5120124396896888) q[6];
rz(1.0783895627311395) q[6];
ry(-3.1414472155064934) q[7];
rz(-0.7318529457666435) q[7];
ry(-4.661094169122748e-05) q[8];
rz(-2.533087160394613) q[8];
ry(-3.1415915290493133) q[9];
rz(1.1823270476977943) q[9];
ry(-3.1415842311961866) q[10];
rz(-2.093878836948738) q[10];
ry(3.1414174152462007) q[11];
rz(1.5292617587497652) q[11];
ry(-2.349648055006339) q[12];
rz(1.5708280472190106) q[12];
ry(2.0525200236664944e-05) q[13];
rz(-1.02727106432264) q[13];
ry(-3.1217783543167408) q[14];
rz(1.5825748012578353) q[14];
ry(-3.1415913890772473) q[15];
rz(1.1043339807693044) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(1.9879569511576864) q[0];
rz(0.04228847117467182) q[0];
ry(-1.5673037734117379) q[1];
rz(-0.2788783435554688) q[1];
ry(2.8758081900032244) q[2];
rz(2.5664510633093927) q[2];
ry(1.570758217056021) q[3];
rz(-3.141589712190353) q[3];
ry(-1.7343241492469392) q[4];
rz(1.4287950624077832) q[4];
ry(0.27030656690411065) q[5];
rz(0.5005692606750092) q[5];
ry(3.1413383189217594) q[6];
rz(2.937708057838063) q[6];
ry(-0.0026809961589184376) q[7];
rz(0.6506885417155006) q[7];
ry(-1.6465027751422587) q[8];
rz(1.7889772839100204) q[8];
ry(3.1415923525306577) q[9];
rz(2.9268544180512768) q[9];
ry(-6.729523351987425e-05) q[10];
rz(3.0391641301181727) q[10];
ry(3.1379213932509646) q[11];
rz(0.3335815119648006) q[11];
ry(0.4020846545230906) q[12];
rz(1.2669995472159856) q[12];
ry(-3.1415846977484385) q[13];
rz(1.3984977890208448) q[13];
ry(-3.1358923856780154) q[14];
rz(-1.5419817877296556) q[14];
ry(5.370800487935533e-06) q[15];
rz(-2.0816817028167023) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.1415922930612723) q[0];
rz(-1.3925500392054904) q[0];
ry(3.1415800264989686) q[1];
rz(1.2201209306172602) q[1];
ry(1.6935794792967673e-06) q[2];
rz(-2.566555552020708) q[2];
ry(-1.5707936533509923) q[3];
rz(-1.5709251468095333) q[3];
ry(-3.141587621077416) q[4];
rz(1.4024708015373129) q[4];
ry(0.0002109707617190537) q[5];
rz(-1.3078493977526708) q[5];
ry(1.3673541631398224e-05) q[6];
rz(-0.34006459508391185) q[6];
ry(3.137035320909183) q[7];
rz(1.252452774580266) q[7];
ry(3.0917256103167623) q[8];
rz(2.4320645059045005) q[8];
ry(-3.141591930203469) q[9];
rz(0.3738473484134017) q[9];
ry(-0.24056384481332774) q[10];
rz(-0.3069274769742476) q[10];
ry(1.5708001271411787) q[11];
rz(3.1415818127368627) q[11];
ry(-1.061514747830735) q[12];
rz(-1.2710574906711196) q[12];
ry(0.07630434619527549) q[13];
rz(-0.000628274858777711) q[13];
ry(1.4675785729128348) q[14];
rz(2.130915553013261) q[14];
ry(-2.7205799680002407) q[15];
rz(1.603729422619256) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(2.8933012377655283) q[0];
rz(0.28490339901126993) q[0];
ry(2.5945946918508205) q[1];
rz(1.3320634519484822) q[1];
ry(-2.9007358145561155) q[2];
rz(1.2549993401456978) q[2];
ry(1.5697063934500912) q[3];
rz(-1.729001546664196) q[3];
ry(1.5763244526651) q[4];
rz(-1.6684956078151945) q[4];
ry(-9.366900265659553e-05) q[5];
rz(-1.6875597448143427) q[5];
ry(0.0021893139744399615) q[6];
rz(1.8428328615781888) q[6];
ry(-0.0003065697872451795) q[7];
rz(-2.6782554590400616) q[7];
ry(-3.141030025591369) q[8];
rz(-0.9325384091894683) q[8];
ry(3.141588502512942) q[9];
rz(0.2589224364844363) q[9];
ry(2.5306510139182943e-05) q[10];
rz(-1.5663702317593664) q[10];
ry(1.5707913465754317) q[11];
rz(1.8185602276854496) q[11];
ry(-2.9414301736169707e-05) q[12];
rz(-1.7408541197332763) q[12];
ry(-3.1415873318866887) q[13];
rz(1.7611149924456901) q[13];
ry(-3.141580807624201) q[14];
rz(2.132404987204529) q[14];
ry(-0.0023660323780889674) q[15];
rz(1.5378586953429867) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-0.0009556611574419449) q[0];
rz(1.3844500979565924) q[0];
ry(-1.5705554846031635) q[1];
rz(3.139679395702907) q[1];
ry(3.1414331640941393) q[2];
rz(1.9975328266543557) q[2];
ry(3.141372409431472) q[3];
rz(1.9129060713512835) q[3];
ry(-0.11320342036464924) q[4];
rz(0.7673592616700508) q[4];
ry(3.1409397852542678) q[5];
rz(2.1092762663318965) q[5];
ry(0.003174730278976412) q[6];
rz(-0.8882151973496218) q[6];
ry(6.985384210622669e-06) q[7];
rz(-1.6509729191292106) q[7];
ry(0.013720918509589453) q[8];
rz(1.9152246176789605) q[8];
ry(3.1415836340296357) q[9];
rz(-1.508473041047603) q[9];
ry(9.054272505338105e-06) q[10];
rz(2.315383904282174) q[10];
ry(-2.242227968426362e-05) q[11];
rz(-1.2936105773491908) q[11];
ry(-7.51782618100066e-05) q[12];
rz(-1.8080727498054932) q[12];
ry(-3.14159034936229) q[13];
rz(-2.5613819509006035) q[13];
ry(-0.004890183042372357) q[14];
rz(-1.6205842806367157) q[14];
ry(1.549187586971298) q[15];
rz(1.570795058902809) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(3.14040124501179) q[0];
rz(-2.7589628417381364) q[0];
ry(-1.5711573001794044) q[1];
rz(-0.6187332430892658) q[1];
ry(-0.00037590587806231264) q[2];
rz(0.6058953943811024) q[2];
ry(3.141198023129822) q[3];
rz(2.0664355365528637) q[3];
ry(3.12127948219171) q[4];
rz(2.244929751495978) q[4];
ry(-1.1963762929339425) q[5];
rz(-2.0966164287329914) q[5];
ry(-3.140877498577492) q[6];
rz(-0.9317355238885998) q[6];
ry(3.1408268260437846) q[7];
rz(-0.6380072443928301) q[7];
ry(3.1415409187895813) q[8];
rz(-0.1556735094181878) q[8];
ry(-3.141592273193751) q[9];
rz(-2.622903109474545) q[9];
ry(1.7947041417443188e-05) q[10];
rz(2.758986741740908) q[10];
ry(-3.1415885370113816) q[11];
rz(0.5010260507013008) q[11];
ry(0.00015839287086304778) q[12];
rz(-0.512869542352774) q[12];
ry(-3.1415743010152966) q[13];
rz(-2.8727270234652256) q[13];
ry(3.1385444397780176) q[14];
rz(-0.27847316859668103) q[14];
ry(2.2285197048168763) q[15];
rz(-1.5702101945174283) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(0.0005926047887259145) q[0];
rz(-1.5151697752141375) q[0];
ry(-0.0004850237382632731) q[1];
rz(1.6125358038526265) q[1];
ry(-3.1397282959195287) q[2];
rz(2.279027350098664) q[2];
ry(-1.5707750093715438) q[3];
rz(-1.9176084597268663) q[3];
ry(-1.5332830307589755) q[4];
rz(2.250033225159359) q[4];
ry(2.9936672270492735) q[5];
rz(2.237254391362521) q[5];
ry(1.5698818403303314) q[6];
rz(0.046130067447380796) q[6];
ry(-1.5722811241462191) q[7];
rz(-3.040847763892097) q[7];
ry(0.2740961941793678) q[8];
rz(-1.0547453833252656) q[8];
ry(-4.379957629856107e-06) q[9];
rz(1.1171212512345527) q[9];
ry(-3.129059339955769) q[10];
rz(1.6265333692333888) q[10];
ry(0.007165904309607402) q[11];
rz(-1.578704498369671) q[11];
ry(-3.137642303956419) q[12];
rz(-2.40006234175182) q[12];
ry(-3.1412430666275686) q[13];
rz(-0.09182977827610807) q[13];
ry(-3.141406639083134) q[14];
rz(2.230812254842274) q[14];
ry(-0.004573225487210841) q[15];
rz(-1.571384677352772) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.1415466275048645) q[0];
rz(-1.956724872852674) q[0];
ry(3.141473314301515) q[1];
rz(1.4279451991020755) q[1];
ry(-0.00022842705388903542) q[2];
rz(-0.5821433885441971) q[2];
ry(-5.292724879168986e-05) q[3];
rz(1.6168412703423074) q[3];
ry(-0.0072391527903867825) q[4];
rz(0.4768691110352123) q[4];
ry(-3.127020849655873) q[5];
rz(-1.381776199453978) q[5];
ry(-0.03892070594962327) q[6];
rz(-2.7456680734898042) q[6];
ry(-1.5729096837285486) q[7];
rz(1.3546780917312835) q[7];
ry(0.2655708050302321) q[8];
rz(2.802602339407101) q[8];
ry(1.570793944768176) q[9];
rz(3.1415918033608436) q[9];
ry(-0.26497807142534935) q[10];
rz(1.7969511973885681) q[10];
ry(0.09917703400224366) q[11];
rz(2.377841188526721) q[11];
ry(-3.103050336125055) q[12];
rz(-0.5215545031988741) q[12];
ry(0.014238804904750713) q[13];
rz(1.1841472538674063) q[13];
ry(3.136671488139989) q[14];
rz(-2.2549009622477767) q[14];
ry(-2.1335845396295445) q[15];
rz(-1.5707958627958019) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(-3.4487447058974483e-06) q[0];
rz(-1.2179219032697706) q[0];
ry(-3.1415925974898524) q[1];
rz(-2.3562809103167335) q[1];
ry(-3.1415893343955448) q[2];
rz(-2.792883923795084) q[2];
ry(-3.1415858371523404) q[3];
rz(2.8407614187367076) q[3];
ry(-3.141592173154574) q[4];
rz(1.899982180977774) q[4];
ry(-3.141591671148194) q[5];
rz(1.1699833108485505) q[5];
ry(-3.1415846391541526) q[6];
rz(-1.2853870229219648) q[6];
ry(3.141586134770894) q[7];
rz(-0.21339113844480995) q[7];
ry(-5.980031065000446e-06) q[8];
rz(-1.596458273385731) q[8];
ry(1.5707871663987987) q[9];
rz(3.1409697723559438) q[9];
ry(3.7048198731071125e-07) q[10];
rz(1.3473812282888984) q[10];
ry(-2.723555999703921e-06) q[11];
rz(0.7952408689217522) q[11];
ry(-1.8646731983373854e-05) q[12];
rz(1.9969507886682865) q[12];
ry(7.006529145902474e-07) q[13];
rz(1.9289452181759925) q[13];
ry(-3.137493254263549) q[14];
rz(-1.5741866254251209) q[14];
ry(1.5748900481867434) q[15];
rz(-1.7017834166923864) q[15];
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
cz q[0],q[12];
cz q[0],q[13];
cz q[0],q[14];
cz q[0],q[15];
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
cz q[1],q[12];
cz q[1],q[13];
cz q[1],q[14];
cz q[1],q[15];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[2],q[12];
cz q[2],q[13];
cz q[2],q[14];
cz q[2],q[15];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[3],q[12];
cz q[3],q[13];
cz q[3],q[14];
cz q[3],q[15];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[4],q[12];
cz q[4],q[13];
cz q[4],q[14];
cz q[4],q[15];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[5],q[12];
cz q[5],q[13];
cz q[5],q[14];
cz q[5],q[15];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[6],q[12];
cz q[6],q[13];
cz q[6],q[14];
cz q[6],q[15];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[7],q[12];
cz q[7],q[13];
cz q[7],q[14];
cz q[7],q[15];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[8],q[12];
cz q[8],q[13];
cz q[8],q[14];
cz q[8],q[15];
cz q[9],q[10];
cz q[9],q[11];
cz q[9],q[12];
cz q[9],q[13];
cz q[9],q[14];
cz q[9],q[15];
cz q[10],q[11];
cz q[10],q[12];
cz q[10],q[13];
cz q[10],q[14];
cz q[10],q[15];
cz q[11],q[12];
cz q[11],q[13];
cz q[11],q[14];
cz q[11],q[15];
cz q[12],q[13];
cz q[12],q[14];
cz q[12],q[15];
cz q[13],q[14];
cz q[13],q[15];
cz q[14],q[15];
ry(0.0002031661834852411) q[0];
rz(-2.645008660290141) q[0];
ry(3.1415919726441506) q[1];
rz(-2.754102692988051) q[1];
ry(1.7350256679739087) q[2];
rz(-3.119336094207532) q[2];
ry(1.571034430993142) q[3];
rz(-1.6506481417187573) q[3];
ry(-3.1413759049777807) q[4];
rz(0.7422291497430952) q[4];
ry(3.140359887330634) q[5];
rz(2.1678422608742896) q[5];
ry(3.1304363445332095) q[6];
rz(1.4140849586816635) q[6];
ry(-1.571430934452638) q[7];
rz(-3.008798902526926) q[7];
ry(-0.001321347232619863) q[8];
rz(-1.2114469191267583) q[8];
ry(1.5731666715036259) q[9];
rz(-2.127757032153943) q[9];
ry(-2.3223592694081434) q[10];
rz(-0.005758672529311236) q[10];
ry(1.5707961419493783) q[11];
rz(-0.01924329898671659) q[11];
ry(0.0005469406078720939) q[12];
rz(-1.5325920652776466) q[12];
ry(-1.1500957561717557) q[13];
rz(-0.33830791087253503) q[13];
ry(2.369630530502708) q[14];
rz(1.5689466553048739) q[14];
ry(-0.0008364839725210326) q[15];
rz(-1.4398100654559318) q[15];