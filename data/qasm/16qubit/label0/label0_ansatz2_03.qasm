OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(3.141531762130626) q[0];
rz(-1.5313790273671843) q[0];
ry(0.0007638391758462129) q[1];
rz(3.0122672112676403) q[1];
ry(0.0008749348985501726) q[2];
rz(-1.9462558569885315) q[2];
ry(0.001086695714399788) q[3];
rz(0.9398868974957485) q[3];
ry(3.1414382459539723) q[4];
rz(2.5808441734146306) q[4];
ry(-0.023085154189907087) q[5];
rz(2.926212761557506) q[5];
ry(3.0013111060899136) q[6];
rz(-2.9448742549007747) q[6];
ry(-2.7764454411205683) q[7];
rz(-2.9555769647684564) q[7];
ry(-2.334550425969818) q[8];
rz(-2.9880937583266993) q[8];
ry(-2.8428124718170444) q[9];
rz(1.002101326770818) q[9];
ry(-0.011592697719048141) q[10];
rz(1.4445240151769068) q[10];
ry(-0.016089052728438524) q[11];
rz(-1.5907351456011793) q[11];
ry(-0.002283054244183567) q[12];
rz(1.450623814798301) q[12];
ry(3.1390531532013752) q[13];
rz(-1.582675086097256) q[13];
ry(0.0007706095729984597) q[14];
rz(-1.4180619304963828) q[14];
ry(-3.141189903125541) q[15];
rz(0.5462961057300293) q[15];
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
ry(0.000220627558655373) q[0];
rz(0.2907730172264049) q[0];
ry(1.140783242004062e-05) q[1];
rz(-0.9726544625108247) q[1];
ry(3.141416920914161) q[2];
rz(-1.1598981403909638) q[2];
ry(3.1409514912824825) q[3];
rz(1.2532265147904909) q[3];
ry(-3.1394551520402323) q[4];
rz(0.458255312299614) q[4];
ry(3.1232597744187607) q[5];
rz(2.9297676404533637) q[5];
ry(-3.0508700399359974) q[6];
rz(-2.9482820153872273) q[6];
ry(-0.24994067048185278) q[7];
rz(2.97130091002276) q[7];
ry(2.3882371868734706) q[8];
rz(3.135947421576343) q[8];
ry(-2.801377657737242) q[9];
rz(-0.42505044133162784) q[9];
ry(1.975843947536677) q[10];
rz(3.1374193184113706) q[10];
ry(0.26138093147773217) q[11];
rz(0.0053849195033075494) q[11];
ry(-0.015932497890401118) q[12];
rz(-0.029085203055135626) q[12];
ry(-3.1276779362281117) q[13];
rz(3.0287170701972395) q[13];
ry(3.141439113171894) q[14];
rz(-3.0303150559579812) q[14];
ry(-3.1397939752014574) q[15];
rz(2.1293800542368215) q[15];
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
ry(-0.01171916145673535) q[0];
rz(3.1230187106818805) q[0];
ry(-0.014105166268133296) q[1];
rz(-3.1413417706093774) q[1];
ry(0.02623567461975734) q[2];
rz(-3.137632706514862) q[2];
ry(2.9780421305175135) q[3];
rz(-0.0022437076102894697) q[3];
ry(-2.755973226933855) q[4];
rz(-3.1390795574210753) q[4];
ry(-0.6566564483444663) q[5];
rz(-3.1385118281174553) q[5];
ry(-0.4059653198074818) q[6];
rz(3.1270657734765677) q[6];
ry(-0.18644006360138177) q[7];
rz(0.08498466774108805) q[7];
ry(-3.0903092489092385) q[8];
rz(-0.6692688300208981) q[8];
ry(-3.0347840300390407) q[9];
rz(-2.7749452767153264) q[9];
ry(1.9973286553078369) q[10];
rz(3.132194313733214) q[10];
ry(-3.0381912767634534) q[11];
rz(0.032542151470772336) q[11];
ry(3.133404811332518) q[12];
rz(-1.4899828559285768) q[12];
ry(-0.0072009824060451895) q[13];
rz(0.004436984629356534) q[13];
ry(3.131864129645563) q[14];
rz(3.1292224509698103) q[14];
ry(-0.00017011504114059422) q[15];
rz(0.826217447801529) q[15];
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
ry(3.1340734897593334) q[0];
rz(-2.1520278066110254) q[0];
ry(0.008815091417497811) q[1];
rz(-3.0278303706283123) q[1];
ry(0.018155352087616015) q[2];
rz(-0.8615856742971604) q[2];
ry(-0.11083999062313765) q[3];
rz(1.8476112725991982) q[3];
ry(0.2818472499174263) q[4];
rz(-2.9625346473374483) q[4];
ry(2.572564604776732) q[5];
rz(1.6437946669573211) q[5];
ry(-2.8479114923065008) q[6];
rz(-2.071032402978145) q[6];
ry(0.11298562042307658) q[7];
rz(-0.19882871773300081) q[7];
ry(-3.1038525085428263) q[8];
rz(1.569726430674737) q[8];
ry(-3.0949420207328764) q[9];
rz(1.6612140878670933) q[9];
ry(0.11115569004870451) q[10];
rz(-1.7105833845956098) q[10];
ry(2.8731369294164457) q[11];
rz(1.5849356687894962) q[11];
ry(1.5690441697465372) q[12];
rz(-3.0442530728404917) q[12];
ry(2.8987445262691427) q[13];
rz(-1.5729288764264204) q[13];
ry(0.006148147689282623) q[14];
rz(-1.4881229573611663) q[14];
ry(-0.010731041174063746) q[15];
rz(-1.5563935980857626) q[15];
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
ry(4.264022712640363e-05) q[0];
rz(0.5552483565877584) q[0];
ry(3.141537391622927) q[1];
rz(-1.455275139081184) q[1];
ry(0.00010895951312861662) q[2];
rz(-0.7188555564248338) q[2];
ry(-8.666858218268915e-05) q[3];
rz(-0.2735533767605931) q[3];
ry(3.141473098210251) q[4];
rz(-1.3882026379501333) q[4];
ry(3.140686258011838) q[5];
rz(0.0639967826514658) q[5];
ry(3.141287514301394) q[6];
rz(2.6399908467252877) q[6];
ry(3.141439140234691) q[7];
rz(-1.8378252139725104) q[7];
ry(0.0005066926309244929) q[8];
rz(2.451484566855836) q[8];
ry(-3.139077952087924) q[9];
rz(-1.7292548318140373) q[9];
ry(3.138973314195248) q[10];
rz(-3.117780530583788) q[10];
ry(3.1075375152164795) q[11];
rz(2.3282287779272735) q[11];
ry(-1.568874634044171) q[12];
rz(0.6629539393491033) q[12];
ry(-2.8773467911146233) q[13];
rz(2.0253804689889896) q[13];
ry(1.5734393351706775) q[14];
rz(-3.1412592803023305) q[14];
ry(0.24940293962529037) q[15];
rz(-1.0395495784870903) q[15];
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
ry(-0.22576295074212394) q[0];
rz(1.5707693276981123) q[0];
ry(2.680630455015641) q[1];
rz(-1.5707338147836372) q[1];
ry(-2.531035851346145) q[2];
rz(1.570722726681979) q[2];
ry(2.753241076314577) q[3];
rz(-1.5707221428789626) q[3];
ry(2.959340091107864) q[4];
rz(-1.5701267889056192) q[4];
ry(0.04938364759251534) q[5];
rz(-1.5651635857331767) q[5];
ry(-3.1329394543909657) q[6];
rz(-1.5857222610202768) q[6];
ry(-3.1401557238254623) q[7];
rz(1.418056893357888) q[7];
ry(0.005389781588420739) q[8];
rz(-1.4233706620348021) q[8];
ry(-0.0026496956967515576) q[9];
rz(-1.50463587313725) q[9];
ry(-3.140957866196096) q[10];
rz(0.1579851940682126) q[10];
ry(-0.00010835683913407205) q[11];
rz(0.8284280870874596) q[11];
ry(-3.1412449231092405) q[12];
rz(-0.9071034584907144) q[12];
ry(0.0005219618008887039) q[13];
rz(1.114569907023129) q[13];
ry(-1.5733270351136248) q[14];
rz(-9.087569367271073e-05) q[14];
ry(-0.00041423836733933683) q[15];
rz(1.0395678096284646) q[15];
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
ry(-1.5708082046647) q[0];
rz(0.15364653894653824) q[0];
ry(-1.5708209436163079) q[1];
rz(-0.34894722831108665) q[1];
ry(1.5708269973223699) q[2];
rz(-2.6340003772106946) q[2];
ry(1.570755626038441) q[3];
rz(-0.2759194370715914) q[3];
ry(-1.5708631893362837) q[4];
rz(-3.031654009544142) q[4];
ry(-1.5706867224788832) q[5];
rz(-3.129322774986978) q[5];
ry(1.5710963087361993) q[6];
rz(-0.02208468601335959) q[6];
ry(1.5705433802667095) q[7];
rz(0.010386294750870705) q[7];
ry(-1.570498214945522) q[8];
rz(0.0048513234972507036) q[8];
ry(1.5715070231111254) q[9];
rz(-0.0015392991112708914) q[9];
ry(1.572009013268744) q[10];
rz(3.140716749026225) q[10];
ry(-1.5709754043838444) q[11];
rz(-3.141483608418918) q[11];
ry(-1.569318488799772) q[12];
rz(2.502407065302848e-05) q[12];
ry(-1.5711504091054094) q[13];
rz(-0.00024174138247608866) q[13];
ry(1.5710516316099654) q[14];
rz(-3.1414542969028094) q[14];
ry(1.5710367183179015) q[15];
rz(3.141394746242274) q[15];