OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.096268019105581) q[0];
rz(-1.0832718150702605) q[0];
ry(1.7680068504075495) q[1];
rz(-2.621601127918965) q[1];
ry(3.08602234788662) q[2];
rz(0.474725571505092) q[2];
ry(-3.126599791385201) q[3];
rz(2.5620270769280267) q[3];
ry(-1.5709465156436921) q[4];
rz(-2.894224608024271) q[4];
ry(1.5731592754146413) q[5];
rz(-0.3168973970724419) q[5];
ry(-3.139327975627982) q[6];
rz(2.7581740755268758) q[6];
ry(0.000576221788936096) q[7];
rz(-3.139690695586048) q[7];
ry(1.6107815715769318) q[8];
rz(3.0117356074367736) q[8];
ry(1.6057802162689088) q[9];
rz(-0.13239058629207162) q[9];
ry(-3.1293793276899073) q[10];
rz(0.10554607284078354) q[10];
ry(0.020011359812144747) q[11];
rz(2.8931866082766566) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.26310575957539584) q[0];
rz(-2.652786670506844) q[0];
ry(2.481574162571155) q[1];
rz(2.532283138072208) q[1];
ry(1.2120176328547547) q[2];
rz(2.118723919311991) q[2];
ry(0.9384776651406391) q[3];
rz(-0.6126712727308726) q[3];
ry(2.227627752745219) q[4];
rz(0.45003033115042174) q[4];
ry(2.2060930111733965) q[5];
rz(0.058595982362701804) q[5];
ry(-0.8066592018653882) q[6];
rz(0.8776748762092114) q[6];
ry(2.88161629507312) q[7];
rz(-3.043371422885826) q[7];
ry(0.7959032369179324) q[8];
rz(0.15910779399303895) q[8];
ry(-0.7626062403102711) q[9];
rz(0.03474810958084974) q[9];
ry(-1.926822405077722) q[10];
rz(2.606370861810154) q[10];
ry(1.8612225351180902) q[11];
rz(2.5632817859819403) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.808490047390866) q[0];
rz(-1.1390619885357076) q[0];
ry(-1.6485516737696386) q[1];
rz(-0.9532561895905856) q[1];
ry(1.0108177521222235) q[2];
rz(3.0100470131110075) q[2];
ry(-2.1459355288380744) q[3];
rz(2.0131397976459544) q[3];
ry(-1.5832660275307089) q[4];
rz(1.1846669996151373) q[4];
ry(1.5797760505963794) q[5];
rz(-1.3906971357587812) q[5];
ry(-0.5948871574200744) q[6];
rz(-2.399764219284342) q[6];
ry(-0.1186841198429276) q[7];
rz(1.9158460737087446) q[7];
ry(1.4579562032717046) q[8];
rz(1.7661209160954696) q[8];
ry(1.6706417697793272) q[9];
rz(2.733725637103682) q[9];
ry(1.3556526348813343) q[10];
rz(-1.4520634934183212) q[10];
ry(-0.5454143951513158) q[11];
rz(2.844680307457765) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.014602731979721) q[0];
rz(-2.1880187507537556) q[0];
ry(1.9915045110367735) q[1];
rz(3.0212548868232716) q[1];
ry(0.7246435579114054) q[2];
rz(2.192412094641576) q[2];
ry(0.6393616150333069) q[3];
rz(-0.6445013432500549) q[3];
ry(-0.010233253874519661) q[4];
rz(-2.4005865051383086) q[4];
ry(3.1147693982847793) q[5];
rz(2.957919468417704) q[5];
ry(1.6877685500824264) q[6];
rz(-3.00148823914165) q[6];
ry(-0.0572210126938435) q[7];
rz(-1.906146598906645) q[7];
ry(3.1342106272102352) q[8];
rz(-2.5320517983746567) q[8];
ry(-3.1273385816154002) q[9];
rz(-1.6854211675358135) q[9];
ry(-2.8013572090980947) q[10];
rz(-0.7269865133257413) q[10];
ry(-1.105791718069943) q[11];
rz(-2.563725012964655) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-0.19374578319933455) q[0];
rz(-2.934217852930719) q[0];
ry(-0.8057841437560782) q[1];
rz(-2.2241848742809345) q[1];
ry(-0.22765835576423488) q[2];
rz(-0.08162336524508502) q[2];
ry(2.3498576196804044) q[3];
rz(-1.8403404513698505) q[3];
ry(1.5305378624191457) q[4];
rz(1.5779062382841385) q[4];
ry(-1.535185117032909) q[5];
rz(-1.5750341429546033) q[5];
ry(2.8033897557357985) q[6];
rz(2.750153756437344) q[6];
ry(1.5757063973025844) q[7];
rz(-3.0825851152640973) q[7];
ry(0.5336582353728163) q[8];
rz(-1.8784568825775185) q[8];
ry(-0.8015881920467134) q[9];
rz(2.909132668322969) q[9];
ry(0.7822167942446546) q[10];
rz(3.0322745229882786) q[10];
ry(-1.1356018427893348) q[11];
rz(1.207266210490893) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.1714015883964617) q[0];
rz(0.053285751848350826) q[0];
ry(-2.7107287846798496) q[1];
rz(2.1823443381081638) q[1];
ry(-1.0526773314342126) q[2];
rz(-2.8737793290724523) q[2];
ry(-2.8504381235519474) q[3];
rz(2.7196780372631393) q[3];
ry(-1.5929890552122672) q[4];
rz(-2.950707935586618) q[4];
ry(-1.6013756959170646) q[5];
rz(-0.19758248778228865) q[5];
ry(2.717911721353863) q[6];
rz(3.0161543723834523) q[6];
ry(-2.601815783447817) q[7];
rz(-2.9496247584776523) q[7];
ry(-2.4183506632675424) q[8];
rz(-3.1391566769682564) q[8];
ry(-1.577027640726871) q[9];
rz(0.7827529309671722) q[9];
ry(-3.1114514777471713) q[10];
rz(1.6111115534549363) q[10];
ry(-0.24003817098925972) q[11];
rz(0.257235459062383) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.8526619464585892) q[0];
rz(-1.7908215265935918) q[0];
ry(-0.8528271587420949) q[1];
rz(-2.8478951859617063) q[1];
ry(3.0327688910449244) q[2];
rz(-0.08566088684876137) q[2];
ry(1.6063976974687746) q[3];
rz(0.6065819290689118) q[3];
ry(2.1589490359130004) q[4];
rz(-1.5499142617539698) q[4];
ry(-0.5338869204126292) q[5];
rz(0.3576761123807717) q[5];
ry(1.564165301169603) q[6];
rz(-0.018930117750847412) q[6];
ry(0.024599697194706102) q[7];
rz(2.7871494342489944) q[7];
ry(1.3892031477573872) q[8];
rz(-1.578908712615669) q[8];
ry(1.7232091275458261) q[9];
rz(-2.6782626093824504) q[9];
ry(1.5891843643193864) q[10];
rz(1.5591015253006053) q[10];
ry(1.626321120768793) q[11];
rz(-0.016796862659164354) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.6996489818035796) q[0];
rz(1.3231097814084931) q[0];
ry(-0.0410690280933077) q[1];
rz(0.8925367943055883) q[1];
ry(-1.5930259846381443) q[2];
rz(1.6491677904702957) q[2];
ry(3.115404030167838) q[3];
rz(-0.994515591016656) q[3];
ry(1.5578392147312616) q[4];
rz(-3.048037526377456) q[4];
ry(0.09393942495881942) q[5];
rz(1.2555250827430937) q[5];
ry(1.6214370649239558) q[6];
rz(0.3316610978998531) q[6];
ry(-3.1147313558608727) q[7];
rz(0.29749036426571923) q[7];
ry(1.5584373399569442) q[8];
rz(1.0078496630175326) q[8];
ry(3.1313737870070963) q[9];
rz(2.384588014310611) q[9];
ry(1.565230540539143) q[10];
rz(-0.642487400056745) q[10];
ry(-1.6635954691917025) q[11];
rz(1.5337351009900795) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.140020353405737) q[0];
rz(0.8917155948711413) q[0];
ry(-2.340641610279348) q[1];
rz(-0.983097493065923) q[1];
ry(0.020999253857197853) q[2];
rz(2.084139995530579) q[2];
ry(-0.042242389375728884) q[3];
rz(-2.995701059457627) q[3];
ry(1.5390727662906931) q[4];
rz(-2.0225650980167185) q[4];
ry(-0.22437201624949754) q[5];
rz(-3.089114923701865) q[5];
ry(-0.0036334365428842275) q[6];
rz(1.8057639177948284) q[6];
ry(0.0013654386347514702) q[7];
rz(2.98357388247856) q[7];
ry(-3.1249796698099015) q[8];
rz(0.9932363895145997) q[8];
ry(1.5579370657459748) q[9];
rz(2.894912961182926) q[9];
ry(-0.019441642716410357) q[10];
rz(-0.9883955345228452) q[10];
ry(1.5660347750044217) q[11];
rz(1.137879095734527) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.015269879174762304) q[0];
rz(-1.2515582640712042) q[0];
ry(3.1214487846794774) q[1];
rz(2.1325068117986197) q[1];
ry(-3.135030458302248) q[2];
rz(-2.577170262772537) q[2];
ry(2.0923115922501014) q[3];
rz(-1.5857578798583385) q[3];
ry(3.0706273720208834) q[4];
rz(2.5167493893323245) q[4];
ry(-0.23866401815832902) q[5];
rz(3.0673980666349703) q[5];
ry(-3.1413581038991403) q[6];
rz(0.5968186638948323) q[6];
ry(-0.0013127353587147894) q[7];
rz(2.2377785700866695) q[7];
ry(-0.8055922967890953) q[8];
rz(3.1307975392457763) q[8];
ry(3.120038095664257) q[9];
rz(0.9344667064972397) q[9];
ry(1.3456708517355025) q[10];
rz(2.1255735394275526) q[10];
ry(-1.7262475387093272) q[11];
rz(-1.59061158795077) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.1386078288675185) q[0];
rz(-1.563699641109144) q[0];
ry(0.8229733210601431) q[1];
rz(1.5842282169473672) q[1];
ry(1.5650211194472208) q[2];
rz(1.597353109814533) q[2];
ry(1.5660819959414107) q[3];
rz(0.8150016249329972) q[3];
ry(-2.945682215616448) q[4];
rz(-0.8527702310940075) q[4];
ry(-0.807044386417247) q[5];
rz(-1.8840743782870097) q[5];
ry(-2.4256128858174364) q[6];
rz(1.67971904284952) q[6];
ry(0.007382435306618902) q[7];
rz(1.9633217835971992) q[7];
ry(-0.006811484097257292) q[8];
rz(-3.111078340023503) q[8];
ry(3.00896722324953) q[9];
rz(1.1576113876231293) q[9];
ry(0.011124667601007077) q[10];
rz(-0.9712618438560421) q[10];
ry(-0.01950118693922505) q[11];
rz(-1.736060039276106) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(0.005566108138554071) q[0];
rz(0.021984380516968775) q[0];
ry(1.5719461796840166) q[1];
rz(-1.776245475621086) q[1];
ry(2.313499404972928) q[2];
rz(2.904566324783784) q[2];
ry(-0.0037260998502475218) q[3];
rz(2.5282328506452365) q[3];
ry(3.137531409247114) q[4];
rz(2.6810209150716378) q[4];
ry(3.139369709808579) q[5];
rz(2.833300899962296) q[5];
ry(-0.0003075263931720518) q[6];
rz(-0.08744010538921568) q[6];
ry(-3.138664100974855) q[7];
rz(0.098535156631403) q[7];
ry(1.5542802062146084) q[8];
rz(1.6262161750539395) q[8];
ry(-1.5663528582050823) q[9];
rz(1.5695058261406956) q[9];
ry(3.0780131652086538) q[10];
rz(-0.4231898358491249) q[10];
ry(1.6020388179265697) q[11];
rz(2.04207488394188) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5628151736211455) q[0];
rz(2.7575236877467186) q[0];
ry(3.1329919040841188) q[1];
rz(2.855175006609226) q[1];
ry(0.018101105202365072) q[2];
rz(-1.3335120718670062) q[2];
ry(-1.5291944384164244) q[3];
rz(-1.534336330280515) q[3];
ry(2.909535704865445) q[4];
rz(-0.9385455356626707) q[4];
ry(-1.5772411614962099) q[5];
rz(-2.028517038712276) q[5];
ry(-1.5884906341277647) q[6];
rz(-0.26769908997005487) q[6];
ry(1.6628873472625676) q[7];
rz(-2.8819400095859944) q[7];
ry(3.135575787457305) q[8];
rz(0.0346576261075082) q[8];
ry(3.1413074533185203) q[9];
rz(-0.006165752921085539) q[9];
ry(0.06568041763182246) q[10];
rz(1.7670418296679262) q[10];
ry(1.6157418599337425) q[11];
rz(1.7028940106911253e-05) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-2.796175807969554) q[0];
rz(-0.35873046862693764) q[0];
ry(-1.4324771996952175) q[1];
rz(-1.220630731009323) q[1];
ry(1.5691758324228775) q[2];
rz(-2.9830160054526793) q[2];
ry(1.5780535758071186) q[3];
rz(2.8660860071525676) q[3];
ry(-3.1389940868649338) q[4];
rz(2.0846665458485747) q[4];
ry(-0.00440845471270368) q[5];
rz(-0.36178189333086697) q[5];
ry(-1.7700140415577437e-05) q[6];
rz(-1.270658265295955) q[6];
ry(3.14117418730191) q[7];
rz(-1.1026995185282153) q[7];
ry(1.575548784954801) q[8];
rz(-2.739400106556918) q[8];
ry(1.5885252567810257) q[9];
rz(-2.708566268559009) q[9];
ry(0.01756280904347128) q[10];
rz(1.9803565488342485) q[10];
ry(0.11508009639078676) q[11];
rz(2.885133428656784) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-3.0399823706077935) q[0];
rz(-3.1149370299410206) q[0];
ry(-3.1144238301149483) q[1];
rz(-1.1290757914190963) q[1];
ry(3.105412935455763) q[2];
rz(1.695460180001896) q[2];
ry(3.1386186185842404) q[3];
rz(1.8099361028958176) q[3];
ry(1.722182381792714) q[4];
rz(1.5450383033790365) q[4];
ry(-1.883490276278095) q[5];
rz(3.076217579391291) q[5];
ry(1.6573300590279274) q[6];
rz(-2.1174550730430117) q[6];
ry(0.027812154211046547) q[7];
rz(-1.4915733118259766) q[7];
ry(-1.5771925742268635) q[8];
rz(-3.0244386880544782) q[8];
ry(3.136605513911889) q[9];
rz(2.500980109251986) q[9];
ry(-0.11926259036218045) q[10];
rz(0.9438683987977033) q[10];
ry(1.5674472894204834) q[11];
rz(0.2184211415339936) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.2453347349968193) q[0];
rz(-0.11309803176386876) q[0];
ry(0.15662866669916295) q[1];
rz(-1.6555067657352294) q[1];
ry(2.9932762867881517) q[2];
rz(-0.005356938782015598) q[2];
ry(0.0002622283733546779) q[3];
rz(3.1409733760206144) q[3];
ry(-3.1403460352794794) q[4];
rz(1.5556095865429058) q[4];
ry(0.0014883258856128734) q[5];
rz(0.4656715545621868) q[5];
ry(-3.139632409783985) q[6];
rz(1.4176549626939856) q[6];
ry(0.0020610575512804985) q[7];
rz(-2.898397661792605) q[7];
ry(-3.091502495243042) q[8];
rz(-1.4541814950521714) q[8];
ry(-0.0032507611263288107) q[9];
rz(-1.8389017952488054) q[9];
ry(0.9481646524720135) q[10];
rz(-3.1282567805546493) q[10];
ry(1.3908501993090974) q[11];
rz(2.1867818857675383) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(-1.9649402169701797) q[0];
rz(-2.3883182910049303) q[0];
ry(1.7015486316465005) q[1];
rz(3.1362414690092884) q[1];
ry(-0.28891555633423316) q[2];
rz(2.5679258911150873) q[2];
ry(-3.139492493555903) q[3];
rz(-1.8444351575530409) q[3];
ry(-1.7195401649167907) q[4];
rz(-1.9206971326056674) q[4];
ry(-2.8024207195065935) q[5];
rz(-1.320424526374722) q[5];
ry(-2.0164283859807313) q[6];
rz(-1.7552762544783818) q[6];
ry(2.0909410746378416) q[7];
rz(0.7352711184231557) q[7];
ry(0.9321645816131117) q[8];
rz(-1.2755456879387586) q[8];
ry(-3.129890419240393) q[9];
rz(-2.5273633544485543) q[9];
ry(-1.4761598842903383) q[10];
rz(0.14334270789366207) q[10];
ry(-0.1371938429557833) q[11];
rz(-3.095385264295214) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(3.112810286242667) q[0];
rz(2.3491719445124595) q[0];
ry(1.5081814028929323) q[1];
rz(-0.005764707595994267) q[1];
ry(-3.1370306874881577) q[2];
rz(2.5511279948701473) q[2];
ry(-3.1389723988386424) q[3];
rz(-0.7958866477949607) q[3];
ry(-0.0003693908090198761) q[4];
rz(2.2654386737471386) q[4];
ry(-3.1407488812933892) q[5];
rz(-2.8719277876355176) q[5];
ry(1.5701396000766907) q[6];
rz(3.1376386818018265) q[6];
ry(-1.5701306625726263) q[7];
rz(0.0015135875037809708) q[7];
ry(0.000627050638458691) q[8];
rz(1.2740482517556524) q[8];
ry(0.00048363409959368696) q[9];
rz(-2.252886856642081) q[9];
ry(0.005254986769510417) q[10];
rz(2.814052338680359) q[10];
ry(-1.4365052077354585) q[11];
rz(-0.6679439401633882) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.558136127229265) q[0];
rz(1.628420604167882) q[0];
ry(1.443099532842511) q[1];
rz(-1.084256270649825) q[1];
ry(-0.024656100175847066) q[2];
rz(-1.5085329558635596) q[2];
ry(1.651836838623138) q[3];
rz(1.116712984212601) q[3];
ry(-0.001477273738180962) q[4];
rz(1.3401437498189246) q[4];
ry(0.0008202293429402374) q[5];
rz(1.1784775630247895) q[5];
ry(1.5721221489713035) q[6];
rz(0.09571342107782321) q[6];
ry(-1.5712150471333919) q[7];
rz(-2.099768766579853) q[7];
ry(-3.1155190644038453) q[8];
rz(0.0006842400480197539) q[8];
ry(3.1336629083988887) q[9];
rz(1.0091356883291405) q[9];
ry(0.5035676938071513) q[10];
rz(1.4275391063274387) q[10];
ry(3.0434863740374576) q[11];
rz(2.428735040400381) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
ry(1.5738079141975385) q[0];
rz(2.118977412026445) q[0];
ry(-0.005985370162693937) q[1];
rz(0.7305365632064422) q[1];
ry(-2.949510830000001) q[2];
rz(-1.0151153602231826) q[2];
ry(3.141486788879794) q[3];
rz(-2.413230387457598) q[3];
ry(-1.5712719197438885) q[4];
rz(0.541360870917434) q[4];
ry(1.5699992313089526) q[5];
rz(-1.9551625726165578) q[5];
ry(2.3056159182068736) q[6];
rz(1.5691706933293006) q[6];
ry(1.6782605639541635) q[7];
rz(0.7242645913531964) q[7];
ry(-1.5691384448369279) q[8];
rz(2.03716994283461) q[8];
ry(-0.0056509115207923405) q[9];
rz(-0.11725138376112239) q[9];
ry(-0.019294795048663712) q[10];
rz(2.216460084969418) q[10];
ry(1.7155722698184714) q[11];
rz(-2.0144744993871226) q[11];