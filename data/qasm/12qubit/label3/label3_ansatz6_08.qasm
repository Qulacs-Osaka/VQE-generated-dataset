OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.5842791517936909) q[0];
ry(2.1266255801851286) q[1];
cx q[0],q[1];
ry(1.5938027943080229) q[0];
ry(-2.411202254427891) q[1];
cx q[0],q[1];
ry(1.0827136037874876) q[1];
ry(-0.6038360387353775) q[2];
cx q[1],q[2];
ry(-1.2520339659337365) q[1];
ry(-0.8275575458675402) q[2];
cx q[1],q[2];
ry(-1.0531309945266474) q[2];
ry(-1.1372049930642953) q[3];
cx q[2],q[3];
ry(3.1140975552532306) q[2];
ry(3.0186842273586003) q[3];
cx q[2],q[3];
ry(-1.1300609909852695) q[3];
ry(0.06573092726638663) q[4];
cx q[3],q[4];
ry(-1.9125824471314765) q[3];
ry(1.3211009408054082) q[4];
cx q[3],q[4];
ry(-0.9763245122721829) q[4];
ry(-2.7631214490873286) q[5];
cx q[4],q[5];
ry(1.102663708138358) q[4];
ry(2.051789430107828) q[5];
cx q[4],q[5];
ry(-0.8021334200557897) q[5];
ry(3.0408925771301805) q[6];
cx q[5],q[6];
ry(1.6016813193376525) q[5];
ry(-1.5396244946829452) q[6];
cx q[5],q[6];
ry(-1.363927291803434) q[6];
ry(-2.9361819331955896) q[7];
cx q[6],q[7];
ry(-0.8607148481428544) q[6];
ry(1.5337311100227304) q[7];
cx q[6],q[7];
ry(1.4025589026811485) q[7];
ry(-1.4388808921496468) q[8];
cx q[7],q[8];
ry(-2.6071938455346735) q[7];
ry(-2.0771814331813987) q[8];
cx q[7],q[8];
ry(1.4581594416800403) q[8];
ry(2.5573190145969527) q[9];
cx q[8],q[9];
ry(1.6914997613477987) q[8];
ry(0.7032561221068729) q[9];
cx q[8],q[9];
ry(2.7403883368016175) q[9];
ry(-1.5778589861264045) q[10];
cx q[9],q[10];
ry(0.8997410179099692) q[9];
ry(1.3825730295117253) q[10];
cx q[9],q[10];
ry(2.4620276895166495) q[10];
ry(-3.1083793982218646) q[11];
cx q[10],q[11];
ry(-0.787672872681803) q[10];
ry(-0.15820193547214453) q[11];
cx q[10],q[11];
ry(1.0876981737913534) q[0];
ry(-1.94991897637418) q[1];
cx q[0],q[1];
ry(-0.04387242829436654) q[0];
ry(-3.0219926913779833) q[1];
cx q[0],q[1];
ry(-1.1021434430484005) q[1];
ry(-0.7675572634205946) q[2];
cx q[1],q[2];
ry(1.428738787651774) q[1];
ry(1.263122555896055) q[2];
cx q[1],q[2];
ry(1.8117002590659341) q[2];
ry(1.597485653017685) q[3];
cx q[2],q[3];
ry(-2.519860693774399) q[2];
ry(1.7543843368274237) q[3];
cx q[2],q[3];
ry(-1.8210136948929279) q[3];
ry(-3.08104338366265) q[4];
cx q[3],q[4];
ry(0.03654072986304485) q[3];
ry(-1.5429407362588523) q[4];
cx q[3],q[4];
ry(2.3031609159944906) q[4];
ry(-3.079723343388274) q[5];
cx q[4],q[5];
ry(-0.9480730078399162) q[4];
ry(-1.3799481443350432) q[5];
cx q[4],q[5];
ry(1.185842550192115) q[5];
ry(-0.5329355140493132) q[6];
cx q[5],q[6];
ry(0.11079808449088002) q[5];
ry(0.005552098795822826) q[6];
cx q[5],q[6];
ry(3.024881443089532) q[6];
ry(2.0156921397985403) q[7];
cx q[6],q[7];
ry(3.0006096285258455) q[6];
ry(1.4124193611499234) q[7];
cx q[6],q[7];
ry(-0.7538833711753435) q[7];
ry(3.131428476604278) q[8];
cx q[7],q[8];
ry(-1.4185341037966674) q[7];
ry(2.614178324184673) q[8];
cx q[7],q[8];
ry(-0.1625236779448971) q[8];
ry(-1.5773769164597837) q[9];
cx q[8],q[9];
ry(-1.5869701669464948) q[8];
ry(-0.01239798706595635) q[9];
cx q[8],q[9];
ry(1.9558668148407428) q[9];
ry(1.4393173149210676) q[10];
cx q[9],q[10];
ry(-0.5511750288455168) q[9];
ry(-0.4848672331821522) q[10];
cx q[9],q[10];
ry(0.706383352532244) q[10];
ry(1.221675989463193) q[11];
cx q[10],q[11];
ry(-0.4327120118671096) q[10];
ry(0.9025965294861312) q[11];
cx q[10],q[11];
ry(-1.2114293456173977) q[0];
ry(-0.5639542879548838) q[1];
cx q[0],q[1];
ry(-2.797855429022607) q[0];
ry(0.440561826462475) q[1];
cx q[0],q[1];
ry(0.7830899907821097) q[1];
ry(-0.5943748615556359) q[2];
cx q[1],q[2];
ry(-0.5163474538156612) q[1];
ry(-1.0598222878850452) q[2];
cx q[1],q[2];
ry(-0.2139209013619867) q[2];
ry(0.3151693262925719) q[3];
cx q[2],q[3];
ry(-1.446879292353888) q[2];
ry(1.2332480054841446) q[3];
cx q[2],q[3];
ry(0.8872161105982783) q[3];
ry(-0.47152553808275494) q[4];
cx q[3],q[4];
ry(-0.07222126122437464) q[3];
ry(0.1523586794781772) q[4];
cx q[3],q[4];
ry(2.046888126492502) q[4];
ry(-1.2214552425403895) q[5];
cx q[4],q[5];
ry(-0.08736578305215494) q[4];
ry(3.1039983621010148) q[5];
cx q[4],q[5];
ry(-1.1321834019354) q[5];
ry(-2.2763658084113683) q[6];
cx q[5],q[6];
ry(0.011632253689614735) q[5];
ry(-0.042120865341031426) q[6];
cx q[5],q[6];
ry(0.8043849825481191) q[6];
ry(-2.118709437863598) q[7];
cx q[6],q[7];
ry(-3.1329948263173772) q[6];
ry(2.9606396359525817) q[7];
cx q[6],q[7];
ry(0.29421148188871843) q[7];
ry(0.09793686674824896) q[8];
cx q[7],q[8];
ry(1.5926766184192358) q[7];
ry(-1.8651736724694006) q[8];
cx q[7],q[8];
ry(0.2321720429444433) q[8];
ry(-0.6591949365807687) q[9];
cx q[8],q[9];
ry(-2.3333938510448693) q[8];
ry(-0.03125381047835745) q[9];
cx q[8],q[9];
ry(-1.5242922386452327) q[9];
ry(-2.99624809003307) q[10];
cx q[9],q[10];
ry(1.9027171856619252) q[9];
ry(1.47220138523988) q[10];
cx q[9],q[10];
ry(-1.8479204267170186) q[10];
ry(-3.015938301315305) q[11];
cx q[10],q[11];
ry(-1.6645667473103751) q[10];
ry(0.5332200781538409) q[11];
cx q[10],q[11];
ry(1.739574393986192) q[0];
ry(0.11059737415644597) q[1];
cx q[0],q[1];
ry(2.0803755492950797) q[0];
ry(0.6360780799512239) q[1];
cx q[0],q[1];
ry(0.04218194478209458) q[1];
ry(2.366362026994936) q[2];
cx q[1],q[2];
ry(2.020562674698795) q[1];
ry(2.0463348074275665) q[2];
cx q[1],q[2];
ry(2.8272103462690485) q[2];
ry(-0.28877855909918515) q[3];
cx q[2],q[3];
ry(-1.0530461297989353) q[2];
ry(-0.13847557366471983) q[3];
cx q[2],q[3];
ry(-2.3550098017160805) q[3];
ry(-0.6128203685294551) q[4];
cx q[3],q[4];
ry(0.04407935388700953) q[3];
ry(-0.13793127287993556) q[4];
cx q[3],q[4];
ry(-0.35853522367676405) q[4];
ry(-2.8816181097226883) q[5];
cx q[4],q[5];
ry(-3.0453133916142354) q[4];
ry(0.3535028245679279) q[5];
cx q[4],q[5];
ry(0.752140902993359) q[5];
ry(-2.8277180654032326) q[6];
cx q[5],q[6];
ry(0.03519884163079201) q[5];
ry(-0.7851498214514052) q[6];
cx q[5],q[6];
ry(-1.8685719007595025) q[6];
ry(-0.318504103188775) q[7];
cx q[6],q[7];
ry(-2.245986515240313) q[6];
ry(2.0726229235025797) q[7];
cx q[6],q[7];
ry(-1.5522254634484358) q[7];
ry(1.3661732498582584) q[8];
cx q[7],q[8];
ry(0.004574929627162174) q[7];
ry(-1.755853563506018) q[8];
cx q[7],q[8];
ry(1.1633129881980144) q[8];
ry(-0.7761509558673216) q[9];
cx q[8],q[9];
ry(1.5594215809911214) q[8];
ry(-3.1146067747902175) q[9];
cx q[8],q[9];
ry(2.0243423624342083) q[9];
ry(0.7203156020092247) q[10];
cx q[9],q[10];
ry(2.3970398329802887) q[9];
ry(-0.08352169337080595) q[10];
cx q[9],q[10];
ry(0.18024034053453186) q[10];
ry(2.0530470810566377) q[11];
cx q[10],q[11];
ry(-2.4280989399485926) q[10];
ry(0.51927557079014) q[11];
cx q[10],q[11];
ry(-1.5716240717752537) q[0];
ry(-2.88117536130372) q[1];
cx q[0],q[1];
ry(1.6306070658672445) q[0];
ry(2.7982546945261526) q[1];
cx q[0],q[1];
ry(1.988568471520109) q[1];
ry(-2.11803706922861) q[2];
cx q[1],q[2];
ry(0.8533422274326794) q[1];
ry(-0.8322961211625186) q[2];
cx q[1],q[2];
ry(-2.6994204066678065) q[2];
ry(1.5348202873231263) q[3];
cx q[2],q[3];
ry(1.7726732654983424) q[2];
ry(3.0889092176873185) q[3];
cx q[2],q[3];
ry(-0.11497714479831611) q[3];
ry(-2.227112715440579) q[4];
cx q[3],q[4];
ry(1.751910403843894) q[3];
ry(-0.08343836976431884) q[4];
cx q[3],q[4];
ry(1.6652130410575472) q[4];
ry(-1.544587527727141) q[5];
cx q[4],q[5];
ry(-1.9777288875539494) q[4];
ry(2.9826632451843507) q[5];
cx q[4],q[5];
ry(0.8165610466873872) q[5];
ry(1.5752004229795897) q[6];
cx q[5],q[6];
ry(1.4768210035545346) q[5];
ry(3.1384336997188846) q[6];
cx q[5],q[6];
ry(-2.2605389308799615) q[6];
ry(1.5527628364666632) q[7];
cx q[6],q[7];
ry(-1.6780804180722408) q[6];
ry(-3.101122851298406) q[7];
cx q[6],q[7];
ry(-1.9832731995819455) q[7];
ry(2.3293583935321704) q[8];
cx q[7],q[8];
ry(1.3521219160542808) q[7];
ry(0.036499786038822535) q[8];
cx q[7],q[8];
ry(0.3618778076366969) q[8];
ry(-2.811311721760698) q[9];
cx q[8],q[9];
ry(0.054341513175185284) q[8];
ry(1.6047359803752945) q[9];
cx q[8],q[9];
ry(-1.2275066870997386) q[9];
ry(1.3393225689941692) q[10];
cx q[9],q[10];
ry(-1.437664768717435) q[9];
ry(-0.00142400170256618) q[10];
cx q[9],q[10];
ry(-2.07872937704036) q[10];
ry(-2.667915329489814) q[11];
cx q[10],q[11];
ry(-0.613811012299295) q[10];
ry(2.1250433280711105) q[11];
cx q[10],q[11];
ry(-2.2321108444817166) q[0];
ry(0.2458339301459305) q[1];
cx q[0],q[1];
ry(2.097483384222673) q[0];
ry(1.5203550203797191) q[1];
cx q[0],q[1];
ry(-2.370140004998088) q[1];
ry(2.6564446908444994) q[2];
cx q[1],q[2];
ry(1.6325630735625207) q[1];
ry(-1.536687251423167) q[2];
cx q[1],q[2];
ry(2.0275462625928613) q[2];
ry(1.612688028709072) q[3];
cx q[2],q[3];
ry(0.5228059192915255) q[2];
ry(1.7346187928072823) q[3];
cx q[2],q[3];
ry(2.900329178672726) q[3];
ry(-2.649188261357963) q[4];
cx q[3],q[4];
ry(-3.1335567709381156) q[3];
ry(-3.129078292398979) q[4];
cx q[3],q[4];
ry(0.7863434714514953) q[4];
ry(0.8202247601391811) q[5];
cx q[4],q[5];
ry(1.4683039467819876) q[4];
ry(-3.1263555146163924) q[5];
cx q[4],q[5];
ry(1.5314059243153884) q[5];
ry(-2.294134947465612) q[6];
cx q[5],q[6];
ry(1.5580678337968907) q[5];
ry(-0.5507201626957627) q[6];
cx q[5],q[6];
ry(-3.115291772533545) q[6];
ry(-2.828712480131776) q[7];
cx q[6],q[7];
ry(1.4817416880079664) q[6];
ry(-1.4398653832432533) q[7];
cx q[6],q[7];
ry(-2.9545786733690305) q[7];
ry(-0.06141570934210323) q[8];
cx q[7],q[8];
ry(-2.885248483246282) q[7];
ry(0.008477136018363357) q[8];
cx q[7],q[8];
ry(3.0927398981492664) q[8];
ry(1.7632139720033992) q[9];
cx q[8],q[9];
ry(1.2588778048567306) q[8];
ry(1.5536204904265727) q[9];
cx q[8],q[9];
ry(-1.7871899480514897) q[9];
ry(-2.8393659527776207) q[10];
cx q[9],q[10];
ry(3.028933388705177) q[9];
ry(-3.0581426360929473) q[10];
cx q[9],q[10];
ry(0.7228005949741366) q[10];
ry(-0.15678491886542825) q[11];
cx q[10],q[11];
ry(-0.6853094429104418) q[10];
ry(2.72072520196734) q[11];
cx q[10],q[11];
ry(-2.095372360155358) q[0];
ry(2.9548620041868787) q[1];
cx q[0],q[1];
ry(0.0313532563981509) q[0];
ry(1.056688586273717) q[1];
cx q[0],q[1];
ry(0.20321744504853806) q[1];
ry(-0.04487215129770039) q[2];
cx q[1],q[2];
ry(2.8595489467959587) q[1];
ry(0.16797657597585136) q[2];
cx q[1],q[2];
ry(2.1252382342802454) q[2];
ry(-0.05485165814691318) q[3];
cx q[2],q[3];
ry(-0.457807860440895) q[2];
ry(-1.6835637907904761) q[3];
cx q[2],q[3];
ry(-0.5434780109024819) q[3];
ry(2.289041287309223) q[4];
cx q[3],q[4];
ry(-2.0181209777947284) q[3];
ry(-0.2247183560740693) q[4];
cx q[3],q[4];
ry(-0.47299081522185465) q[4];
ry(-2.6491473884247276) q[5];
cx q[4],q[5];
ry(-0.004205372365081672) q[4];
ry(-3.141357423534841) q[5];
cx q[4],q[5];
ry(-0.45594333688957533) q[5];
ry(0.23252110447038007) q[6];
cx q[5],q[6];
ry(0.0287069090154649) q[5];
ry(-1.545800375940118) q[6];
cx q[5],q[6];
ry(2.5951325987058205) q[6];
ry(-2.984662167489411) q[7];
cx q[6],q[7];
ry(1.491418852306838) q[6];
ry(1.48406126597584) q[7];
cx q[6],q[7];
ry(0.05996298493253338) q[7];
ry(-1.9660456902882473) q[8];
cx q[7],q[8];
ry(-1.572895930470087) q[7];
ry(-1.5591502992745767) q[8];
cx q[7],q[8];
ry(-2.988581326351584) q[8];
ry(-1.6666241273014561) q[9];
cx q[8],q[9];
ry(1.57739369920111) q[8];
ry(-1.584935568876973) q[9];
cx q[8],q[9];
ry(-0.08872298983882153) q[9];
ry(-2.3006557392453857) q[10];
cx q[9],q[10];
ry(0.039317312275542804) q[9];
ry(-1.574403133311459) q[10];
cx q[9],q[10];
ry(-1.4391638582972242) q[10];
ry(1.8159653669697304) q[11];
cx q[10],q[11];
ry(-2.3650139860476456) q[10];
ry(1.5673742453094055) q[11];
cx q[10],q[11];
ry(1.3411706249390074) q[0];
ry(1.914618180200934) q[1];
cx q[0],q[1];
ry(-0.4002873310855341) q[0];
ry(-1.83186758098425) q[1];
cx q[0],q[1];
ry(-0.3644798380686449) q[1];
ry(0.5490004030444159) q[2];
cx q[1],q[2];
ry(0.581322587220086) q[1];
ry(-0.39725786334961205) q[2];
cx q[1],q[2];
ry(-0.8132581429418354) q[2];
ry(-1.8216364414593564) q[3];
cx q[2],q[3];
ry(-0.7164685310006171) q[2];
ry(2.984337573999757) q[3];
cx q[2],q[3];
ry(-1.1702216440329893) q[3];
ry(0.020883482708530403) q[4];
cx q[3],q[4];
ry(-0.14604719775949535) q[3];
ry(2.89595691516441) q[4];
cx q[3],q[4];
ry(0.09373449520083721) q[4];
ry(-2.83085796358974) q[5];
cx q[4],q[5];
ry(0.021680306460295284) q[4];
ry(-0.0009065555186261303) q[5];
cx q[4],q[5];
ry(-0.4084397861100202) q[5];
ry(0.18927173138154427) q[6];
cx q[5],q[6];
ry(-2.462237229828044) q[5];
ry(-3.0029471333638877) q[6];
cx q[5],q[6];
ry(-1.4990539820062314) q[6];
ry(2.3712365280615866) q[7];
cx q[6],q[7];
ry(0.004281778795390334) q[6];
ry(1.5720572994709274) q[7];
cx q[6],q[7];
ry(-2.3856440431265136) q[7];
ry(1.5647873410432103) q[8];
cx q[7],q[8];
ry(0.3979611845804394) q[7];
ry(-1.5639812316102262) q[8];
cx q[7],q[8];
ry(0.04023451425831581) q[8];
ry(1.6421713052161935) q[9];
cx q[8],q[9];
ry(-3.1122908162003773) q[8];
ry(0.11117094592485499) q[9];
cx q[8],q[9];
ry(-3.05460567084341) q[9];
ry(-0.18052685853810413) q[10];
cx q[9],q[10];
ry(1.6503968639359403) q[9];
ry(1.6131725805184782) q[10];
cx q[9],q[10];
ry(-0.10059093342819203) q[10];
ry(-2.178013962021012) q[11];
cx q[10],q[11];
ry(-0.6922217563361939) q[10];
ry(1.695013824849335) q[11];
cx q[10],q[11];
ry(0.2917665677727728) q[0];
ry(-0.8884578180979634) q[1];
cx q[0],q[1];
ry(0.49333043854193814) q[0];
ry(-1.879932908339267) q[1];
cx q[0],q[1];
ry(-2.7663652482684973) q[1];
ry(2.7853211170349885) q[2];
cx q[1],q[2];
ry(0.12220246654425004) q[1];
ry(-2.1530375951537626) q[2];
cx q[1],q[2];
ry(-0.13624530388131006) q[2];
ry(-1.1596877175944993) q[3];
cx q[2],q[3];
ry(-0.6589015498667167) q[2];
ry(-1.3464519177241652) q[3];
cx q[2],q[3];
ry(-1.2063601418391783) q[3];
ry(-1.4974382994123212) q[4];
cx q[3],q[4];
ry(2.636193759942987) q[3];
ry(0.002468845473225215) q[4];
cx q[3],q[4];
ry(-2.0285260449692797) q[4];
ry(-0.6198553762825196) q[5];
cx q[4],q[5];
ry(0.02668555213686652) q[4];
ry(-3.13964813514693) q[5];
cx q[4],q[5];
ry(-2.8305684575763497) q[5];
ry(-1.5793547259337801) q[6];
cx q[5],q[6];
ry(2.292756388690817) q[5];
ry(0.9734645668925169) q[6];
cx q[5],q[6];
ry(-1.8251203160571023) q[6];
ry(0.3628547307131258) q[7];
cx q[6],q[7];
ry(3.132870037341227) q[6];
ry(-0.018241426476400587) q[7];
cx q[6],q[7];
ry(1.9188318538246032) q[7];
ry(3.0687345390597973) q[8];
cx q[7],q[8];
ry(-3.021787500777574) q[7];
ry(-0.7333151671292777) q[8];
cx q[7],q[8];
ry(-1.561493688688084) q[8];
ry(-2.45655618242511) q[9];
cx q[8],q[9];
ry(3.1190460090890073) q[8];
ry(0.1650180233693757) q[9];
cx q[8],q[9];
ry(1.978422126484106) q[9];
ry(1.8614354252504564) q[10];
cx q[9],q[10];
ry(2.897691537069371) q[9];
ry(3.0497403241111285) q[10];
cx q[9],q[10];
ry(0.2984995842974998) q[10];
ry(2.695108423636954) q[11];
cx q[10],q[11];
ry(-1.4078291668317984) q[10];
ry(-1.490928258757667) q[11];
cx q[10],q[11];
ry(1.4082240499297196) q[0];
ry(1.1638718812151865) q[1];
cx q[0],q[1];
ry(2.9642596158036008) q[0];
ry(-2.411866988285654) q[1];
cx q[0],q[1];
ry(-0.28421879163765595) q[1];
ry(-3.060529666181185) q[2];
cx q[1],q[2];
ry(-2.1453913036876084) q[1];
ry(-1.5497900835863234) q[2];
cx q[1],q[2];
ry(-1.9059558183672405) q[2];
ry(-1.939328552091862) q[3];
cx q[2],q[3];
ry(-1.5835295431295853) q[2];
ry(-1.6644810795779523) q[3];
cx q[2],q[3];
ry(-0.9347464003436151) q[3];
ry(0.6211404704703265) q[4];
cx q[3],q[4];
ry(3.119325480499107) q[3];
ry(-0.04646289514098065) q[4];
cx q[3],q[4];
ry(-0.5018486079260676) q[4];
ry(2.756777040305486) q[5];
cx q[4],q[5];
ry(-0.014363817869098128) q[4];
ry(3.139954098329656) q[5];
cx q[4],q[5];
ry(-0.9829583457268569) q[5];
ry(2.469169072633106) q[6];
cx q[5],q[6];
ry(0.2398515238603381) q[5];
ry(-0.07035817464587878) q[6];
cx q[5],q[6];
ry(0.8820779200198874) q[6];
ry(-2.684139400820833) q[7];
cx q[6],q[7];
ry(-0.0025528629911848455) q[6];
ry(3.1105976614025916) q[7];
cx q[6],q[7];
ry(2.675973694098472) q[7];
ry(-2.080751387279879) q[8];
cx q[7],q[8];
ry(1.0063665805824673) q[7];
ry(-2.280932868664806) q[8];
cx q[7],q[8];
ry(2.426377770782552) q[8];
ry(2.7799968488509115) q[9];
cx q[8],q[9];
ry(-3.104014844960858) q[8];
ry(2.999434333510194) q[9];
cx q[8],q[9];
ry(-1.8828280942419922) q[9];
ry(3.136587355456871) q[10];
cx q[9],q[10];
ry(-0.22781708815113078) q[9];
ry(0.8752946729408895) q[10];
cx q[9],q[10];
ry(-3.051202582151097) q[10];
ry(2.331310912881475) q[11];
cx q[10],q[11];
ry(2.0400441468017796) q[10];
ry(-3.091003547711816) q[11];
cx q[10],q[11];
ry(1.272678831388954) q[0];
ry(2.7993186557673146) q[1];
cx q[0],q[1];
ry(-2.0135612888226424) q[0];
ry(1.6835895950088986) q[1];
cx q[0],q[1];
ry(3.0072294972215263) q[1];
ry(-1.1776082626003266) q[2];
cx q[1],q[2];
ry(1.5147225255457881) q[1];
ry(1.504470895226621) q[2];
cx q[1],q[2];
ry(3.127331831209597) q[2];
ry(0.8878469989107314) q[3];
cx q[2],q[3];
ry(-2.0557332869223415) q[2];
ry(1.5830642717148198) q[3];
cx q[2],q[3];
ry(3.1117790875667697) q[3];
ry(-0.7478989236954812) q[4];
cx q[3],q[4];
ry(1.6015341166850714) q[3];
ry(-1.5304693728024157) q[4];
cx q[3],q[4];
ry(2.9574630792038548) q[4];
ry(0.69727309364304) q[5];
cx q[4],q[5];
ry(3.1241399217693737) q[4];
ry(-3.1349749607711144) q[5];
cx q[4],q[5];
ry(-2.512250682193528) q[5];
ry(-1.6261650091879565) q[6];
cx q[5],q[6];
ry(-1.4150025466523346) q[5];
ry(3.0394619662712175) q[6];
cx q[5],q[6];
ry(-0.15522673872281345) q[6];
ry(1.439651884842187) q[7];
cx q[6],q[7];
ry(-3.131625602891729) q[6];
ry(-1.5578384100323674) q[7];
cx q[6],q[7];
ry(-1.7055618602570557) q[7];
ry(0.5954512034525323) q[8];
cx q[7],q[8];
ry(-1.9532171532693567) q[7];
ry(-1.581027921683404) q[8];
cx q[7],q[8];
ry(0.6111893222411284) q[8];
ry(-0.9775610402452288) q[9];
cx q[8],q[9];
ry(3.127039758759772) q[8];
ry(0.008590403860914364) q[9];
cx q[8],q[9];
ry(1.7383718419339234) q[9];
ry(-0.18653292723859471) q[10];
cx q[9],q[10];
ry(3.0733911430954217) q[9];
ry(-0.7045640595861784) q[10];
cx q[9],q[10];
ry(-1.9652244408344444) q[10];
ry(2.1938898008833947) q[11];
cx q[10],q[11];
ry(1.7241582768794181) q[10];
ry(-1.5148127068941677) q[11];
cx q[10],q[11];
ry(-0.24126022480487563) q[0];
ry(-0.10189327496760801) q[1];
ry(1.5575540513095747) q[2];
ry(-3.12444091404964) q[3];
ry(1.1533056141844007) q[4];
ry(-2.2690797691981377) q[5];
ry(-0.6833211509866168) q[6];
ry(-2.5279701301624615) q[7];
ry(-1.949691343711219) q[8];
ry(-2.207950620111796) q[9];
ry(0.9364307994615474) q[10];
ry(0.34522427165689223) q[11];