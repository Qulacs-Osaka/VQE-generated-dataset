OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.6436843874520459) q[0];
rz(-0.9127067982472795) q[0];
ry(-0.36630186856833036) q[1];
rz(0.9299876493950588) q[1];
ry(-0.0037878560171785065) q[2];
rz(-2.742357841023146) q[2];
ry(-0.006891327845192307) q[3];
rz(-1.3100610240030521) q[3];
ry(-2.2107742282476996) q[4];
rz(-1.4748833582726293) q[4];
ry(-1.1508106138612426) q[5];
rz(-0.7947835641724712) q[5];
ry(2.373361715028785) q[6];
rz(-1.526946530652741) q[6];
ry(0.19881505339427674) q[7];
rz(2.0224629252730164) q[7];
ry(-0.0007133975888073962) q[8];
rz(1.729842677130083) q[8];
ry(0.4325763342059723) q[9];
rz(2.966926231642214) q[9];
ry(0.453094082921642) q[10];
rz(0.277894820215038) q[10];
ry(-0.08367649401674271) q[11];
rz(-2.7827384675965785) q[11];
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
ry(0.006309984384325438) q[0];
rz(-0.42116560530408176) q[0];
ry(-2.747998180491042) q[1];
rz(-1.6424094646756715) q[1];
ry(-3.132621849954068) q[2];
rz(0.11648505921368973) q[2];
ry(-0.005516825156541232) q[3];
rz(-2.7379504623176) q[3];
ry(-2.7354206703419006) q[4];
rz(-0.8988382275786677) q[4];
ry(2.4737735493987953) q[5];
rz(1.990005504379157) q[5];
ry(1.6293152966359978) q[6];
rz(-2.8521418191178034) q[6];
ry(0.27776642983848365) q[7];
rz(1.6571173644123065) q[7];
ry(-3.139677738069691) q[8];
rz(1.2824766331404949) q[8];
ry(-1.8195537819982626) q[9];
rz(2.4786171709470355) q[9];
ry(-1.346918184568798) q[10];
rz(-3.128369108439447) q[10];
ry(-0.3908439470694594) q[11];
rz(2.6426449282278774) q[11];
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
ry(0.2679503064441613) q[0];
rz(-1.7391990544893359) q[0];
ry(1.1224521116826738) q[1];
rz(1.19207986445461) q[1];
ry(3.140302431926687) q[2];
rz(-0.39556686123840473) q[2];
ry(-3.0899683153511495) q[3];
rz(3.0357003448655218) q[3];
ry(2.2878093830455484) q[4];
rz(-1.2403740812162842) q[4];
ry(-1.8397939501630125) q[5];
rz(-2.6415646205404784) q[5];
ry(-0.6126354077560667) q[6];
rz(-0.4663545514544252) q[6];
ry(2.85315925798369) q[7];
rz(0.7140527912124693) q[7];
ry(1.8399713310662869) q[8];
rz(-1.6537369516227038) q[8];
ry(0.06678538583422267) q[9];
rz(0.41943058047590376) q[9];
ry(-2.175435939846838) q[10];
rz(1.8042104928251979) q[10];
ry(0.48710226423602665) q[11];
rz(1.7871554887221377) q[11];
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
ry(2.2426140355070485) q[0];
rz(1.2282294152096194) q[0];
ry(2.699175468291903) q[1];
rz(-1.4629352237834965) q[1];
ry(-2.9540835927741997) q[2];
rz(-1.0671322467317257) q[2];
ry(-3.1316538047266893) q[3];
rz(0.34450896157997807) q[3];
ry(0.9150745497586172) q[4];
rz(2.6134936036660177) q[4];
ry(0.13018286278739216) q[5];
rz(-2.1081421761176706) q[5];
ry(-3.1359858673499676) q[6];
rz(-2.101956670553327) q[6];
ry(-1.7728610294603984) q[7];
rz(0.5387046958026672) q[7];
ry(-3.078724420533904) q[8];
rz(0.5150677404242918) q[8];
ry(0.24324703807111533) q[9];
rz(-2.4536170710084746) q[9];
ry(0.23130625682879202) q[10];
rz(-0.18827271446204674) q[10];
ry(2.5008400606933554) q[11];
rz(1.8185288956186145) q[11];
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
ry(2.942657057907399) q[0];
rz(1.9870244312671037) q[0];
ry(1.2544842046649842) q[1];
rz(1.2408405190544471) q[1];
ry(3.139300162923047) q[2];
rz(2.6057721095078707) q[2];
ry(0.03558942775907159) q[3];
rz(1.4806753806119346) q[3];
ry(2.046614380215389) q[4];
rz(-3.0848179787074987) q[4];
ry(2.0099759577547966) q[5];
rz(-0.386387451825172) q[5];
ry(-1.0793353596166357) q[6];
rz(-1.7425142162384315) q[6];
ry(1.9866508153346691) q[7];
rz(-1.3466239961074289) q[7];
ry(-2.594144421144068) q[8];
rz(1.2596135506755601) q[8];
ry(-2.5275709631707977) q[9];
rz(1.592992176914235) q[9];
ry(-2.071974805114075) q[10];
rz(0.6518036879718477) q[10];
ry(2.591585937797672) q[11];
rz(1.3288103474542385) q[11];
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
ry(-1.2659920147218324) q[0];
rz(-1.6638648278606034) q[0];
ry(-1.5961384541983268) q[1];
rz(-0.7563795594932529) q[1];
ry(1.6665210425397259) q[2];
rz(-2.1475408940824536) q[2];
ry(0.003939549494551997) q[3];
rz(2.034181763140884) q[3];
ry(0.5244697128506562) q[4];
rz(-0.24269379405152328) q[4];
ry(-0.032792495838173785) q[5];
rz(-3.036600521978101) q[5];
ry(3.133787255037522) q[6];
rz(1.9442641850255387) q[6];
ry(3.1149610194715938) q[7];
rz(3.1275167193580886) q[7];
ry(-0.010554452801985031) q[8];
rz(-1.9164070702381677) q[8];
ry(1.8449743733786217) q[9];
rz(-2.394844653210702) q[9];
ry(3.116065261488837) q[10];
rz(-0.2838831355899601) q[10];
ry(0.9309452881168853) q[11];
rz(-1.2338038375145368) q[11];
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
ry(1.42346202573356) q[0];
rz(-1.4844850101027969) q[0];
ry(1.0196760792805122) q[1];
rz(1.0625757335013883) q[1];
ry(2.0918513289524205) q[2];
rz(-2.2499944285279123) q[2];
ry(1.5896786294858274) q[3];
rz(0.20161907514191096) q[3];
ry(-0.8477573586204832) q[4];
rz(-0.8048570324958917) q[4];
ry(-2.17390132653493) q[5];
rz(3.0094466575283065) q[5];
ry(0.9628713109149505) q[6];
rz(0.8179240159110419) q[6];
ry(2.781299340157678) q[7];
rz(0.3036023637476062) q[7];
ry(1.133103470927083) q[8];
rz(-1.8737725572271615) q[8];
ry(-0.7598099767816349) q[9];
rz(0.04588692483331602) q[9];
ry(0.7716879396224199) q[10];
rz(-1.0241339260890217) q[10];
ry(3.062012921527681) q[11];
rz(-0.6490658123448867) q[11];
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
ry(1.985349087789964) q[0];
rz(-0.5905660984227139) q[0];
ry(0.7956466813873986) q[1];
rz(0.31124725555842586) q[1];
ry(0.00899752171256253) q[2];
rz(2.262056174599925) q[2];
ry(-0.015528880465162181) q[3];
rz(-2.054486572594879) q[3];
ry(3.1136728395275077) q[4];
rz(0.6956738643800602) q[4];
ry(0.00111455389155999) q[5];
rz(-2.2487289858617276) q[5];
ry(-3.1404625109270428) q[6];
rz(-1.5870667707433754) q[6];
ry(-3.1267244801540603) q[7];
rz(-1.0949849294515444) q[7];
ry(0.0077826757252106304) q[8];
rz(-1.371752244106709) q[8];
ry(0.8317988869314714) q[9];
rz(2.7742118141332046) q[9];
ry(1.5636704517397864) q[10];
rz(1.2429022751676038) q[10];
ry(-0.43214486460066653) q[11];
rz(-2.4556418735561665) q[11];
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
ry(1.7440400122607242) q[0];
rz(-0.17414810929995816) q[0];
ry(-3.14049890904958) q[1];
rz(0.8495180016872693) q[1];
ry(2.0475803642768673) q[2];
rz(1.5911435077970466) q[2];
ry(0.05885934613547227) q[3];
rz(0.2909964813151496) q[3];
ry(1.213663725064655) q[4];
rz(1.4843302376685614) q[4];
ry(0.7346627888419873) q[5];
rz(-1.2960367596934255) q[5];
ry(-0.20019742188784398) q[6];
rz(-1.441445315006482) q[6];
ry(-0.2193053538331755) q[7];
rz(-3.072764902110526) q[7];
ry(-1.3608787359255603) q[8];
rz(0.7221866916423432) q[8];
ry(-2.345925433431245) q[9];
rz(-1.2423173340961426) q[9];
ry(2.7876980732589933) q[10];
rz(2.492687019428808) q[10];
ry(2.911970095676675) q[11];
rz(0.2937432375617055) q[11];
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
ry(-1.2164241456603415) q[0];
rz(-1.6451178974668794) q[0];
ry(-1.17084107719305) q[1];
rz(3.059139849384345) q[1];
ry(-2.9441029138841386) q[2];
rz(-1.0534012083268558) q[2];
ry(1.6978752275344136) q[3];
rz(1.5256196404299942) q[3];
ry(-0.006961411566113184) q[4];
rz(-2.0292828825676095) q[4];
ry(-0.002123906180226953) q[5];
rz(-1.1200361064025708) q[5];
ry(-3.1404854387346686) q[6];
rz(0.006802924089377883) q[6];
ry(3.1204258009714425) q[7];
rz(-1.5473030850157983) q[7];
ry(-0.003225924629276733) q[8];
rz(1.6987835398486784) q[8];
ry(-1.8541418870525028) q[9];
rz(2.6883429568464488) q[9];
ry(-0.9447317966832819) q[10];
rz(-1.706223402559937) q[10];
ry(-2.2585183638883652) q[11];
rz(-2.8912427413839215) q[11];
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
ry(1.7362256839260306) q[0];
rz(-1.066451143684061) q[0];
ry(-1.1686351936543042) q[1];
rz(-2.3411178944489324) q[1];
ry(-1.2269587757741842) q[2];
rz(3.1130947277229515) q[2];
ry(-1.632643424010833) q[3];
rz(0.9872209216380128) q[3];
ry(-1.231165720523279) q[4];
rz(1.8686347935492367) q[4];
ry(0.05546947442244665) q[5];
rz(-2.6425340375143276) q[5];
ry(1.8190624807092526) q[6];
rz(3.0254436850911453) q[6];
ry(-2.0335561059925946) q[7];
rz(1.4564164101324497) q[7];
ry(3.0430442995150235) q[8];
rz(-2.951092452940167) q[8];
ry(-1.507229182990453) q[9];
rz(-3.0295884265652506) q[9];
ry(0.5917793971582264) q[10];
rz(-2.68484018607454) q[10];
ry(0.6212309477990425) q[11];
rz(2.5893598613769195) q[11];
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
ry(2.50599290671016) q[0];
rz(-1.7323751201443425) q[0];
ry(-0.020847178006888362) q[1];
rz(-0.8668125335214292) q[1];
ry(-1.5483690198609719) q[2];
rz(0.4841958508090443) q[2];
ry(-0.027974289090330163) q[3];
rz(-2.4263360697947975) q[3];
ry(-3.1214792507508795) q[4];
rz(2.8239796643579287) q[4];
ry(-3.1236211486532146) q[5];
rz(1.320246771184962) q[5];
ry(-1.381438017626662) q[6];
rz(0.5926699521834927) q[6];
ry(1.7056426587434066) q[7];
rz(-2.0248280871251527) q[7];
ry(3.138310016789596) q[8];
rz(0.4920076222296404) q[8];
ry(-2.8025095294633555) q[9];
rz(-2.4028928980697337) q[9];
ry(2.449147640400859) q[10];
rz(-0.10582428100551491) q[10];
ry(-0.5025599223196817) q[11];
rz(-3.1136099846629564) q[11];
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
ry(2.443659928133231) q[0];
rz(-0.034861220638679136) q[0];
ry(0.9989396254665897) q[1];
rz(1.2004406441828435) q[1];
ry(2.4823984811634943) q[2];
rz(2.7577140933107214) q[2];
ry(0.9752544941855434) q[3];
rz(-1.6708571325871961) q[3];
ry(0.06401454221477465) q[4];
rz(-0.07239189142148139) q[4];
ry(-3.139459049850431) q[5];
rz(-2.037607316637099) q[5];
ry(-3.133363041382172) q[6];
rz(-2.5880335941113453) q[6];
ry(-3.122845272155986) q[7];
rz(-1.1036621914205547) q[7];
ry(3.1367986699068147) q[8];
rz(2.2841794713175876) q[8];
ry(-3.135395542808169) q[9];
rz(-0.78844439465568) q[9];
ry(2.05819745818295) q[10];
rz(-1.8511458127395823) q[10];
ry(2.5658855329020254) q[11];
rz(-1.5995956968662195) q[11];
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
ry(1.6097640925960786) q[0];
rz(-0.68126578409217) q[0];
ry(0.4672768423118783) q[1];
rz(2.979678698119871) q[1];
ry(-0.23984058772213365) q[2];
rz(-2.3803897505425007) q[2];
ry(0.006151316505895288) q[3];
rz(1.511897631461629) q[3];
ry(3.113513328054502) q[4];
rz(2.836296090898124) q[4];
ry(-3.123742597943569) q[5];
rz(2.5318262293864073) q[5];
ry(-1.386091372120907) q[6];
rz(0.6436128262060716) q[6];
ry(1.2878346281877953) q[7];
rz(1.9151663627694946) q[7];
ry(0.00929072289621935) q[8];
rz(0.2486469478200819) q[8];
ry(0.20609902157556625) q[9];
rz(1.5925882174871937) q[9];
ry(-0.7301907641342299) q[10];
rz(2.2786045429943504) q[10];
ry(0.8463821318673981) q[11];
rz(-2.613326081414791) q[11];
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
ry(-0.9965234490253864) q[0];
rz(1.853263411508393) q[0];
ry(-3.048972373562124) q[1];
rz(3.068833835507891) q[1];
ry(1.4741690427782208) q[2];
rz(1.1597137890607156) q[2];
ry(-2.559481010309726) q[3];
rz(2.2752521464789703) q[3];
ry(1.4819598953427249) q[4];
rz(-2.9687813966239607) q[4];
ry(-0.01272356621651305) q[5];
rz(-3.1061643987239034) q[5];
ry(-3.1401185697045837) q[6];
rz(2.5086080913202067) q[6];
ry(-1.8956846782564334) q[7];
rz(-3.139239418231967) q[7];
ry(0.4642334967122594) q[8];
rz(0.20133879775437244) q[8];
ry(3.131304528699663) q[9];
rz(2.6623523299088423) q[9];
ry(-2.6663423353482316) q[10];
rz(2.8201596591338634) q[10];
ry(-1.4228856228500915) q[11];
rz(1.267078789012437) q[11];
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
ry(-2.3648591007829864) q[0];
rz(2.916509622533619) q[0];
ry(-1.299217328009217) q[1];
rz(0.4513404245951022) q[1];
ry(2.1696036515666304) q[2];
rz(-3.0959044018226716) q[2];
ry(0.0014188084256912235) q[3];
rz(2.150846908162964) q[3];
ry(-3.807409826084296e-05) q[4];
rz(0.5048834448999919) q[4];
ry(3.0988293086861467) q[5];
rz(-1.310354939282) q[5];
ry(-0.009469115245353876) q[6];
rz(0.30166683243985304) q[6];
ry(0.9909369720122205) q[7];
rz(1.3771029744403191) q[7];
ry(-3.1412961857442836) q[8];
rz(0.4047688385860786) q[8];
ry(-2.9538644016863786) q[9];
rz(-2.4650063053946547) q[9];
ry(-0.39778193243458215) q[10];
rz(-0.018840881415606074) q[10];
ry(-1.1695499614736118) q[11];
rz(0.4703224377383932) q[11];
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
ry(-0.0004698028111566712) q[0];
rz(-1.8838522430640845) q[0];
ry(-1.9076170752526787) q[1];
rz(-0.2201362046294822) q[1];
ry(-1.0205083729871234) q[2];
rz(-0.00823164533527372) q[2];
ry(2.3893690260205873) q[3];
rz(-0.41199010014007165) q[3];
ry(-2.434880584048717) q[4];
rz(0.549016043884281) q[4];
ry(1.178311885111567) q[5];
rz(-0.5059836857347575) q[5];
ry(-2.2582069745933815) q[6];
rz(-2.450991879515244) q[6];
ry(-0.9624426608454923) q[7];
rz(-2.543262144122486) q[7];
ry(0.7399884722285858) q[8];
rz(2.2478385953255637) q[8];
ry(-0.01088903998305669) q[9];
rz(-2.6999533421740596) q[9];
ry(-1.2808112254740847) q[10];
rz(-1.7619966951059514) q[10];
ry(-2.525464353955771) q[11];
rz(0.928617954103046) q[11];
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
ry(-2.9433501359739336) q[0];
rz(-1.5168215181860278) q[0];
ry(-2.743384083197126) q[1];
rz(0.3204969412942038) q[1];
ry(2.163450504176748) q[2];
rz(-0.8631734026918744) q[2];
ry(3.1226881801420014) q[3];
rz(2.3217583858347797) q[3];
ry(3.1397249327142234) q[4];
rz(-0.9348932149175945) q[4];
ry(-3.1164897577599278) q[5];
rz(0.8500085056647897) q[5];
ry(-3.129216269067584) q[6];
rz(2.6816827775332204) q[6];
ry(3.140671280666483) q[7];
rz(-0.8324296476792596) q[7];
ry(-0.016750609123283765) q[8];
rz(-2.109017856365159) q[8];
ry(0.10587084560342629) q[9];
rz(0.9375790626582603) q[9];
ry(2.474654560803562) q[10];
rz(2.644452847051197) q[10];
ry(1.8301130998535493) q[11];
rz(-0.8275704742342186) q[11];
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
ry(-1.3579522191657307) q[0];
rz(-2.141544861215743) q[0];
ry(-1.655697638999719) q[1];
rz(2.1770810087168178) q[1];
ry(0.8290616760575489) q[2];
rz(-0.1789417894697447) q[2];
ry(1.7966953750862875) q[3];
rz(2.431463390739569) q[3];
ry(1.3456782012898025) q[4];
rz(2.7686997626004013) q[4];
ry(-0.3899912062833608) q[5];
rz(0.5064806805426725) q[5];
ry(0.41516304528619313) q[6];
rz(-1.8755523279574937) q[6];
ry(-1.2855936499674776) q[7];
rz(0.7527629955247291) q[7];
ry(1.4687106134129184) q[8];
rz(1.5088401074834712) q[8];
ry(-0.03246853665349913) q[9];
rz(-0.9966831416505497) q[9];
ry(2.466790189083497) q[10];
rz(-2.9612062875495893) q[10];
ry(-1.3207760201500038) q[11];
rz(-1.0358576294509465) q[11];
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
ry(-1.6873037549846452) q[0];
rz(2.7284603132455842) q[0];
ry(3.136765719362129) q[1];
rz(-0.9067323002235113) q[1];
ry(3.0530546148340036) q[2];
rz(-1.7833999102052585) q[2];
ry(-3.1268697547698365) q[3];
rz(-2.2016806555749757) q[3];
ry(3.1395832718630183) q[4];
rz(-3.045547228387499) q[4];
ry(3.099851700411157) q[5];
rz(1.259940514157072) q[5];
ry(-0.031633069881012055) q[6];
rz(-0.00839234848543611) q[6];
ry(-3.1368199745457437) q[7];
rz(-0.10773343713053579) q[7];
ry(-0.016810389688684695) q[8];
rz(2.388807202057745) q[8];
ry(2.9890901021419767) q[9];
rz(-0.5817042330142081) q[9];
ry(-3.1410935374102666) q[10];
rz(1.2841515253676299) q[10];
ry(-1.961035235955472) q[11];
rz(1.297086326195243) q[11];
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
ry(-1.3564260470557357) q[0];
rz(-2.2092116926024) q[0];
ry(0.9888424733461034) q[1];
rz(1.1506539411320889) q[1];
ry(-1.0215407230741658) q[2];
rz(2.4450784267831667) q[2];
ry(1.5514541850584986) q[3];
rz(-0.3559673324840299) q[3];
ry(0.2503035966777585) q[4];
rz(1.0035111588998278) q[4];
ry(1.7287894604411167) q[5];
rz(0.2425571785318674) q[5];
ry(-1.663213373111083) q[6];
rz(1.107744431029779) q[6];
ry(1.7716577995790492) q[7];
rz(1.2467193940926173) q[7];
ry(-2.9656226832377164) q[8];
rz(-0.2210149417705782) q[8];
ry(-1.4686016496294734) q[9];
rz(-2.5167937994312033) q[9];
ry(1.1857585360548741) q[10];
rz(0.7859609308394502) q[10];
ry(-2.9616892097985974) q[11];
rz(-0.42292093021656896) q[11];