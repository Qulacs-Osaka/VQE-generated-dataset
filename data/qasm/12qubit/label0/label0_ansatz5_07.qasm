OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.914974591965459) q[0];
ry(-3.1250284477832606) q[1];
cx q[0],q[1];
ry(-1.78450109390225) q[0];
ry(-1.8625811365761438) q[1];
cx q[0],q[1];
ry(-0.6833861744013747) q[2];
ry(1.4960503617566543) q[3];
cx q[2],q[3];
ry(-2.4252429816826515) q[2];
ry(-1.875597798598169) q[3];
cx q[2],q[3];
ry(1.8204655715075937) q[4];
ry(1.632062907655065) q[5];
cx q[4],q[5];
ry(-1.7318991651273465) q[4];
ry(-2.6746211297341116) q[5];
cx q[4],q[5];
ry(-0.27660232108188243) q[6];
ry(-1.1475046372893367) q[7];
cx q[6],q[7];
ry(-0.22063283267896633) q[6];
ry(-2.2735556992339547) q[7];
cx q[6],q[7];
ry(0.13463211538185127) q[8];
ry(-1.4330045387105041) q[9];
cx q[8],q[9];
ry(-2.936732849670204) q[8];
ry(2.212744065711319) q[9];
cx q[8],q[9];
ry(-0.8527909764925344) q[10];
ry(-1.7019121558427504) q[11];
cx q[10],q[11];
ry(2.314569185707183) q[10];
ry(-1.4367190951276805) q[11];
cx q[10],q[11];
ry(-2.0239406816785577) q[1];
ry(2.043276182761741) q[2];
cx q[1],q[2];
ry(-1.452915664391681) q[1];
ry(2.5611591057716527) q[2];
cx q[1],q[2];
ry(-2.172276339719473) q[3];
ry(-2.6661647244538154) q[4];
cx q[3],q[4];
ry(-0.005835452805694605) q[3];
ry(-3.1343853085680196) q[4];
cx q[3],q[4];
ry(-1.786259374900077) q[5];
ry(-3.0268906564439657) q[6];
cx q[5],q[6];
ry(-1.7297616093696566) q[5];
ry(2.100341512437357) q[6];
cx q[5],q[6];
ry(1.7654905718960174) q[7];
ry(-2.153742440337952) q[8];
cx q[7],q[8];
ry(-1.0141253243388448) q[7];
ry(0.34989796256836936) q[8];
cx q[7],q[8];
ry(0.6743883452201196) q[9];
ry(2.1009922358678477) q[10];
cx q[9],q[10];
ry(0.30998557976728913) q[9];
ry(0.335754662105664) q[10];
cx q[9],q[10];
ry(-2.3819384443308382) q[0];
ry(2.562912153609592) q[1];
cx q[0],q[1];
ry(0.7004332754815469) q[0];
ry(-2.110411422516015) q[1];
cx q[0],q[1];
ry(0.8419560575311567) q[2];
ry(3.0522801440146483) q[3];
cx q[2],q[3];
ry(1.1327855704852445) q[2];
ry(2.2598517634852056) q[3];
cx q[2],q[3];
ry(0.8091578662216646) q[4];
ry(2.967694846709892) q[5];
cx q[4],q[5];
ry(0.03730034721432762) q[4];
ry(-0.22342228146494225) q[5];
cx q[4],q[5];
ry(-3.0126874620916513) q[6];
ry(1.3558124742835078) q[7];
cx q[6],q[7];
ry(0.0005047516959527205) q[6];
ry(-3.1404526093357967) q[7];
cx q[6],q[7];
ry(3.0677001638439867) q[8];
ry(2.0749715881861577) q[9];
cx q[8],q[9];
ry(1.1155027266884447) q[8];
ry(3.1327442951369906) q[9];
cx q[8],q[9];
ry(-0.5323047882415626) q[10];
ry(1.9657598847708933) q[11];
cx q[10],q[11];
ry(-2.9038434003015983) q[10];
ry(2.990409164790366) q[11];
cx q[10],q[11];
ry(2.8293026224051383) q[1];
ry(-2.7830616769262053) q[2];
cx q[1],q[2];
ry(-0.0164849964821121) q[1];
ry(2.8705666506639833) q[2];
cx q[1],q[2];
ry(-0.8080942422710509) q[3];
ry(-2.7451155368405824) q[4];
cx q[3],q[4];
ry(2.571695835892217) q[3];
ry(0.00011192768773401457) q[4];
cx q[3],q[4];
ry(-0.761807299949492) q[5];
ry(-1.9218481851000035) q[6];
cx q[5],q[6];
ry(2.3465520223889973) q[5];
ry(2.7780422762182333) q[6];
cx q[5],q[6];
ry(-0.9100716348362805) q[7];
ry(-1.6158835744666695) q[8];
cx q[7],q[8];
ry(1.3931929061007085) q[7];
ry(-0.3465580755784714) q[8];
cx q[7],q[8];
ry(2.7074895129011165) q[9];
ry(-1.7822237875341411) q[10];
cx q[9],q[10];
ry(2.8244266127570112) q[9];
ry(0.03027575675978905) q[10];
cx q[9],q[10];
ry(1.3531026995900088) q[0];
ry(1.0212510697454986) q[1];
cx q[0],q[1];
ry(-2.681910780668948) q[0];
ry(-0.3126186681382279) q[1];
cx q[0],q[1];
ry(2.863619242387188) q[2];
ry(-2.460507933645565) q[3];
cx q[2],q[3];
ry(3.1318719258767356) q[2];
ry(-1.0363129901663601) q[3];
cx q[2],q[3];
ry(0.49856678383211234) q[4];
ry(1.1807729548526895) q[5];
cx q[4],q[5];
ry(-0.09053598163094392) q[4];
ry(-0.11926385941548062) q[5];
cx q[4],q[5];
ry(-1.7603562773914834) q[6];
ry(-2.732734090392317) q[7];
cx q[6],q[7];
ry(0.0016576790030479098) q[6];
ry(3.1355827833724077) q[7];
cx q[6],q[7];
ry(-1.9912349861758587) q[8];
ry(0.9423688805341502) q[9];
cx q[8],q[9];
ry(0.29391825198482735) q[8];
ry(0.26570475262026694) q[9];
cx q[8],q[9];
ry(-0.4942255062806718) q[10];
ry(0.9027480594976006) q[11];
cx q[10],q[11];
ry(-1.9103361610937402) q[10];
ry(-0.4509950005205869) q[11];
cx q[10],q[11];
ry(-3.0827814353926133) q[1];
ry(0.8589791396623454) q[2];
cx q[1],q[2];
ry(0.28967860065498735) q[1];
ry(3.098282076631851) q[2];
cx q[1],q[2];
ry(1.427235530918921) q[3];
ry(-2.6291069236769284) q[4];
cx q[3],q[4];
ry(0.30450805500414263) q[3];
ry(0.0034053607945665476) q[4];
cx q[3],q[4];
ry(0.628413452583919) q[5];
ry(-0.007832390126894451) q[6];
cx q[5],q[6];
ry(-0.3099533762217086) q[5];
ry(-0.19429966242372299) q[6];
cx q[5],q[6];
ry(1.1402782924662038) q[7];
ry(-1.7908034051530972) q[8];
cx q[7],q[8];
ry(2.8106168102312488) q[7];
ry(2.012239701772784) q[8];
cx q[7],q[8];
ry(0.25575682992610993) q[9];
ry(-0.9722945521237181) q[10];
cx q[9],q[10];
ry(2.605543874019241) q[9];
ry(-0.8850798609604551) q[10];
cx q[9],q[10];
ry(-3.005088651537853) q[0];
ry(-1.5747059705540867) q[1];
cx q[0],q[1];
ry(2.7022322672924415) q[0];
ry(-2.474532400495481) q[1];
cx q[0],q[1];
ry(-0.9420971785078498) q[2];
ry(-1.5358257259859522) q[3];
cx q[2],q[3];
ry(-2.750562688726997) q[2];
ry(-2.8178027952824936) q[3];
cx q[2],q[3];
ry(0.5238562818385235) q[4];
ry(0.6990288829159367) q[5];
cx q[4],q[5];
ry(3.1275445054497113) q[4];
ry(0.5976895255226787) q[5];
cx q[4],q[5];
ry(-0.9896696047741761) q[6];
ry(1.7883899286181641) q[7];
cx q[6],q[7];
ry(3.1409569115922205) q[6];
ry(-3.139838589962277) q[7];
cx q[6],q[7];
ry(1.1587315924956822) q[8];
ry(-0.7728723218001579) q[9];
cx q[8],q[9];
ry(-2.0620191094399427) q[8];
ry(0.18790984094625904) q[9];
cx q[8],q[9];
ry(-2.7961578953865214) q[10];
ry(2.7349223363498085) q[11];
cx q[10],q[11];
ry(0.3933561722085989) q[10];
ry(1.679043189083603) q[11];
cx q[10],q[11];
ry(-2.983616736163435) q[1];
ry(3.100944229388254) q[2];
cx q[1],q[2];
ry(-3.0853066897063113) q[1];
ry(3.0793276957463607) q[2];
cx q[1],q[2];
ry(-2.101136023950596) q[3];
ry(0.8615344436432659) q[4];
cx q[3],q[4];
ry(2.340995195902719) q[3];
ry(-2.2059430354142338) q[4];
cx q[3],q[4];
ry(-1.9011148296227853) q[5];
ry(0.6352019771414028) q[6];
cx q[5],q[6];
ry(-0.10794249410876323) q[5];
ry(-0.0746760103380459) q[6];
cx q[5],q[6];
ry(2.111160907762055) q[7];
ry(1.3226023310581363) q[8];
cx q[7],q[8];
ry(-0.3497617303469568) q[7];
ry(2.7080672045754404) q[8];
cx q[7],q[8];
ry(-0.601962578743173) q[9];
ry(0.8569516819874549) q[10];
cx q[9],q[10];
ry(-1.482747273251908) q[9];
ry(2.5450665020710903) q[10];
cx q[9],q[10];
ry(-1.2523458853355942) q[0];
ry(0.9551063123689998) q[1];
cx q[0],q[1];
ry(0.17161673630582897) q[0];
ry(2.577076406096035) q[1];
cx q[0],q[1];
ry(2.189630722139354) q[2];
ry(0.7598180031331528) q[3];
cx q[2],q[3];
ry(0.6328960629796692) q[2];
ry(-0.5840031192635005) q[3];
cx q[2],q[3];
ry(2.4547248843244547) q[4];
ry(-2.8321503161378008) q[5];
cx q[4],q[5];
ry(3.1248049062966428) q[4];
ry(0.0003098333797977304) q[5];
cx q[4],q[5];
ry(2.0283908879353003) q[6];
ry(2.0799307305485484) q[7];
cx q[6],q[7];
ry(-3.1409915446945127) q[6];
ry(3.128415159187883) q[7];
cx q[6],q[7];
ry(-0.2514780618204089) q[8];
ry(1.3793978046170985) q[9];
cx q[8],q[9];
ry(1.2467823609195174) q[8];
ry(-0.48301353566984595) q[9];
cx q[8],q[9];
ry(-1.1700276880088047) q[10];
ry(0.4000630024459526) q[11];
cx q[10],q[11];
ry(0.5423052831498516) q[10];
ry(-1.4405033036359711) q[11];
cx q[10],q[11];
ry(-1.6030466211514547) q[1];
ry(-1.7481751204638005) q[2];
cx q[1],q[2];
ry(3.1238294519917975) q[1];
ry(1.9064252088111866) q[2];
cx q[1],q[2];
ry(2.350018864383687) q[3];
ry(0.8705255992904419) q[4];
cx q[3],q[4];
ry(1.7688434147906207) q[3];
ry(3.0646847218263122) q[4];
cx q[3],q[4];
ry(-1.9865654115418554) q[5];
ry(-1.1715663409260912) q[6];
cx q[5],q[6];
ry(1.1991518433271962) q[5];
ry(0.19121659469416652) q[6];
cx q[5],q[6];
ry(-0.9746765461801888) q[7];
ry(1.6024641599765153) q[8];
cx q[7],q[8];
ry(-2.950411515779664) q[7];
ry(-0.9615190413142701) q[8];
cx q[7],q[8];
ry(0.9801324632164179) q[9];
ry(-1.91685622473668) q[10];
cx q[9],q[10];
ry(-0.266485486895026) q[9];
ry(0.07663020469673298) q[10];
cx q[9],q[10];
ry(-0.5771402724055514) q[0];
ry(1.2608187613512243) q[1];
cx q[0],q[1];
ry(0.11937840343693028) q[0];
ry(-1.585792095933738) q[1];
cx q[0],q[1];
ry(-0.7471742257362566) q[2];
ry(2.3313111683615744) q[3];
cx q[2],q[3];
ry(2.1585204305320236) q[2];
ry(0.11003012330231342) q[3];
cx q[2],q[3];
ry(2.820422837544731) q[4];
ry(2.334864521714567) q[5];
cx q[4],q[5];
ry(0.00230621212370013) q[4];
ry(3.125492608616677) q[5];
cx q[4],q[5];
ry(0.4147766357388951) q[6];
ry(-0.3900140618090493) q[7];
cx q[6],q[7];
ry(-3.139082150988003) q[6];
ry(0.0017035789168417825) q[7];
cx q[6],q[7];
ry(1.3741245143132854) q[8];
ry(2.3390645946420743) q[9];
cx q[8],q[9];
ry(-1.1685350627811082) q[8];
ry(0.8305070481129695) q[9];
cx q[8],q[9];
ry(1.7627520912919128) q[10];
ry(-0.5798273600285214) q[11];
cx q[10],q[11];
ry(-1.8939614298651684) q[10];
ry(-1.3927593033648569) q[11];
cx q[10],q[11];
ry(2.7019360364495717) q[1];
ry(-1.8397985473041052) q[2];
cx q[1],q[2];
ry(0.0856755772276452) q[1];
ry(0.7613749988831678) q[2];
cx q[1],q[2];
ry(-2.177682571897567) q[3];
ry(2.5076006618439) q[4];
cx q[3],q[4];
ry(0.9503453119303403) q[3];
ry(-0.005344885661708604) q[4];
cx q[3],q[4];
ry(1.7765711386539618) q[5];
ry(0.6474890514290781) q[6];
cx q[5],q[6];
ry(0.3183365321932921) q[5];
ry(-0.14282626696985457) q[6];
cx q[5],q[6];
ry(1.739432391419252) q[7];
ry(-1.4257029785237318) q[8];
cx q[7],q[8];
ry(2.985276341946621) q[7];
ry(1.0577285237884175) q[8];
cx q[7],q[8];
ry(-3.1387941637457017) q[9];
ry(-1.1468807242239316) q[10];
cx q[9],q[10];
ry(3.0614491752129314) q[9];
ry(-0.024091917475778938) q[10];
cx q[9],q[10];
ry(-2.9868136071380365) q[0];
ry(1.5326929422821314) q[1];
cx q[0],q[1];
ry(-0.4213556370686069) q[0];
ry(1.1055546515227468) q[1];
cx q[0],q[1];
ry(0.17105492334231087) q[2];
ry(2.5670394739747415) q[3];
cx q[2],q[3];
ry(-0.980036869734599) q[2];
ry(2.5573219659743933) q[3];
cx q[2],q[3];
ry(0.3694616559668613) q[4];
ry(1.3945542404134006) q[5];
cx q[4],q[5];
ry(-3.068984677320321) q[4];
ry(-0.014121527794165088) q[5];
cx q[4],q[5];
ry(0.7940333910607338) q[6];
ry(0.16636850477147114) q[7];
cx q[6],q[7];
ry(-0.04270001526233891) q[6];
ry(2.9785074773864375) q[7];
cx q[6],q[7];
ry(-2.2027765199237295) q[8];
ry(-0.711129990302851) q[9];
cx q[8],q[9];
ry(2.088232926516131) q[8];
ry(-2.7900134238332415) q[9];
cx q[8],q[9];
ry(-3.06950699901454) q[10];
ry(0.40480100911396727) q[11];
cx q[10],q[11];
ry(0.01601802576541022) q[10];
ry(-0.6921640305216156) q[11];
cx q[10],q[11];
ry(0.8610635764114111) q[1];
ry(1.0374599305347065) q[2];
cx q[1],q[2];
ry(0.19858407179088822) q[1];
ry(0.5410043690995913) q[2];
cx q[1],q[2];
ry(3.091047178985829) q[3];
ry(0.7838316219830403) q[4];
cx q[3],q[4];
ry(-0.05063646892781091) q[3];
ry(-0.0029660573467191753) q[4];
cx q[3],q[4];
ry(1.6561124534019385) q[5];
ry(0.9981303767632143) q[6];
cx q[5],q[6];
ry(0.012390109653216186) q[5];
ry(-0.07736505515362908) q[6];
cx q[5],q[6];
ry(2.3022762538597616) q[7];
ry(-1.9906545613179627) q[8];
cx q[7],q[8];
ry(-0.01893849671744796) q[7];
ry(-2.9519588934801733) q[8];
cx q[7],q[8];
ry(1.4513054829965053) q[9];
ry(-2.5863896774334316) q[10];
cx q[9],q[10];
ry(2.1496124112024804) q[9];
ry(2.9102103591685227) q[10];
cx q[9],q[10];
ry(1.2795912699506045) q[0];
ry(-3.0505857146945465) q[1];
cx q[0],q[1];
ry(2.403173312506141) q[0];
ry(-0.0015178212668658375) q[1];
cx q[0],q[1];
ry(2.6839709831504495) q[2];
ry(2.332897792634802) q[3];
cx q[2],q[3];
ry(-1.1053161565250957) q[2];
ry(-0.756285486256055) q[3];
cx q[2],q[3];
ry(0.8573951682076713) q[4];
ry(-0.743918998067098) q[5];
cx q[4],q[5];
ry(3.027533697344806) q[4];
ry(-0.12403983968117595) q[5];
cx q[4],q[5];
ry(3.0713316440800758) q[6];
ry(3.098658585589027) q[7];
cx q[6],q[7];
ry(0.04651644033435242) q[6];
ry(-3.129588782158911) q[7];
cx q[6],q[7];
ry(-0.1203479378603447) q[8];
ry(1.4205738870859816) q[9];
cx q[8],q[9];
ry(0.5217749970898646) q[8];
ry(-0.06381272656707715) q[9];
cx q[8],q[9];
ry(0.15607812073655047) q[10];
ry(-0.24710700729302904) q[11];
cx q[10],q[11];
ry(0.42668652520310957) q[10];
ry(-3.0789311509739457) q[11];
cx q[10],q[11];
ry(1.7870626413306878) q[1];
ry(2.954682992989648) q[2];
cx q[1],q[2];
ry(-3.129646205992584) q[1];
ry(-2.44290499236603) q[2];
cx q[1],q[2];
ry(-3.01655784819313) q[3];
ry(0.8060757226531488) q[4];
cx q[3],q[4];
ry(-0.0008497510220067552) q[3];
ry(0.011995748016130677) q[4];
cx q[3],q[4];
ry(-0.05618897270695822) q[5];
ry(-0.5876808983539146) q[6];
cx q[5],q[6];
ry(0.3846247279773545) q[5];
ry(0.0236657986037212) q[6];
cx q[5],q[6];
ry(1.2678454275605384) q[7];
ry(2.5833669951701173) q[8];
cx q[7],q[8];
ry(0.0072857305819739315) q[7];
ry(-0.7670159474722363) q[8];
cx q[7],q[8];
ry(1.0501924459628371) q[9];
ry(-2.559623648438083) q[10];
cx q[9],q[10];
ry(1.458678578704986) q[9];
ry(2.07569026921478) q[10];
cx q[9],q[10];
ry(-0.1736978026982889) q[0];
ry(2.494393481107464) q[1];
cx q[0],q[1];
ry(1.1392174547688405) q[0];
ry(-2.724916680279781) q[1];
cx q[0],q[1];
ry(1.5985876784544093) q[2];
ry(0.18416560460975973) q[3];
cx q[2],q[3];
ry(0.6095364141676693) q[2];
ry(-0.008193371067928721) q[3];
cx q[2],q[3];
ry(-2.6226777002251107) q[4];
ry(0.7645222298559453) q[5];
cx q[4],q[5];
ry(-0.736623965349283) q[4];
ry(0.06227849623835802) q[5];
cx q[4],q[5];
ry(-1.405690215781354) q[6];
ry(2.788013484288459) q[7];
cx q[6],q[7];
ry(0.01918710780898536) q[6];
ry(0.09111378141895714) q[7];
cx q[6],q[7];
ry(-0.21370380205897554) q[8];
ry(1.591549787449458) q[9];
cx q[8],q[9];
ry(0.06234897099240655) q[8];
ry(-0.10645464975023344) q[9];
cx q[8],q[9];
ry(-0.758275250645018) q[10];
ry(-2.7347647725612667) q[11];
cx q[10],q[11];
ry(1.1059163342287803) q[10];
ry(-2.896388625323302) q[11];
cx q[10],q[11];
ry(-0.8231598549300578) q[1];
ry(2.902812760376574) q[2];
cx q[1],q[2];
ry(1.3671211558066585) q[1];
ry(-1.060572317062552) q[2];
cx q[1],q[2];
ry(0.7417339345430278) q[3];
ry(-2.421869772334101) q[4];
cx q[3],q[4];
ry(3.135199326725472) q[3];
ry(3.1342505061671964) q[4];
cx q[3],q[4];
ry(-1.6124405219911992) q[5];
ry(-1.769928043829223) q[6];
cx q[5],q[6];
ry(-0.049396569426459536) q[5];
ry(0.26107659673823863) q[6];
cx q[5],q[6];
ry(1.8027464981982648) q[7];
ry(-0.20267208765346137) q[8];
cx q[7],q[8];
ry(-3.140139230190131) q[7];
ry(-3.094580017675859) q[8];
cx q[7],q[8];
ry(-1.3612326492162958) q[9];
ry(1.1792817524672472) q[10];
cx q[9],q[10];
ry(2.6594882328925933) q[9];
ry(-0.5145505295156546) q[10];
cx q[9],q[10];
ry(-2.207935445334704) q[0];
ry(2.944916170549018) q[1];
cx q[0],q[1];
ry(-0.04330204853518982) q[0];
ry(-0.4960311527206151) q[1];
cx q[0],q[1];
ry(-0.1495720036011311) q[2];
ry(-2.0875914293546196) q[3];
cx q[2],q[3];
ry(3.1273172210418574) q[2];
ry(0.04095226582251676) q[3];
cx q[2],q[3];
ry(1.4601726242245887) q[4];
ry(-1.576212072206095) q[5];
cx q[4],q[5];
ry(1.0038755918324984) q[4];
ry(0.1542689635176444) q[5];
cx q[4],q[5];
ry(1.3335875495683742) q[6];
ry(-2.5273346933017105) q[7];
cx q[6],q[7];
ry(0.02031127862313509) q[6];
ry(-0.10290480579652073) q[7];
cx q[6],q[7];
ry(-0.08558976886064268) q[8];
ry(0.02173948118324809) q[9];
cx q[8],q[9];
ry(0.08909078458713093) q[8];
ry(-2.655531455922165) q[9];
cx q[8],q[9];
ry(-2.1304565873744155) q[10];
ry(-1.1959050785790688) q[11];
cx q[10],q[11];
ry(2.4636267555423124) q[10];
ry(-1.1240078060139833) q[11];
cx q[10],q[11];
ry(-1.2780874407881901) q[1];
ry(3.0237639439431776) q[2];
cx q[1],q[2];
ry(-1.6850655120566134) q[1];
ry(-1.9288027078422063) q[2];
cx q[1],q[2];
ry(-2.3751877140323248) q[3];
ry(-1.5860262451340734) q[4];
cx q[3],q[4];
ry(0.14518974224867953) q[3];
ry(-0.02431047360056038) q[4];
cx q[3],q[4];
ry(-2.869103782663707) q[5];
ry(0.16802353985079957) q[6];
cx q[5],q[6];
ry(0.001975569269093036) q[5];
ry(-0.015255630142777345) q[6];
cx q[5],q[6];
ry(-0.07863753416707342) q[7];
ry(1.820871451168361) q[8];
cx q[7],q[8];
ry(-2.858247415403412) q[7];
ry(-3.0857742451089165) q[8];
cx q[7],q[8];
ry(-0.5961596847491011) q[9];
ry(2.684464806956844) q[10];
cx q[9],q[10];
ry(2.9200377320858) q[9];
ry(-0.2846920098066681) q[10];
cx q[9],q[10];
ry(-1.1289710690149786) q[0];
ry(-2.359380661563864) q[1];
ry(-1.9633937618311914) q[2];
ry(1.7741935453838025) q[3];
ry(1.6687971759879772) q[4];
ry(-1.9065658808019048) q[5];
ry(-3.0413126990508035) q[6];
ry(2.6371858101363705) q[7];
ry(-1.262282086322677) q[8];
ry(0.10159366497444376) q[9];
ry(-0.46518436719998135) q[10];
ry(1.6618962987149777) q[11];