OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.04162735165256315) q[0];
ry(-2.172674830834267) q[1];
cx q[0],q[1];
ry(-2.4043806890699213) q[0];
ry(-1.4326254544105677) q[1];
cx q[0],q[1];
ry(0.5471985448007701) q[1];
ry(-3.0913367076544085) q[2];
cx q[1],q[2];
ry(0.7465900824817334) q[1];
ry(2.6473472079523037) q[2];
cx q[1],q[2];
ry(2.0352929217437996) q[2];
ry(-0.9058302200048782) q[3];
cx q[2],q[3];
ry(-0.35102522421914095) q[2];
ry(-0.5738869027580356) q[3];
cx q[2],q[3];
ry(-2.6980164041230936) q[3];
ry(1.5751730811291376) q[4];
cx q[3],q[4];
ry(0.4535475916122306) q[3];
ry(-2.5862531122087113) q[4];
cx q[3],q[4];
ry(1.3128588597102766) q[4];
ry(-0.6488843997076172) q[5];
cx q[4],q[5];
ry(1.8002583187945045) q[4];
ry(2.35536340743584) q[5];
cx q[4],q[5];
ry(-0.16743543066260624) q[5];
ry(-0.5657031208270462) q[6];
cx q[5],q[6];
ry(-0.013544589956427239) q[5];
ry(-0.013349892612922964) q[6];
cx q[5],q[6];
ry(0.1679074756395229) q[6];
ry(-1.483671950131984) q[7];
cx q[6],q[7];
ry(1.5909034800905104) q[6];
ry(0.17301741211956645) q[7];
cx q[6],q[7];
ry(-2.4940937310665583) q[7];
ry(2.595512696756456) q[8];
cx q[7],q[8];
ry(-1.8106626950832796) q[7];
ry(-1.6549279315899967) q[8];
cx q[7],q[8];
ry(1.4325810625934319) q[8];
ry(1.704398696110608) q[9];
cx q[8],q[9];
ry(1.5156027394287364) q[8];
ry(-0.008978852498602308) q[9];
cx q[8],q[9];
ry(1.2303164275286067) q[9];
ry(-2.9312738080122718) q[10];
cx q[9],q[10];
ry(-0.011700272645663165) q[9];
ry(-0.11045842012924734) q[10];
cx q[9],q[10];
ry(1.9087222231050385) q[10];
ry(-1.5465565183534) q[11];
cx q[10],q[11];
ry(-0.3773055509343927) q[10];
ry(-0.007184536207665385) q[11];
cx q[10],q[11];
ry(0.5453925122442286) q[0];
ry(1.3257700341556262) q[1];
cx q[0],q[1];
ry(1.0669842906943003) q[0];
ry(1.6897023445490715) q[1];
cx q[0],q[1];
ry(-1.0826837961572662) q[1];
ry(-1.824542201393953) q[2];
cx q[1],q[2];
ry(-0.10310693086666499) q[1];
ry(0.7460961873115892) q[2];
cx q[1],q[2];
ry(-2.854240432499725) q[2];
ry(-0.9114538831854332) q[3];
cx q[2],q[3];
ry(-0.003238789801469999) q[2];
ry(0.008397006679263797) q[3];
cx q[2],q[3];
ry(0.93323825607943) q[3];
ry(-0.546845912503362) q[4];
cx q[3],q[4];
ry(3.1231990340340947) q[3];
ry(1.2528660683144421) q[4];
cx q[3],q[4];
ry(2.958931310307299) q[4];
ry(-0.3058049851080305) q[5];
cx q[4],q[5];
ry(2.054122768793241) q[4];
ry(-0.6208643586564158) q[5];
cx q[4],q[5];
ry(-2.7929720683996098) q[5];
ry(-3.0571293400717403) q[6];
cx q[5],q[6];
ry(-0.015764289472294024) q[5];
ry(3.1382506615799124) q[6];
cx q[5],q[6];
ry(0.2725681692408628) q[6];
ry(1.0246254869388107) q[7];
cx q[6],q[7];
ry(1.4313903037235718) q[6];
ry(-1.0937661676755115) q[7];
cx q[6],q[7];
ry(0.2854362478718534) q[7];
ry(-2.584974216757526) q[8];
cx q[7],q[8];
ry(-3.06338313357515) q[7];
ry(1.6421204169943184) q[8];
cx q[7],q[8];
ry(-2.8596049983889764) q[8];
ry(1.1343519043956363) q[9];
cx q[8],q[9];
ry(-2.895404324079825) q[8];
ry(3.133295461650532) q[9];
cx q[8],q[9];
ry(3.0010722439992374) q[9];
ry(-1.3319556849422411) q[10];
cx q[9],q[10];
ry(1.685507966381857) q[9];
ry(0.15925078531285397) q[10];
cx q[9],q[10];
ry(0.43291169188421547) q[10];
ry(0.36007812897980784) q[11];
cx q[10],q[11];
ry(1.5885347509103875) q[10];
ry(-3.057701282963371) q[11];
cx q[10],q[11];
ry(1.888526503691267) q[0];
ry(1.2018254922037677) q[1];
cx q[0],q[1];
ry(-1.7901601042484918) q[0];
ry(1.2244259590066466) q[1];
cx q[0],q[1];
ry(-1.8506444376792768) q[1];
ry(2.958734049261536) q[2];
cx q[1],q[2];
ry(-0.7764495573490444) q[1];
ry(0.033838465983031085) q[2];
cx q[1],q[2];
ry(-1.0740832342185271) q[2];
ry(-1.6336421390814522) q[3];
cx q[2],q[3];
ry(-0.7775937840788254) q[2];
ry(-0.02769126821725898) q[3];
cx q[2],q[3];
ry(-2.516503918038328) q[3];
ry(-1.6205832293918419) q[4];
cx q[3],q[4];
ry(-2.53216494017656) q[3];
ry(0.7544373727199662) q[4];
cx q[3],q[4];
ry(0.9489566720619278) q[4];
ry(-0.0595436511563765) q[5];
cx q[4],q[5];
ry(-2.200814097928414) q[4];
ry(-2.494406074924876) q[5];
cx q[4],q[5];
ry(2.771129438871622) q[5];
ry(-1.7442676481952555) q[6];
cx q[5],q[6];
ry(-0.00013456616898643858) q[5];
ry(3.141514606541854) q[6];
cx q[5],q[6];
ry(-2.895708370425869) q[6];
ry(-1.4355260797012641) q[7];
cx q[6],q[7];
ry(-0.6110291941126382) q[6];
ry(-1.8448347901799043) q[7];
cx q[6],q[7];
ry(-1.3465534338779162) q[7];
ry(3.0493987021654303) q[8];
cx q[7],q[8];
ry(0.20832422916583493) q[7];
ry(-1.6805302103536253) q[8];
cx q[7],q[8];
ry(2.965665886618197) q[8];
ry(-2.6116910501443034) q[9];
cx q[8],q[9];
ry(0.19588112981648087) q[8];
ry(0.09766006026088547) q[9];
cx q[8],q[9];
ry(-1.804980717433196) q[9];
ry(2.474847049823662) q[10];
cx q[9],q[10];
ry(-3.0945339623632875) q[9];
ry(2.5645088922542842) q[10];
cx q[9],q[10];
ry(-2.1942782014667115) q[10];
ry(0.47986951915993187) q[11];
cx q[10],q[11];
ry(-2.413570186229676) q[10];
ry(0.014717725722212194) q[11];
cx q[10],q[11];
ry(1.8039575152526466) q[0];
ry(0.802775001328424) q[1];
cx q[0],q[1];
ry(-1.8193081711844477) q[0];
ry(1.408244418908198) q[1];
cx q[0],q[1];
ry(0.697615878926742) q[1];
ry(0.6740637721679148) q[2];
cx q[1],q[2];
ry(-1.6301719641794592) q[1];
ry(1.8200152893723738) q[2];
cx q[1],q[2];
ry(0.8111943469854224) q[2];
ry(0.06893330282816201) q[3];
cx q[2],q[3];
ry(3.102591884966728) q[2];
ry(-0.572662045757742) q[3];
cx q[2],q[3];
ry(1.0983626337335695) q[3];
ry(1.1918835698349177) q[4];
cx q[3],q[4];
ry(0.14293655364898952) q[3];
ry(-0.030009713149474815) q[4];
cx q[3],q[4];
ry(1.6461066832794657) q[4];
ry(2.169622477038279) q[5];
cx q[4],q[5];
ry(0.4030111897603313) q[4];
ry(-2.6287944145553004) q[5];
cx q[4],q[5];
ry(1.7194606574574527) q[5];
ry(0.5838987773236655) q[6];
cx q[5],q[6];
ry(3.140902870556114) q[5];
ry(0.00025787511148800114) q[6];
cx q[5],q[6];
ry(-2.843593955515567) q[6];
ry(1.9964266039546192) q[7];
cx q[6],q[7];
ry(-0.8046148906987529) q[6];
ry(1.370308809873795) q[7];
cx q[6],q[7];
ry(3.1220379622718832) q[7];
ry(-0.867598266689388) q[8];
cx q[7],q[8];
ry(-0.07825142428535509) q[7];
ry(2.1343066602588547) q[8];
cx q[7],q[8];
ry(3.0759634932575266) q[8];
ry(2.320743611164749) q[9];
cx q[8],q[9];
ry(2.080113242320988) q[8];
ry(2.241586715329994) q[9];
cx q[8],q[9];
ry(1.5487239464545364) q[9];
ry(1.1164063737826435) q[10];
cx q[9],q[10];
ry(-1.961977331493296) q[9];
ry(-2.94049194186294) q[10];
cx q[9],q[10];
ry(0.011823980788311594) q[10];
ry(-1.837584952975172) q[11];
cx q[10],q[11];
ry(-0.2784368916399913) q[10];
ry(-0.0616869018829611) q[11];
cx q[10],q[11];
ry(3.109430527083375) q[0];
ry(-2.6184263720329635) q[1];
cx q[0],q[1];
ry(-1.8915897356162434) q[0];
ry(2.190018370209137) q[1];
cx q[0],q[1];
ry(0.764674140695093) q[1];
ry(-0.49317410423096725) q[2];
cx q[1],q[2];
ry(-0.49704870587174305) q[1];
ry(1.0867458979731257) q[2];
cx q[1],q[2];
ry(0.19385138387034395) q[2];
ry(1.8297213941131665) q[3];
cx q[2],q[3];
ry(-0.22957661635430324) q[2];
ry(1.6457577606297882) q[3];
cx q[2],q[3];
ry(-1.4779536817852057) q[3];
ry(1.228261562906229) q[4];
cx q[3],q[4];
ry(-0.014101668656381926) q[3];
ry(3.125332369535859) q[4];
cx q[3],q[4];
ry(-2.7010579980488902) q[4];
ry(1.126720234478) q[5];
cx q[4],q[5];
ry(-0.8944557711224141) q[4];
ry(2.946458354975454) q[5];
cx q[4],q[5];
ry(-1.372522389187818) q[5];
ry(0.958975289266392) q[6];
cx q[5],q[6];
ry(3.1403881189214475) q[5];
ry(-3.1396115212938995) q[6];
cx q[5],q[6];
ry(-0.06742495928454151) q[6];
ry(2.023957706855045) q[7];
cx q[6],q[7];
ry(0.2491282402519296) q[6];
ry(2.999804181178093) q[7];
cx q[6],q[7];
ry(-2.5547651780106073) q[7];
ry(-1.0221615869248977) q[8];
cx q[7],q[8];
ry(-0.5572958670490166) q[7];
ry(2.6280667365493366) q[8];
cx q[7],q[8];
ry(-1.571673715647539) q[8];
ry(2.134118454633998) q[9];
cx q[8],q[9];
ry(0.1433957283125644) q[8];
ry(1.1601752007752109) q[9];
cx q[8],q[9];
ry(-1.6380915663275566) q[9];
ry(-1.06687323399525) q[10];
cx q[9],q[10];
ry(2.5702392500537203) q[9];
ry(2.96860986196576) q[10];
cx q[9],q[10];
ry(-1.496258274003264) q[10];
ry(-0.3730797684548617) q[11];
cx q[10],q[11];
ry(-0.31815571026536804) q[10];
ry(2.1856815014611435) q[11];
cx q[10],q[11];
ry(2.242095430410436) q[0];
ry(2.8191488954619603) q[1];
cx q[0],q[1];
ry(-2.8568440106904034) q[0];
ry(2.9154120825126877) q[1];
cx q[0],q[1];
ry(1.7521620607513173) q[1];
ry(2.6068986874455) q[2];
cx q[1],q[2];
ry(0.11011027762301939) q[1];
ry(-3.109137778775597) q[2];
cx q[1],q[2];
ry(-2.324862788885086) q[2];
ry(-1.3883337436481258) q[3];
cx q[2],q[3];
ry(-0.1778869167594808) q[2];
ry(-2.6626447001929736) q[3];
cx q[2],q[3];
ry(-2.231780729697948) q[3];
ry(2.431367063730573) q[4];
cx q[3],q[4];
ry(-2.921374734316328) q[3];
ry(3.110368294973559) q[4];
cx q[3],q[4];
ry(1.3786981378318433) q[4];
ry(1.5502447844372256) q[5];
cx q[4],q[5];
ry(-2.7293668534354336) q[4];
ry(1.5232227424242648) q[5];
cx q[4],q[5];
ry(1.668398225501142) q[5];
ry(2.0806495855536475) q[6];
cx q[5],q[6];
ry(-3.087686279695634) q[5];
ry(-0.6416602720494531) q[6];
cx q[5],q[6];
ry(-0.4328936195847293) q[6];
ry(0.573112446878171) q[7];
cx q[6],q[7];
ry(-3.0995870302939945) q[6];
ry(-3.140904748294309) q[7];
cx q[6],q[7];
ry(-2.9686812379776115) q[7];
ry(0.4777242950348315) q[8];
cx q[7],q[8];
ry(0.7210628789477717) q[7];
ry(2.664234735862649) q[8];
cx q[7],q[8];
ry(-1.197948859169238) q[8];
ry(0.4998115801099114) q[9];
cx q[8],q[9];
ry(0.023188026653745038) q[8];
ry(2.7811806825359473) q[9];
cx q[8],q[9];
ry(-2.429420393074214) q[9];
ry(1.4362298841050556) q[10];
cx q[9],q[10];
ry(-2.170185104369445) q[9];
ry(-0.008974836207556628) q[10];
cx q[9],q[10];
ry(-2.0455108939426045) q[10];
ry(1.1245767343852817) q[11];
cx q[10],q[11];
ry(-0.5599176687695353) q[10];
ry(-2.6740435421569164) q[11];
cx q[10],q[11];
ry(-0.07521045695874669) q[0];
ry(2.9956185419380743) q[1];
cx q[0],q[1];
ry(1.4202606465446566) q[0];
ry(0.31386579731186404) q[1];
cx q[0],q[1];
ry(-1.5944246516390466) q[1];
ry(-1.729017590523652) q[2];
cx q[1],q[2];
ry(1.5097851303567864) q[1];
ry(-0.44132668271795067) q[2];
cx q[1],q[2];
ry(1.8207201561697035) q[2];
ry(0.15807536256589022) q[3];
cx q[2],q[3];
ry(1.660969360461297) q[2];
ry(0.9746679372415806) q[3];
cx q[2],q[3];
ry(2.4546817172880564) q[3];
ry(-2.740708243124133) q[4];
cx q[3],q[4];
ry(-0.004377618118156512) q[3];
ry(0.004516430849432705) q[4];
cx q[3],q[4];
ry(-3.0146761097238426) q[4];
ry(1.5836828825042504) q[5];
cx q[4],q[5];
ry(0.6511670802871848) q[4];
ry(3.140291572058593) q[5];
cx q[4],q[5];
ry(1.555227078989331) q[5];
ry(0.48426555560058393) q[6];
cx q[5],q[6];
ry(-0.0005239277234272421) q[5];
ry(-0.6454473987278532) q[6];
cx q[5],q[6];
ry(0.9036398093408184) q[6];
ry(-0.31755407162893245) q[7];
cx q[6],q[7];
ry(-0.03744735317225434) q[6];
ry(3.137663338009341) q[7];
cx q[6],q[7];
ry(0.01896231930512826) q[7];
ry(0.7197661002757929) q[8];
cx q[7],q[8];
ry(-3.0047291558378655) q[7];
ry(-0.3479620631475191) q[8];
cx q[7],q[8];
ry(-1.759289407453914) q[8];
ry(1.1946569780450815) q[9];
cx q[8],q[9];
ry(2.71862354692559) q[8];
ry(-1.5261361242794191) q[9];
cx q[8],q[9];
ry(2.8193476076539334) q[9];
ry(1.11536347951866) q[10];
cx q[9],q[10];
ry(3.1013367750001986) q[9];
ry(0.01162901068824369) q[10];
cx q[9],q[10];
ry(-2.128611493007672) q[10];
ry(-2.835397861668559) q[11];
cx q[10],q[11];
ry(-0.5756271396978612) q[10];
ry(-0.6297161786797143) q[11];
cx q[10],q[11];
ry(0.13223962273291168) q[0];
ry(1.6487418125302415) q[1];
cx q[0],q[1];
ry(2.2509372754892887) q[0];
ry(-1.3607593836477436) q[1];
cx q[0],q[1];
ry(-1.1765240134115829) q[1];
ry(-1.3948431890493964) q[2];
cx q[1],q[2];
ry(-0.5219463551493322) q[1];
ry(-1.563984203386347) q[2];
cx q[1],q[2];
ry(0.1067656488498882) q[2];
ry(0.9546563476141057) q[3];
cx q[2],q[3];
ry(0.6902973933138029) q[2];
ry(-1.6312936841788077) q[3];
cx q[2],q[3];
ry(0.19689329886641893) q[3];
ry(1.1473561075231613) q[4];
cx q[3],q[4];
ry(-3.1391853694076883) q[3];
ry(-3.1401997812165283) q[4];
cx q[3],q[4];
ry(1.9448263305833091) q[4];
ry(-0.6861976860922461) q[5];
cx q[4],q[5];
ry(2.84827098207615) q[4];
ry(2.212119841330434) q[5];
cx q[4],q[5];
ry(1.272478721215892) q[5];
ry(-0.6137621312863795) q[6];
cx q[5],q[6];
ry(0.016478050546418474) q[5];
ry(2.7899939107548817) q[6];
cx q[5],q[6];
ry(-1.6612632683953465) q[6];
ry(0.3364119323201127) q[7];
cx q[6],q[7];
ry(3.1394945172401396) q[6];
ry(0.0008906687353791298) q[7];
cx q[6],q[7];
ry(-1.149107805885852) q[7];
ry(-1.4984663564953116) q[8];
cx q[7],q[8];
ry(-2.0677974461116344) q[7];
ry(2.0630423561150546) q[8];
cx q[7],q[8];
ry(0.668004134590211) q[8];
ry(-3.006362340024382) q[9];
cx q[8],q[9];
ry(1.496138365080418) q[8];
ry(2.053913200417154) q[9];
cx q[8],q[9];
ry(0.6448059010907943) q[9];
ry(0.9502895496606959) q[10];
cx q[9],q[10];
ry(-1.8612684836762385) q[9];
ry(0.0056181212842568015) q[10];
cx q[9],q[10];
ry(-2.7847664831632897) q[10];
ry(0.22360408389050956) q[11];
cx q[10],q[11];
ry(1.7403680495228435) q[10];
ry(-0.07195745891618532) q[11];
cx q[10],q[11];
ry(3.0618642661819098) q[0];
ry(1.064398274130195) q[1];
cx q[0],q[1];
ry(-1.289205228335253) q[0];
ry(2.4839832407536364) q[1];
cx q[0],q[1];
ry(2.8846501217299623) q[1];
ry(2.469667119621883) q[2];
cx q[1],q[2];
ry(-0.3424069709239695) q[1];
ry(-1.296633944434297) q[2];
cx q[1],q[2];
ry(2.090403873276031) q[2];
ry(0.8210387827487419) q[3];
cx q[2],q[3];
ry(-1.7110838735243286) q[2];
ry(2.346936650779094) q[3];
cx q[2],q[3];
ry(-1.5618951507208543) q[3];
ry(-1.2316289584691456) q[4];
cx q[3],q[4];
ry(-3.1414995127350753) q[3];
ry(-0.02838764994669262) q[4];
cx q[3],q[4];
ry(-0.053302800713673056) q[4];
ry(0.19053732618805166) q[5];
cx q[4],q[5];
ry(-0.07311869631583705) q[4];
ry(0.004153802578667154) q[5];
cx q[4],q[5];
ry(-0.8168106147940106) q[5];
ry(1.4716976268779747) q[6];
cx q[5],q[6];
ry(-2.594199710802973) q[5];
ry(1.8269208658770604) q[6];
cx q[5],q[6];
ry(-0.14320205002737296) q[6];
ry(-3.0112882754927703) q[7];
cx q[6],q[7];
ry(-0.06968705950169518) q[6];
ry(-0.00030090855016307147) q[7];
cx q[6],q[7];
ry(-1.7645432116717379) q[7];
ry(0.5565437298370122) q[8];
cx q[7],q[8];
ry(-0.03030477984855702) q[7];
ry(2.4576925854035903) q[8];
cx q[7],q[8];
ry(2.3369880301483756) q[8];
ry(-1.245281291073047) q[9];
cx q[8],q[9];
ry(-0.016493257441595684) q[8];
ry(2.001696983491116) q[9];
cx q[8],q[9];
ry(1.7329091231352631) q[9];
ry(-0.260557336912333) q[10];
cx q[9],q[10];
ry(-2.8354228717654295) q[9];
ry(-3.130738865630373) q[10];
cx q[9],q[10];
ry(-3.0454463337695703) q[10];
ry(1.8032001606832226) q[11];
cx q[10],q[11];
ry(2.304571381106711) q[10];
ry(1.5201999590545876) q[11];
cx q[10],q[11];
ry(-3.039313081760406) q[0];
ry(-2.967753035567373) q[1];
cx q[0],q[1];
ry(-1.3380834108222543) q[0];
ry(-0.42762625321657133) q[1];
cx q[0],q[1];
ry(1.7926958665321446) q[1];
ry(2.333032787591953) q[2];
cx q[1],q[2];
ry(0.6555309090188937) q[1];
ry(-2.9674136615524174) q[2];
cx q[1],q[2];
ry(-1.5170861962554802) q[2];
ry(1.9081952404067535) q[3];
cx q[2],q[3];
ry(-2.1846925279645335) q[2];
ry(-2.7035605263520948) q[3];
cx q[2],q[3];
ry(1.5853458224033292) q[3];
ry(0.5219304423954488) q[4];
cx q[3],q[4];
ry(0.00012828383349994255) q[3];
ry(-3.1161939734676567) q[4];
cx q[3],q[4];
ry(-1.0862866954968122) q[4];
ry(0.4792637354947749) q[5];
cx q[4],q[5];
ry(-3.1358547782680275) q[4];
ry(-0.003564967510009786) q[5];
cx q[4],q[5];
ry(-2.1493459928801073) q[5];
ry(0.24679604212255887) q[6];
cx q[5],q[6];
ry(-2.4662903974353276) q[5];
ry(1.390455206892124) q[6];
cx q[5],q[6];
ry(1.3277067033053482) q[6];
ry(-1.5152259074826109) q[7];
cx q[6],q[7];
ry(0.29167322291259445) q[6];
ry(3.1297007933484324) q[7];
cx q[6],q[7];
ry(-1.5389993231912529) q[7];
ry(2.868706771362523) q[8];
cx q[7],q[8];
ry(3.1216416504619526) q[7];
ry(-0.47278983025620214) q[8];
cx q[7],q[8];
ry(1.1223888595125746) q[8];
ry(1.076987251778296) q[9];
cx q[8],q[9];
ry(-1.368084853408642) q[8];
ry(0.279574746179156) q[9];
cx q[8],q[9];
ry(3.001950383367288) q[9];
ry(-2.0278571200559377) q[10];
cx q[9],q[10];
ry(1.7427992070250202) q[9];
ry(3.0896197091322373) q[10];
cx q[9],q[10];
ry(2.106114134980813) q[10];
ry(2.241348046648495) q[11];
cx q[10],q[11];
ry(-1.4169733972188094) q[10];
ry(0.446343257638006) q[11];
cx q[10],q[11];
ry(1.1508448144702053) q[0];
ry(-1.0570168278518095) q[1];
cx q[0],q[1];
ry(2.0941124323381968) q[0];
ry(1.91222919384105) q[1];
cx q[0],q[1];
ry(1.7871703615709518) q[1];
ry(0.6320286157083655) q[2];
cx q[1],q[2];
ry(-1.9799255624782424) q[1];
ry(-0.6722689199215646) q[2];
cx q[1],q[2];
ry(-2.34757146119288) q[2];
ry(1.246154111020022) q[3];
cx q[2],q[3];
ry(0.7856517144813973) q[2];
ry(1.445415412797126) q[3];
cx q[2],q[3];
ry(-0.033754827560581874) q[3];
ry(-1.9891447397188242) q[4];
cx q[3],q[4];
ry(1.353496278555396) q[3];
ry(2.7603337496921205) q[4];
cx q[3],q[4];
ry(1.1832621629077806) q[4];
ry(-2.5876795040273013) q[5];
cx q[4],q[5];
ry(2.913446582847683) q[4];
ry(0.002355804392267286) q[5];
cx q[4],q[5];
ry(0.7378566092682233) q[5];
ry(0.41622789326634013) q[6];
cx q[5],q[6];
ry(-3.140429297282102) q[5];
ry(3.137576018870506) q[6];
cx q[5],q[6];
ry(2.3402107638281) q[6];
ry(-1.091531107356048) q[7];
cx q[6],q[7];
ry(-3.02028996930982) q[6];
ry(1.9026720659265672) q[7];
cx q[6],q[7];
ry(0.709187983851995) q[7];
ry(1.3446204860217863) q[8];
cx q[7],q[8];
ry(0.3167535255921717) q[7];
ry(3.141462220662648) q[8];
cx q[7],q[8];
ry(-2.8480853932962096) q[8];
ry(-2.7623936336689865) q[9];
cx q[8],q[9];
ry(-0.5619488507401105) q[8];
ry(1.0961811056139494) q[9];
cx q[8],q[9];
ry(-1.4256765334397512) q[9];
ry(-2.118753876453791) q[10];
cx q[9],q[10];
ry(3.134872403239559) q[9];
ry(-2.4459621036794053) q[10];
cx q[9],q[10];
ry(1.699851029124925) q[10];
ry(1.738870377383284) q[11];
cx q[10],q[11];
ry(1.515809688667601) q[10];
ry(3.1327850134978745) q[11];
cx q[10],q[11];
ry(1.4423601184623758) q[0];
ry(-2.1938807002865355) q[1];
cx q[0],q[1];
ry(-2.003486269171179) q[0];
ry(-1.4352583943143842) q[1];
cx q[0],q[1];
ry(2.8661290339827565) q[1];
ry(0.6390967345741666) q[2];
cx q[1],q[2];
ry(0.48201573768061123) q[1];
ry(-1.579408850285776) q[2];
cx q[1],q[2];
ry(-0.07716691365354221) q[2];
ry(1.2936191570307576) q[3];
cx q[2],q[3];
ry(3.1408827655697102) q[2];
ry(-0.4916051553329455) q[3];
cx q[2],q[3];
ry(-1.5847851656580891) q[3];
ry(2.0133896691154973) q[4];
cx q[3],q[4];
ry(-2.3622486800125615) q[3];
ry(1.8069642443058154) q[4];
cx q[3],q[4];
ry(1.2236973344645445) q[4];
ry(0.708879986366198) q[5];
cx q[4],q[5];
ry(2.591244691913161) q[4];
ry(0.059276391899349294) q[5];
cx q[4],q[5];
ry(-2.571946464605262) q[5];
ry(1.5920611377027036) q[6];
cx q[5],q[6];
ry(3.1398223941917696) q[5];
ry(0.0001877057968725289) q[6];
cx q[5],q[6];
ry(-1.7702403554458335) q[6];
ry(2.9850767838948036) q[7];
cx q[6],q[7];
ry(-0.04892736335303205) q[6];
ry(1.915995587240202) q[7];
cx q[6],q[7];
ry(-1.600950118855097) q[7];
ry(0.4598178673799933) q[8];
cx q[7],q[8];
ry(0.29489957347279194) q[7];
ry(2.896329727038707) q[8];
cx q[7],q[8];
ry(2.2213140099693764) q[8];
ry(1.5631211701683476) q[9];
cx q[8],q[9];
ry(-1.969045271935365) q[8];
ry(3.132530868712891) q[9];
cx q[8],q[9];
ry(-2.1827447922011576) q[9];
ry(1.944930288235292) q[10];
cx q[9],q[10];
ry(2.9064893357347317) q[9];
ry(-0.9294175546337958) q[10];
cx q[9],q[10];
ry(1.827805590924834) q[10];
ry(-3.040975832540237) q[11];
cx q[10],q[11];
ry(-2.8908612889676033) q[10];
ry(0.4484536286700855) q[11];
cx q[10],q[11];
ry(-2.5577298016895753) q[0];
ry(0.3956166820755505) q[1];
ry(-3.0809444256474645) q[2];
ry(-1.4621743791683413) q[3];
ry(0.42711707372074326) q[4];
ry(-2.132131521796633) q[5];
ry(-1.5345722870141731) q[6];
ry(3.1134456143094638) q[7];
ry(1.5481770990628014) q[8];
ry(-0.8731326170045497) q[9];
ry(-3.101327565678465) q[10];
ry(-1.8664562982000834) q[11];