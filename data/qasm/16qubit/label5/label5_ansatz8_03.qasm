OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(2.9823840957783365) q[0];
ry(-2.9555745831844766) q[1];
cx q[0],q[1];
ry(0.11871484383858787) q[0];
ry(2.5674821981681792) q[1];
cx q[0],q[1];
ry(1.0124068042006993) q[2];
ry(-2.8837148967928723) q[3];
cx q[2],q[3];
ry(-2.3805241954284018) q[2];
ry(-0.9036843501667216) q[3];
cx q[2],q[3];
ry(1.2647120446460365) q[4];
ry(-0.5531160520624369) q[5];
cx q[4],q[5];
ry(-2.956266515171598) q[4];
ry(-2.532038797654237) q[5];
cx q[4],q[5];
ry(0.01559844918753761) q[6];
ry(0.4745937159259057) q[7];
cx q[6],q[7];
ry(-3.089948004689011) q[6];
ry(-2.4156799892766374) q[7];
cx q[6],q[7];
ry(0.9579099251252376) q[8];
ry(0.5724926339934381) q[9];
cx q[8],q[9];
ry(-0.6243296980283807) q[8];
ry(-0.8307992345472641) q[9];
cx q[8],q[9];
ry(-2.7676061727490966) q[10];
ry(-1.0812043605340431) q[11];
cx q[10],q[11];
ry(1.9004702763044499) q[10];
ry(-1.107556253022099) q[11];
cx q[10],q[11];
ry(-0.43696472601008124) q[12];
ry(-2.884325320348254) q[13];
cx q[12],q[13];
ry(0.8522234416132598) q[12];
ry(0.40900506839643835) q[13];
cx q[12],q[13];
ry(1.913161499206764) q[14];
ry(1.651453668333137) q[15];
cx q[14],q[15];
ry(-1.2729523186827227) q[14];
ry(-3.013801236400189) q[15];
cx q[14],q[15];
ry(2.164027850784005) q[0];
ry(-0.3501937531881314) q[2];
cx q[0],q[2];
ry(-2.99693310977922) q[0];
ry(-2.9985891786587366) q[2];
cx q[0],q[2];
ry(-2.445950609909289) q[2];
ry(2.908357482386265) q[4];
cx q[2],q[4];
ry(-0.042562644507856184) q[2];
ry(-0.5310246772045151) q[4];
cx q[2],q[4];
ry(-0.896989870858162) q[4];
ry(-2.4206216859536647) q[6];
cx q[4],q[6];
ry(-0.01963022497150657) q[4];
ry(2.9658768365330634) q[6];
cx q[4],q[6];
ry(0.42439605863918395) q[6];
ry(-2.3755083656453455) q[8];
cx q[6],q[8];
ry(2.795963665241318) q[6];
ry(1.5532348695377545) q[8];
cx q[6],q[8];
ry(3.0534708422254373) q[8];
ry(1.890716701379781) q[10];
cx q[8],q[10];
ry(-1.5713349388845295) q[8];
ry(-1.569232798555494) q[10];
cx q[8],q[10];
ry(1.5752306808674055) q[10];
ry(-1.0039791498430706) q[12];
cx q[10],q[12];
ry(-0.000623022749038249) q[10];
ry(-2.7508039795021006) q[12];
cx q[10],q[12];
ry(0.9670248459284627) q[12];
ry(0.6557759009228104) q[14];
cx q[12],q[14];
ry(-1.5693910060571572) q[12];
ry(1.5910828229258103) q[14];
cx q[12],q[14];
ry(1.3661814038631883) q[1];
ry(2.234751119623457) q[3];
cx q[1],q[3];
ry(0.5583949564679215) q[1];
ry(-0.6454739808138443) q[3];
cx q[1],q[3];
ry(0.23100674279597988) q[3];
ry(-2.7788692919076694) q[5];
cx q[3],q[5];
ry(0.30077391699474687) q[3];
ry(-1.5542099039384922) q[5];
cx q[3],q[5];
ry(-1.593624215627984) q[5];
ry(-0.9821689317819615) q[7];
cx q[5],q[7];
ry(-0.3763854535527399) q[5];
ry(-0.5575999077383422) q[7];
cx q[5],q[7];
ry(2.6329055791148366) q[7];
ry(-2.474032572378672) q[9];
cx q[7],q[9];
ry(0.8826131536279783) q[7];
ry(-1.2708041758663946) q[9];
cx q[7],q[9];
ry(0.21228249348239203) q[9];
ry(-2.595634833347802) q[11];
cx q[9],q[11];
ry(3.140368863627886) q[9];
ry(1.5757877866129952) q[11];
cx q[9],q[11];
ry(-1.1884928493467886) q[11];
ry(-2.781833235965409) q[13];
cx q[11],q[13];
ry(-0.5218951859396195) q[11];
ry(3.1406370950919595) q[13];
cx q[11],q[13];
ry(-1.1827677888732955) q[13];
ry(-1.6725511618846571) q[15];
cx q[13],q[15];
ry(-1.554771356041897) q[13];
ry(1.4926082895443757) q[15];
cx q[13],q[15];
ry(3.094313015397996) q[0];
ry(-0.7941814166925161) q[1];
cx q[0],q[1];
ry(2.3562554248737997) q[0];
ry(0.1512433387939005) q[1];
cx q[0],q[1];
ry(-0.5521953471086665) q[2];
ry(1.6477933807741962) q[3];
cx q[2],q[3];
ry(-1.5731349988583627) q[2];
ry(2.959268939005989) q[3];
cx q[2],q[3];
ry(-1.6822882434448534) q[4];
ry(0.11049028972529042) q[5];
cx q[4],q[5];
ry(-1.594001420792924) q[4];
ry(1.303321307577586) q[5];
cx q[4],q[5];
ry(-0.011863913719549157) q[6];
ry(-1.2751918277480907) q[7];
cx q[6],q[7];
ry(3.1245427140130397) q[6];
ry(-3.1405150352096562) q[7];
cx q[6],q[7];
ry(-3.0743185096245798) q[8];
ry(1.5741824000066282) q[9];
cx q[8],q[9];
ry(-1.5687041858213) q[8];
ry(-3.117060841128183) q[9];
cx q[8],q[9];
ry(0.0010332067312841533) q[10];
ry(3.069188939699664) q[11];
cx q[10],q[11];
ry(-1.6258904832782806) q[10];
ry(-1.571491174828549) q[11];
cx q[10],q[11];
ry(-1.0456898958409913) q[12];
ry(2.7019918798009326) q[13];
cx q[12],q[13];
ry(1.44072103291012) q[12];
ry(2.6322756893617103) q[13];
cx q[12],q[13];
ry(0.8438438289535339) q[14];
ry(2.851958580744769) q[15];
cx q[14],q[15];
ry(1.9030068129797788) q[14];
ry(0.041545843751712455) q[15];
cx q[14],q[15];
ry(-0.0920826387989626) q[0];
ry(2.115020109163124) q[2];
cx q[0],q[2];
ry(-3.1412771384128884) q[0];
ry(0.274977057598746) q[2];
cx q[0],q[2];
ry(0.854591655606936) q[2];
ry(2.568318222109175) q[4];
cx q[2],q[4];
ry(-0.01628164481658797) q[2];
ry(2.194834177482544) q[4];
cx q[2],q[4];
ry(-1.946071291628936) q[4];
ry(-2.7194309115488666) q[6];
cx q[4],q[6];
ry(0.0023186628773377007) q[4];
ry(-0.005202656222145662) q[6];
cx q[4],q[6];
ry(-0.35903182608144396) q[6];
ry(-2.3377197033993933) q[8];
cx q[6],q[8];
ry(0.00011329359995837507) q[6];
ry(-1.5686802579069772) q[8];
cx q[6],q[8];
ry(0.8355878412109938) q[8];
ry(0.7583096253734123) q[10];
cx q[8],q[10];
ry(-3.134757990340708) q[8];
ry(-1.5696011976899813) q[10];
cx q[8],q[10];
ry(0.7356156592559824) q[10];
ry(2.9500477983093294) q[12];
cx q[10],q[12];
ry(-3.105556693298101) q[10];
ry(0.242056359081829) q[12];
cx q[10],q[12];
ry(-1.1087492110663406) q[12];
ry(1.5286097424761618) q[14];
cx q[12],q[14];
ry(-0.15815335277155818) q[12];
ry(1.5440998668293515) q[14];
cx q[12],q[14];
ry(-1.4146974432347537) q[1];
ry(-2.1863919821994697) q[3];
cx q[1],q[3];
ry(-3.1369705180553304) q[1];
ry(-0.7334661500108551) q[3];
cx q[1],q[3];
ry(1.7048019014573843) q[3];
ry(-0.6205954950154258) q[5];
cx q[3],q[5];
ry(-1.5555552348793915) q[3];
ry(1.4336139977687725) q[5];
cx q[3],q[5];
ry(-0.9623334600671749) q[5];
ry(-2.59093987154658) q[7];
cx q[5],q[7];
ry(-0.007839318167982334) q[5];
ry(-0.007015302084729668) q[7];
cx q[5],q[7];
ry(-0.09586926979028565) q[7];
ry(0.008153974683640172) q[9];
cx q[7],q[9];
ry(-1.039333784217808) q[7];
ry(-1.568158076487796) q[9];
cx q[7],q[9];
ry(-0.004633985252798567) q[9];
ry(1.567324315098916) q[11];
cx q[9],q[11];
ry(-1.2658286330701185) q[9];
ry(1.5639735298570985) q[11];
cx q[9],q[11];
ry(0.014670563014403726) q[11];
ry(2.7842533598150654) q[13];
cx q[11],q[13];
ry(0.01260094150404889) q[11];
ry(2.5255083123638853) q[13];
cx q[11],q[13];
ry(2.4016927031077775) q[13];
ry(0.05038322750145932) q[15];
cx q[13],q[15];
ry(-3.1323858610595865) q[13];
ry(-0.0042381166326590555) q[15];
cx q[13],q[15];
ry(-0.42205139008653386) q[0];
ry(3.1261284055659235) q[1];
cx q[0],q[1];
ry(1.5669097114122108) q[0];
ry(0.020476316827355498) q[1];
cx q[0],q[1];
ry(-3.1412320954929456) q[2];
ry(3.108524100995927) q[3];
cx q[2],q[3];
ry(-1.5675889326951262) q[2];
ry(1.5987849798693856) q[3];
cx q[2],q[3];
ry(-2.468978091003174) q[4];
ry(0.9722820156522544) q[5];
cx q[4],q[5];
ry(-1.6743383749515557) q[4];
ry(-0.0004392637863378946) q[5];
cx q[4],q[5];
ry(-2.370790042303031) q[6];
ry(2.138581359027766) q[7];
cx q[6],q[7];
ry(-0.0012326575548016194) q[6];
ry(-3.138506296604954) q[7];
cx q[6],q[7];
ry(-1.0619597687382036) q[8];
ry(-2.903452251830755) q[9];
cx q[8],q[9];
ry(1.5696058497060428) q[8];
ry(1.5711514626877865) q[9];
cx q[8],q[9];
ry(1.5701795087198118) q[10];
ry(2.261841797156724) q[11];
cx q[10],q[11];
ry(-0.28426566671681996) q[10];
ry(-1.5721745407979357) q[11];
cx q[10],q[11];
ry(0.47345608479119333) q[12];
ry(-1.3304466007918008) q[13];
cx q[12],q[13];
ry(3.1401539449054345) q[12];
ry(0.1815424023151971) q[13];
cx q[12],q[13];
ry(-1.3164192854674441) q[14];
ry(-1.9049879581484133) q[15];
cx q[14],q[15];
ry(0.04201811011876) q[14];
ry(-1.9478433208246688) q[15];
cx q[14],q[15];
ry(-1.403832254352806) q[0];
ry(3.023440437423111) q[2];
cx q[0],q[2];
ry(1.093614694538796) q[0];
ry(-1.570063972133294) q[2];
cx q[0],q[2];
ry(0.7749886051929318) q[2];
ry(-1.3333168542752987) q[4];
cx q[2],q[4];
ry(-1.5828776095845212) q[2];
ry(0.0017650399123505522) q[4];
cx q[2],q[4];
ry(-1.5666122477074043) q[4];
ry(-0.8129844354972633) q[6];
cx q[4],q[6];
ry(-0.6554357916636891) q[4];
ry(1.5826487640831655) q[6];
cx q[4],q[6];
ry(-1.0145104114417225) q[6];
ry(-1.4580516812279924) q[8];
cx q[6],q[8];
ry(3.140568025526443) q[6];
ry(0.004446822975223032) q[8];
cx q[6],q[8];
ry(1.4246173804251367) q[8];
ry(1.6423481562384108) q[10];
cx q[8],q[10];
ry(-3.1415144540051525) q[8];
ry(0.00012381498720070994) q[10];
cx q[8],q[10];
ry(-1.5400419616459333) q[10];
ry(-2.471587804955103) q[12];
cx q[10],q[12];
ry(0.00046658380580721036) q[10];
ry(3.115859733796562) q[12];
cx q[10],q[12];
ry(0.293304971072307) q[12];
ry(1.9675197776139832) q[14];
cx q[12],q[14];
ry(-2.96932492599561) q[12];
ry(-2.374998350652883) q[14];
cx q[12],q[14];
ry(1.8712814259044583) q[1];
ry(1.757592325936234) q[3];
cx q[1],q[3];
ry(0.027917757222644286) q[1];
ry(3.138310453799821) q[3];
cx q[1],q[3];
ry(1.9628150369530204) q[3];
ry(0.024173247991966743) q[5];
cx q[3],q[5];
ry(2.7202956632568056) q[3];
ry(0.1229655960012801) q[5];
cx q[3],q[5];
ry(-1.4194272138090716) q[5];
ry(0.1380897264305965) q[7];
cx q[5],q[7];
ry(0.0003347532563433874) q[5];
ry(-3.141030020172327) q[7];
cx q[5],q[7];
ry(2.455474429305571) q[7];
ry(-1.5689849596838705) q[9];
cx q[7],q[9];
ry(1.578119552021799) q[7];
ry(-0.005217098526609126) q[9];
cx q[7],q[9];
ry(1.4021221264621138) q[9];
ry(0.8718686220938302) q[11];
cx q[9],q[11];
ry(-0.035892233390947936) q[9];
ry(-0.0004582296647576412) q[11];
cx q[9],q[11];
ry(0.2340508174352215) q[11];
ry(-0.23857050567861737) q[13];
cx q[11],q[13];
ry(-0.0021028904593229925) q[11];
ry(3.1346461657514464) q[13];
cx q[11],q[13];
ry(1.3378204850576605) q[13];
ry(2.612066292276737) q[15];
cx q[13],q[15];
ry(-0.012898734848750155) q[13];
ry(3.140563988992642) q[15];
cx q[13],q[15];
ry(1.5718476626088487) q[0];
ry(0.5915530350632876) q[1];
cx q[0],q[1];
ry(0.3704649075604509) q[0];
ry(2.6409016232733475) q[1];
cx q[0],q[1];
ry(-1.25320970957709) q[2];
ry(-1.8589564303696253) q[3];
cx q[2],q[3];
ry(-0.009746298809806177) q[2];
ry(1.5692505951323712) q[3];
cx q[2],q[3];
ry(-2.92464263310366) q[4];
ry(1.519505547747023) q[5];
cx q[4],q[5];
ry(0.001548389854870713) q[4];
ry(-3.1398470334259065) q[5];
cx q[4],q[5];
ry(-2.532707097409258) q[6];
ry(-1.5873406756585071) q[7];
cx q[6],q[7];
ry(-1.5702542523827496) q[6];
ry(3.130549331287887) q[7];
cx q[6],q[7];
ry(2.7098560291886042) q[8];
ry(-1.407995069435271) q[9];
cx q[8],q[9];
ry(-1.8095204214538039) q[8];
ry(0.004397699019184742) q[9];
cx q[8],q[9];
ry(-1.9682868968688874) q[10];
ry(0.07165380177263003) q[11];
cx q[10],q[11];
ry(1.4600745081187112) q[10];
ry(0.003729154433102301) q[11];
cx q[10],q[11];
ry(2.9671785198440923) q[12];
ry(-3.0422974401863407) q[13];
cx q[12],q[13];
ry(1.5711481263898754) q[12];
ry(1.6709961238426514) q[13];
cx q[12],q[13];
ry(-1.1844311756185857) q[14];
ry(-1.4089277062679928) q[15];
cx q[14],q[15];
ry(0.6790457553787682) q[14];
ry(1.0814691285139162) q[15];
cx q[14],q[15];
ry(-1.8570731859177843) q[0];
ry(0.010836833123065385) q[2];
cx q[0],q[2];
ry(-1.568841507153643) q[0];
ry(3.1317045565844506) q[2];
cx q[0],q[2];
ry(-0.001959527785322268) q[2];
ry(-1.3401529387631301) q[4];
cx q[2],q[4];
ry(1.0009709842264796) q[2];
ry(-1.5701970743899856) q[4];
cx q[2],q[4];
ry(1.7164092208983615) q[4];
ry(0.2614795244866961) q[6];
cx q[4],q[6];
ry(0.00049279169152225) q[4];
ry(-3.137907147783399) q[6];
cx q[4],q[6];
ry(-0.6618650333481454) q[6];
ry(-2.451558747407067) q[8];
cx q[6],q[8];
ry(1.588012182043058) q[6];
ry(-2.9175879653909913) q[8];
cx q[6],q[8];
ry(-1.5765463308176475) q[8];
ry(1.0918879205181427) q[10];
cx q[8],q[10];
ry(0.004546706801003177) q[8];
ry(3.1409147605148378) q[10];
cx q[8],q[10];
ry(-0.7751655601682579) q[10];
ry(2.921297974853724) q[12];
cx q[10],q[12];
ry(0.6009220319540169) q[10];
ry(0.0004038981432330502) q[12];
cx q[10],q[12];
ry(-1.612567649137608) q[12];
ry(3.03109040248434) q[14];
cx q[12],q[14];
ry(-1.5830698595712374) q[12];
ry(3.1335906498297583) q[14];
cx q[12],q[14];
ry(-1.3266850245569763) q[1];
ry(-1.8347690596513448) q[3];
cx q[1],q[3];
ry(-1.5719810248829837) q[1];
ry(-0.020518168616890264) q[3];
cx q[1],q[3];
ry(-0.0844031474910303) q[3];
ry(0.1754089420360172) q[5];
cx q[3],q[5];
ry(-1.5451801737415276) q[3];
ry(3.1396067793145286) q[5];
cx q[3],q[5];
ry(-2.8490669933330044) q[5];
ry(-1.8909951641486693) q[7];
cx q[5],q[7];
ry(-1.5448034521773462) q[5];
ry(-2.646085214238991) q[7];
cx q[5],q[7];
ry(2.115002305673599) q[7];
ry(-1.2018657532139532) q[9];
cx q[7],q[9];
ry(-0.00020204772132435418) q[7];
ry(3.139884331329988) q[9];
cx q[7],q[9];
ry(-2.107338958411593) q[9];
ry(-0.1657038198197697) q[11];
cx q[9],q[11];
ry(-3.133200569663779) q[9];
ry(-3.1127673788244903) q[11];
cx q[9],q[11];
ry(-1.015131522534836) q[11];
ry(2.8030492516757923) q[13];
cx q[11],q[13];
ry(0.00798270970520418) q[11];
ry(0.002760886867277712) q[13];
cx q[11],q[13];
ry(-2.8153129299898376) q[13];
ry(0.5401839825347228) q[15];
cx q[13],q[15];
ry(-1.5725734300334435) q[13];
ry(1.5662426651162482) q[15];
cx q[13],q[15];
ry(-0.013638509333712355) q[0];
ry(1.004097592745367) q[1];
cx q[0],q[1];
ry(-2.959589182481812) q[0];
ry(3.1286001738882265) q[1];
cx q[0],q[1];
ry(3.136748502517984) q[2];
ry(-0.9112774922962243) q[3];
cx q[2],q[3];
ry(-0.0021695933860731644) q[2];
ry(1.568939528417351) q[3];
cx q[2],q[3];
ry(-3.092744960380209) q[4];
ry(-1.4977696592887337) q[5];
cx q[4],q[5];
ry(-0.005428178914483439) q[4];
ry(3.1344297918629604) q[5];
cx q[4],q[5];
ry(1.393132627599715) q[6];
ry(-3.055238260908564) q[7];
cx q[6],q[7];
ry(3.141399951714965) q[6];
ry(-3.1388670902429983) q[7];
cx q[6],q[7];
ry(-0.008502775417305486) q[8];
ry(2.981495771947056) q[9];
cx q[8],q[9];
ry(-1.506678176637335) q[8];
ry(1.581952750930018) q[9];
cx q[8],q[9];
ry(0.061140789725000914) q[10];
ry(-1.1688414574293744) q[11];
cx q[10],q[11];
ry(2.9903721524806572) q[10];
ry(1.5531712707734817) q[11];
cx q[10],q[11];
ry(1.429216247156856) q[12];
ry(-3.1217941350398055) q[13];
cx q[12],q[13];
ry(-1.6157139421750764) q[12];
ry(-2.447932687083217) q[13];
cx q[12],q[13];
ry(-1.2546985889727686) q[14];
ry(-0.10129820183398021) q[15];
cx q[14],q[15];
ry(-1.4942568594434293) q[14];
ry(0.001550835971663083) q[15];
cx q[14],q[15];
ry(-1.6494598114273742) q[0];
ry(-3.1407826656231124) q[2];
cx q[0],q[2];
ry(-1.5671919572120165) q[0];
ry(3.14147398273073) q[2];
cx q[0],q[2];
ry(3.1393120916839243) q[2];
ry(0.8289581411683642) q[4];
cx q[2],q[4];
ry(0.0009198431467074973) q[2];
ry(-1.001255234402585) q[4];
cx q[2],q[4];
ry(1.0653720729085474) q[4];
ry(1.3813258608617531) q[6];
cx q[4],q[6];
ry(1.5774354820071326) q[4];
ry(-3.140400381403747) q[6];
cx q[4],q[6];
ry(0.9179882187479225) q[6];
ry(0.31455580737241295) q[8];
cx q[6],q[8];
ry(-3.140714358596607) q[6];
ry(-3.1415752340343057) q[8];
cx q[6],q[8];
ry(1.3236769683379128) q[8];
ry(0.7911575035552341) q[10];
cx q[8],q[10];
ry(-3.1376939090856286) q[8];
ry(3.137798724315142) q[10];
cx q[8],q[10];
ry(1.8191221113125042) q[10];
ry(1.5477713181441421) q[12];
cx q[10],q[12];
ry(0.027217923852954407) q[10];
ry(0.0011813568331380253) q[12];
cx q[10],q[12];
ry(1.516875919803031) q[12];
ry(2.3540019788583777) q[14];
cx q[12],q[14];
ry(-3.089358321354391) q[12];
ry(-1.5696357945108224) q[14];
cx q[12],q[14];
ry(-1.8548781225832753) q[1];
ry(-0.5719859058197967) q[3];
cx q[1],q[3];
ry(0.021263645622418) q[1];
ry(0.018036834571237392) q[3];
cx q[1],q[3];
ry(0.013185441413324561) q[3];
ry(2.7983002508900605) q[5];
cx q[3],q[5];
ry(3.141589590751581) q[3];
ry(3.139118023567325) q[5];
cx q[3],q[5];
ry(2.6115805801664784) q[5];
ry(2.6070286248937697) q[7];
cx q[5],q[7];
ry(-2.821270439097972) q[5];
ry(1.7091384564907077) q[7];
cx q[5],q[7];
ry(-0.20912438921450335) q[7];
ry(3.101939864849683) q[9];
cx q[7],q[9];
ry(-1.5708589591160176) q[7];
ry(-0.0546205934012419) q[9];
cx q[7],q[9];
ry(-2.8163381032845862) q[9];
ry(-1.7603373020509734) q[11];
cx q[9],q[11];
ry(-3.1415713141625625) q[9];
ry(-3.140884851983648) q[11];
cx q[9],q[11];
ry(0.0012334574899925954) q[11];
ry(0.011212856735951961) q[13];
cx q[11],q[13];
ry(-0.46315233478948414) q[11];
ry(-0.048517687085362304) q[13];
cx q[11],q[13];
ry(0.06460826144798215) q[13];
ry(1.5638047577175018) q[15];
cx q[13],q[15];
ry(-0.0013295544164776) q[13];
ry(0.0034254964981968285) q[15];
cx q[13],q[15];
ry(-0.35392714510390805) q[0];
ry(-1.6502350885274675) q[1];
cx q[0],q[1];
ry(-1.767740216407434) q[0];
ry(1.570621376015299) q[1];
cx q[0],q[1];
ry(-0.0007623099637311949) q[2];
ry(-1.3942280600233865) q[3];
cx q[2],q[3];
ry(1.5700201769154964) q[2];
ry(-1.5732679973370765) q[3];
cx q[2],q[3];
ry(2.0014075260815742) q[4];
ry(-2.6633333583550747) q[5];
cx q[4],q[5];
ry(-1.572959088372032) q[4];
ry(3.1410500789340077) q[5];
cx q[4],q[5];
ry(-0.6704259234610617) q[6];
ry(0.12659686078125235) q[7];
cx q[6],q[7];
ry(2.992101688321744) q[6];
ry(3.1379297468179095) q[7];
cx q[6],q[7];
ry(-0.5082163110908242) q[8];
ry(-2.836203701840382) q[9];
cx q[8],q[9];
ry(0.062370223671263325) q[8];
ry(-3.141576715208611) q[9];
cx q[8],q[9];
ry(0.5481847344270924) q[10];
ry(1.5893036625855488) q[11];
cx q[10],q[11];
ry(-1.5709721342832026) q[10];
ry(1.5498661970583685) q[11];
cx q[10],q[11];
ry(0.0003869648792500513) q[12];
ry(0.6762143338020979) q[13];
cx q[12],q[13];
ry(3.1339830435743536) q[12];
ry(-1.564627460331467) q[13];
cx q[12],q[13];
ry(-0.4549684575984365) q[14];
ry(-3.125384600731901) q[15];
cx q[14],q[15];
ry(1.473615245562768) q[14];
ry(-1.5671915916543169) q[15];
cx q[14],q[15];
ry(1.8275083793070728) q[0];
ry(0.016282648375045557) q[2];
cx q[0],q[2];
ry(-1.4751843407140464) q[0];
ry(0.05954556083235499) q[2];
cx q[0],q[2];
ry(0.25345879460000736) q[2];
ry(-1.5150050522975924) q[4];
cx q[2],q[4];
ry(0.012451441315305296) q[2];
ry(-3.1411127819411067) q[4];
cx q[2],q[4];
ry(-0.12434646954648265) q[4];
ry(3.123876375292028) q[6];
cx q[4],q[6];
ry(-1.5631294981185349) q[4];
ry(7.204787315107096e-05) q[6];
cx q[4],q[6];
ry(-3.133769832903965) q[6];
ry(1.6239697361759697) q[8];
cx q[6],q[8];
ry(-1.5707798221224922) q[6];
ry(-1.5680722675547818) q[8];
cx q[6],q[8];
ry(1.375168807821347) q[8];
ry(-2.85764121805876) q[10];
cx q[8],q[10];
ry(-0.005602883994731266) q[8];
ry(-0.01375342919558431) q[10];
cx q[8],q[10];
ry(-1.8785613064820428) q[10];
ry(-1.6671127236990504) q[12];
cx q[10],q[12];
ry(3.1410298389740117) q[10];
ry(3.139926890367547) q[12];
cx q[10],q[12];
ry(-1.5775985203955998) q[12];
ry(-1.5412266641739143) q[14];
cx q[12],q[14];
ry(0.10744296031142397) q[12];
ry(-0.05837890629335471) q[14];
cx q[12],q[14];
ry(2.6727516371396693) q[1];
ry(-1.5655587913550029) q[3];
cx q[1],q[3];
ry(3.0869947663445196) q[1];
ry(-0.0026297856630765892) q[3];
cx q[1],q[3];
ry(-1.6895480100780365) q[3];
ry(-1.4049645814874039) q[5];
cx q[3],q[5];
ry(3.136605905636961) q[3];
ry(-3.1415675915272123) q[5];
cx q[3],q[5];
ry(-1.4060319148548635) q[5];
ry(3.140295260737759) q[7];
cx q[5],q[7];
ry(-1.570774785670773) q[5];
ry(-3.058190284822444) q[7];
cx q[5],q[7];
ry(-1.5706690796963279) q[7];
ry(-3.113787849086014) q[9];
cx q[7],q[9];
ry(-3.101070349545181) q[7];
ry(1.5775542666916111) q[9];
cx q[7],q[9];
ry(-0.305580109959382) q[9];
ry(-1.5811510750895759) q[11];
cx q[9],q[11];
ry(-1.5706959849086504) q[9];
ry(-0.006899164564397823) q[11];
cx q[9],q[11];
ry(1.0372538164244913) q[11];
ry(2.302076022001037) q[13];
cx q[11],q[13];
ry(1.5708346691014454) q[11];
ry(3.141465577996984) q[13];
cx q[11],q[13];
ry(1.5706187603453292) q[13];
ry(3.1051210243615057) q[15];
cx q[13],q[15];
ry(-1.5707573694619719) q[13];
ry(1.5181817948122678) q[15];
cx q[13],q[15];
ry(1.316803331479682) q[0];
ry(-2.739436774415067) q[1];
ry(-2.8920291579851387) q[2];
ry(-1.6962147043115192) q[3];
ry(1.6488126731738804) q[4];
ry(5.78431906241017e-06) q[5];
ry(-1.5708789148178797) q[6];
ry(-3.1414961678379614) q[7];
ry(-1.3753880575979642) q[8];
ry(-1.2572500988303819) q[9];
ry(-1.5785353808921139) q[10];
ry(-2.104476317178162) q[11];
ry(1.4791629733529834) q[12];
ry(1.5707868019769284) q[13];
ry(-0.026360586958816355) q[14];
ry(1.5704764771134079) q[15];