OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.2478694275596514) q[0];
ry(1.7800116149964174) q[1];
cx q[0],q[1];
ry(1.9445287395110888) q[0];
ry(-0.33793163148797256) q[1];
cx q[0],q[1];
ry(-2.0428938025380727) q[1];
ry(3.0175747891085147) q[2];
cx q[1],q[2];
ry(1.1266077245133186) q[1];
ry(1.350340537858212) q[2];
cx q[1],q[2];
ry(-1.815374847691736) q[2];
ry(-1.4681940458484362) q[3];
cx q[2],q[3];
ry(1.8338413602917656) q[2];
ry(0.3602991890571852) q[3];
cx q[2],q[3];
ry(1.5226272109439165) q[3];
ry(-2.742005981564742) q[4];
cx q[3],q[4];
ry(1.6897996839077714) q[3];
ry(-1.7079623191918305) q[4];
cx q[3],q[4];
ry(-0.6503969105913416) q[4];
ry(2.763611687123571) q[5];
cx q[4],q[5];
ry(3.084359180520046) q[4];
ry(1.549152551622563) q[5];
cx q[4],q[5];
ry(-0.27385052294201273) q[5];
ry(1.2863247536174223) q[6];
cx q[5],q[6];
ry(0.00981112699350195) q[5];
ry(0.007363801993020555) q[6];
cx q[5],q[6];
ry(1.8315622618621406) q[6];
ry(-0.7853752514692622) q[7];
cx q[6],q[7];
ry(-2.652798642272609) q[6];
ry(-2.1780986730804552) q[7];
cx q[6],q[7];
ry(-2.0319266175182396) q[7];
ry(-0.8730317976713103) q[8];
cx q[7],q[8];
ry(3.118221981679495) q[7];
ry(-3.105478756217329) q[8];
cx q[7],q[8];
ry(0.5752085772618484) q[8];
ry(2.9703637381180563) q[9];
cx q[8],q[9];
ry(-2.562229446835528) q[8];
ry(-0.5741919817548649) q[9];
cx q[8],q[9];
ry(0.4692505387809325) q[9];
ry(-1.559034724942765) q[10];
cx q[9],q[10];
ry(1.5025722843027198) q[9];
ry(3.1250587399598397) q[10];
cx q[9],q[10];
ry(-1.3363418092255204) q[10];
ry(-0.3848880442705163) q[11];
cx q[10],q[11];
ry(-1.8528078398869403) q[10];
ry(-1.2409333840844705) q[11];
cx q[10],q[11];
ry(-0.502751862020682) q[11];
ry(-1.3301720182369765) q[12];
cx q[11],q[12];
ry(0.2544430247905496) q[11];
ry(-0.06604897766914775) q[12];
cx q[11],q[12];
ry(1.917602257589211) q[12];
ry(-0.7414191315189865) q[13];
cx q[12],q[13];
ry(-1.3867163652550971) q[12];
ry(-2.914572049776743) q[13];
cx q[12],q[13];
ry(-2.642549806317426) q[13];
ry(1.6807837466600475) q[14];
cx q[13],q[14];
ry(3.134005767245517) q[13];
ry(-0.0004494007617109872) q[14];
cx q[13],q[14];
ry(1.7619038641900069) q[14];
ry(-0.46360336708599004) q[15];
cx q[14],q[15];
ry(-2.9806396302059777) q[14];
ry(0.5908773117725481) q[15];
cx q[14],q[15];
ry(0.8822887220706946) q[15];
ry(1.6502387054502181) q[16];
cx q[15],q[16];
ry(0.6376437610965349) q[15];
ry(2.9609164671538997) q[16];
cx q[15],q[16];
ry(-0.9171392325264615) q[16];
ry(1.5645382748706629) q[17];
cx q[16],q[17];
ry(-0.053122937634880074) q[16];
ry(-3.138338574902729) q[17];
cx q[16],q[17];
ry(-0.9898417265003463) q[17];
ry(1.5042425116618434) q[18];
cx q[17],q[18];
ry(2.0739940371880863) q[17];
ry(3.001043740889373) q[18];
cx q[17],q[18];
ry(-2.5331215966557985) q[18];
ry(-0.148493490931914) q[19];
cx q[18],q[19];
ry(2.775781103833026) q[18];
ry(2.498904215272144) q[19];
cx q[18],q[19];
ry(-1.3497915814189536) q[0];
ry(-0.9510073718166483) q[1];
cx q[0],q[1];
ry(2.084242281316098) q[0];
ry(0.2724298711125754) q[1];
cx q[0],q[1];
ry(1.5754244910111446) q[1];
ry(-2.1207448490892875) q[2];
cx q[1],q[2];
ry(-0.2985566657837504) q[1];
ry(-3.0187562132902515) q[2];
cx q[1],q[2];
ry(1.4059952794429238) q[2];
ry(1.8445989467011288) q[3];
cx q[2],q[3];
ry(3.0966740088243947) q[2];
ry(-2.094560206330236) q[3];
cx q[2],q[3];
ry(0.9921148818378391) q[3];
ry(1.1123369419348847) q[4];
cx q[3],q[4];
ry(0.48048608791145053) q[3];
ry(-4.8437834962075694e-05) q[4];
cx q[3],q[4];
ry(-1.5348317547072057) q[4];
ry(-1.6507208075704325) q[5];
cx q[4],q[5];
ry(-1.7774203575222751) q[4];
ry(-0.5139377858300979) q[5];
cx q[4],q[5];
ry(1.884410598709307) q[5];
ry(1.2538963813046582) q[6];
cx q[5],q[6];
ry(-3.1025536786331838) q[5];
ry(0.0005215799462238974) q[6];
cx q[5],q[6];
ry(-1.0605252849035738) q[6];
ry(-3.1010134066585264) q[7];
cx q[6],q[7];
ry(-2.3891516325906506) q[6];
ry(0.8238954255371864) q[7];
cx q[6],q[7];
ry(-0.4303801162800526) q[7];
ry(1.929372978095996) q[8];
cx q[7],q[8];
ry(0.08703993929517928) q[7];
ry(0.0863227445205732) q[8];
cx q[7],q[8];
ry(2.827116885833063) q[8];
ry(-1.5884724461534494) q[9];
cx q[8],q[9];
ry(2.737798023465926) q[8];
ry(0.8002368217548836) q[9];
cx q[8],q[9];
ry(2.807327504006529) q[9];
ry(-1.2061286399165965) q[10];
cx q[9],q[10];
ry(-3.122315496270033) q[9];
ry(0.014949307455006918) q[10];
cx q[9],q[10];
ry(-0.22653999856227636) q[10];
ry(-2.2733149198150753) q[11];
cx q[10],q[11];
ry(-2.524361672628198) q[10];
ry(-0.1088058898189631) q[11];
cx q[10],q[11];
ry(-0.16311966757961027) q[11];
ry(1.3341528124282687) q[12];
cx q[11],q[12];
ry(0.0007018997760299683) q[11];
ry(3.138793229077007) q[12];
cx q[11],q[12];
ry(1.7018496820076239) q[12];
ry(-0.6880322491885744) q[13];
cx q[12],q[13];
ry(1.542630693759441) q[12];
ry(-0.22075193323399445) q[13];
cx q[12],q[13];
ry(0.7451893698735423) q[13];
ry(0.17303144253373112) q[14];
cx q[13],q[14];
ry(-0.003752159409229302) q[13];
ry(-1.658836921854565) q[14];
cx q[13],q[14];
ry(-3.021152095043598) q[14];
ry(-2.351288796635168) q[15];
cx q[14],q[15];
ry(-0.17835544594169495) q[14];
ry(-0.3062963316435141) q[15];
cx q[14],q[15];
ry(-2.972528018160573) q[15];
ry(2.394894835269159) q[16];
cx q[15],q[16];
ry(-3.0564665618486697) q[15];
ry(-1.4565054656415288) q[16];
cx q[15],q[16];
ry(-0.48475070266846476) q[16];
ry(1.6083037685185682) q[17];
cx q[16],q[17];
ry(-0.002705484673644065) q[16];
ry(0.0015664306894733653) q[17];
cx q[16],q[17];
ry(-1.9007997302956134) q[17];
ry(-2.445604710188518) q[18];
cx q[17],q[18];
ry(-0.6268791275525069) q[17];
ry(1.829855981768515) q[18];
cx q[17],q[18];
ry(0.598294109368304) q[18];
ry(-2.241863504379282) q[19];
cx q[18],q[19];
ry(1.9745441272596318) q[18];
ry(3.0240378477157295) q[19];
cx q[18],q[19];
ry(2.9165891376897948) q[0];
ry(-1.9920278164452432) q[1];
cx q[0],q[1];
ry(1.841699187839156) q[0];
ry(-2.254631214627335) q[1];
cx q[0],q[1];
ry(1.5224352300876187) q[1];
ry(0.3149440812522793) q[2];
cx q[1],q[2];
ry(1.4443926761243773) q[1];
ry(-1.562617713437744) q[2];
cx q[1],q[2];
ry(-1.3836971991143319) q[2];
ry(-2.8472307800469174) q[3];
cx q[2],q[3];
ry(3.14124708455355) q[2];
ry(2.763258257154181) q[3];
cx q[2],q[3];
ry(2.6033779796203533) q[3];
ry(-1.2401864868397965) q[4];
cx q[3],q[4];
ry(2.3315953822645143) q[3];
ry(-0.005003688122242167) q[4];
cx q[3],q[4];
ry(-2.318193308829571) q[4];
ry(-1.8868556549008844) q[5];
cx q[4],q[5];
ry(-1.089172213407094) q[4];
ry(2.3094543285186235) q[5];
cx q[4],q[5];
ry(-0.23443734640587768) q[5];
ry(2.3111765734669514) q[6];
cx q[5],q[6];
ry(-0.26485139134919233) q[5];
ry(-0.00854559191902651) q[6];
cx q[5],q[6];
ry(3.1388354595913386) q[6];
ry(1.2120883743547495) q[7];
cx q[6],q[7];
ry(2.1353447593784884) q[6];
ry(1.2056637850336254) q[7];
cx q[6],q[7];
ry(3.0265893738952903) q[7];
ry(-1.9565369447356504) q[8];
cx q[7],q[8];
ry(2.378800335041546) q[7];
ry(0.011035182320983382) q[8];
cx q[7],q[8];
ry(0.6477606998177553) q[8];
ry(-1.085069792681945) q[9];
cx q[8],q[9];
ry(2.3449640996419063) q[8];
ry(-0.8677623348325213) q[9];
cx q[8],q[9];
ry(-0.8077541813234953) q[9];
ry(1.5790988493746594) q[10];
cx q[9],q[10];
ry(2.4478156055076528) q[9];
ry(-0.0008491632487297451) q[10];
cx q[9],q[10];
ry(-0.28226026025575646) q[10];
ry(0.03779144548394308) q[11];
cx q[10],q[11];
ry(1.478859287915391) q[10];
ry(1.5497941241169073) q[11];
cx q[10],q[11];
ry(1.711100656570519) q[11];
ry(1.0737420544670067) q[12];
cx q[11],q[12];
ry(-2.974122939658292) q[11];
ry(0.10774803108867292) q[12];
cx q[11],q[12];
ry(2.341764279147598) q[12];
ry(2.7099420949349495) q[13];
cx q[12],q[13];
ry(-2.0625256903759412) q[12];
ry(-1.084542202693617) q[13];
cx q[12],q[13];
ry(-0.7131740221426871) q[13];
ry(-1.9402076755903233) q[14];
cx q[13],q[14];
ry(2.5995577269945036) q[13];
ry(3.1198041663623517) q[14];
cx q[13],q[14];
ry(2.7483075966646684) q[14];
ry(1.9637317191690364) q[15];
cx q[14],q[15];
ry(3.095363622015098) q[14];
ry(-3.1066895794303404) q[15];
cx q[14],q[15];
ry(2.5895503439922303) q[15];
ry(1.4654186953799373) q[16];
cx q[15],q[16];
ry(1.9160631062880213) q[15];
ry(-1.457082330948263) q[16];
cx q[15],q[16];
ry(0.7122566424142113) q[16];
ry(-0.593820602357681) q[17];
cx q[16],q[17];
ry(-3.140305664980612) q[16];
ry(-0.0032081947245993386) q[17];
cx q[16],q[17];
ry(-1.4966705739930015) q[17];
ry(-0.4877956155556271) q[18];
cx q[17],q[18];
ry(2.505814279459859) q[17];
ry(1.7586870271208923) q[18];
cx q[17],q[18];
ry(2.51769769083519) q[18];
ry(-0.27190021214726995) q[19];
cx q[18],q[19];
ry(-1.8252505758403732) q[18];
ry(0.8279009193426301) q[19];
cx q[18],q[19];
ry(-0.6301382321594702) q[0];
ry(-1.211506689086204) q[1];
cx q[0],q[1];
ry(1.1952539768140111) q[0];
ry(-2.455051146289132) q[1];
cx q[0],q[1];
ry(1.3885946594805545) q[1];
ry(-1.8435824619758239) q[2];
cx q[1],q[2];
ry(0.2077774807551614) q[1];
ry(-2.4769893675015573) q[2];
cx q[1],q[2];
ry(-0.5682378625936142) q[2];
ry(2.824364071886808) q[3];
cx q[2],q[3];
ry(3.1411989657806387) q[2];
ry(-0.13862896011486203) q[3];
cx q[2],q[3];
ry(3.121620132081543) q[3];
ry(-0.04456991864375137) q[4];
cx q[3],q[4];
ry(-0.2664741982706918) q[3];
ry(-3.139647307899804) q[4];
cx q[3],q[4];
ry(0.804813204845894) q[4];
ry(-2.961989367049799) q[5];
cx q[4],q[5];
ry(-3.1205585384158327) q[4];
ry(0.2488505941450816) q[5];
cx q[4],q[5];
ry(-1.6424056860067284) q[5];
ry(-0.7636159079692592) q[6];
cx q[5],q[6];
ry(-1.4245871032336856) q[5];
ry(-0.04099911175493176) q[6];
cx q[5],q[6];
ry(2.6360727525302803) q[6];
ry(-2.496256199320665) q[7];
cx q[6],q[7];
ry(1.4687975157456514) q[6];
ry(0.7808746021564481) q[7];
cx q[6],q[7];
ry(2.8057955159242436) q[7];
ry(3.1167661097271786) q[8];
cx q[7],q[8];
ry(-0.5483161485184341) q[7];
ry(-0.002478859299130671) q[8];
cx q[7],q[8];
ry(2.7241380711638037) q[8];
ry(2.741705602516928) q[9];
cx q[8],q[9];
ry(3.14130968060873) q[8];
ry(2.9999308597976224) q[9];
cx q[8],q[9];
ry(-3.0350788390808185) q[9];
ry(1.895003466215191) q[10];
cx q[9],q[10];
ry(-1.641735347019208) q[9];
ry(0.0536701928083998) q[10];
cx q[9],q[10];
ry(-1.6357919069662736) q[10];
ry(3.0160389926812146) q[11];
cx q[10],q[11];
ry(1.9309468650336257) q[10];
ry(3.0742339122445363) q[11];
cx q[10],q[11];
ry(0.9470990309493174) q[11];
ry(0.011920921200327506) q[12];
cx q[11],q[12];
ry(0.3608188141950892) q[11];
ry(1.0366845697797178) q[12];
cx q[11],q[12];
ry(2.2166973053721453) q[12];
ry(0.8070425521637636) q[13];
cx q[12],q[13];
ry(3.141168216798665) q[12];
ry(0.014415236703153589) q[13];
cx q[12],q[13];
ry(1.5959943691028062) q[13];
ry(1.0065152855523394) q[14];
cx q[13],q[14];
ry(2.210201984311353) q[13];
ry(0.0012855373273001902) q[14];
cx q[13],q[14];
ry(3.110151986947883) q[14];
ry(1.6457813725582877) q[15];
cx q[14],q[15];
ry(-0.9075887830701665) q[14];
ry(-3.0758823838619107) q[15];
cx q[14],q[15];
ry(-2.278302456582659) q[15];
ry(1.372144560153406) q[16];
cx q[15],q[16];
ry(0.21161172503934197) q[15];
ry(-1.664839547411403) q[16];
cx q[15],q[16];
ry(2.2328124088462644) q[16];
ry(1.4946011439164728) q[17];
cx q[16],q[17];
ry(-3.1402197766177173) q[16];
ry(-0.00016538757550854655) q[17];
cx q[16],q[17];
ry(0.28557299195696695) q[17];
ry(-1.1599688060867919) q[18];
cx q[17],q[18];
ry(-2.2163717377559475) q[17];
ry(-0.02187342752061484) q[18];
cx q[17],q[18];
ry(0.7304651639468158) q[18];
ry(-1.2399563734076293) q[19];
cx q[18],q[19];
ry(2.2740806793663566) q[18];
ry(1.953675066608656) q[19];
cx q[18],q[19];
ry(2.5725443712962477) q[0];
ry(0.875365506033019) q[1];
cx q[0],q[1];
ry(0.15419685593932814) q[0];
ry(1.5181791079024345) q[1];
cx q[0],q[1];
ry(3.014784660949169) q[1];
ry(-1.0193757093992488) q[2];
cx q[1],q[2];
ry(-1.4675888228684633) q[1];
ry(-0.9114003583071266) q[2];
cx q[1],q[2];
ry(3.1351990543455814) q[2];
ry(1.3251681704220628) q[3];
cx q[2],q[3];
ry(-0.34668838726609025) q[2];
ry(0.13228320823758288) q[3];
cx q[2],q[3];
ry(2.230973361198143) q[3];
ry(0.054324808903686295) q[4];
cx q[3],q[4];
ry(-0.0025282670479148794) q[3];
ry(3.910967281051114e-05) q[4];
cx q[3],q[4];
ry(-2.4362330489165207) q[4];
ry(-1.1883407519298244) q[5];
cx q[4],q[5];
ry(2.9424088385781175) q[4];
ry(1.669394170573785) q[5];
cx q[4],q[5];
ry(-1.402996765036326) q[5];
ry(1.263849936222658) q[6];
cx q[5],q[6];
ry(-2.103638303600084) q[5];
ry(-0.03210164306139799) q[6];
cx q[5],q[6];
ry(0.7135468582777609) q[6];
ry(1.9534885770158947) q[7];
cx q[6],q[7];
ry(-0.0038942665951466893) q[6];
ry(-0.2691041077446873) q[7];
cx q[6],q[7];
ry(0.9284063820076075) q[7];
ry(-2.774797785552914) q[8];
cx q[7],q[8];
ry(-0.6467683689425048) q[7];
ry(0.0014805985964361432) q[8];
cx q[7],q[8];
ry(-2.1345530688125463) q[8];
ry(0.18615639281417984) q[9];
cx q[8],q[9];
ry(-3.1342359878165444) q[8];
ry(1.089879972087567) q[9];
cx q[8],q[9];
ry(1.78664361068492) q[9];
ry(1.1435392215911415) q[10];
cx q[9],q[10];
ry(3.0660050002399246) q[9];
ry(-3.1122359525822523) q[10];
cx q[9],q[10];
ry(-0.720820730592175) q[10];
ry(-1.647748749477161) q[11];
cx q[10],q[11];
ry(-2.899561199639037) q[10];
ry(3.1408815250526763) q[11];
cx q[10],q[11];
ry(-1.8958870757747528) q[11];
ry(2.836462020611369) q[12];
cx q[11],q[12];
ry(2.2559494269602443) q[11];
ry(-0.03607382024338257) q[12];
cx q[11],q[12];
ry(-2.422402376747422) q[12];
ry(0.8817034521577924) q[13];
cx q[12],q[13];
ry(-3.1348767058799334) q[12];
ry(-0.4962343865673073) q[13];
cx q[12],q[13];
ry(-2.7850326511459236) q[13];
ry(2.7216107192958336) q[14];
cx q[13],q[14];
ry(-2.1675006356755846) q[13];
ry(-0.01195669487852453) q[14];
cx q[13],q[14];
ry(0.20955212928877387) q[14];
ry(-1.429255183903507) q[15];
cx q[14],q[15];
ry(2.984281791960018) q[14];
ry(3.1299029836491763) q[15];
cx q[14],q[15];
ry(2.2277759993397543) q[15];
ry(-1.1699367418686561) q[16];
cx q[15],q[16];
ry(0.30082592137800945) q[15];
ry(0.17417803297305046) q[16];
cx q[15],q[16];
ry(2.1080861335748784) q[16];
ry(2.021200000684847) q[17];
cx q[16],q[17];
ry(3.1195664483786802) q[16];
ry(3.1410282594688437) q[17];
cx q[16],q[17];
ry(1.5531106423687633) q[17];
ry(2.625707803993087) q[18];
cx q[17],q[18];
ry(-0.10881533371606261) q[17];
ry(-2.0798574288557763) q[18];
cx q[17],q[18];
ry(1.7341297384399392) q[18];
ry(-2.5339790554279027) q[19];
cx q[18],q[19];
ry(2.785764239532346) q[18];
ry(2.995059079514301) q[19];
cx q[18],q[19];
ry(-1.3350602886243912) q[0];
ry(-2.3184648865958795) q[1];
cx q[0],q[1];
ry(-2.913950276829734) q[0];
ry(2.886294314967878) q[1];
cx q[0],q[1];
ry(-0.9064640098660942) q[1];
ry(-2.8112034988040144) q[2];
cx q[1],q[2];
ry(-2.51090431778946) q[1];
ry(-3.088648669139879) q[2];
cx q[1],q[2];
ry(-2.9444825847590224) q[2];
ry(-2.1943038187787174) q[3];
cx q[2],q[3];
ry(2.4113984363742667) q[2];
ry(-2.965166139650026) q[3];
cx q[2],q[3];
ry(0.3431309121010635) q[3];
ry(-2.235149285775914) q[4];
cx q[3],q[4];
ry(-3.0493464820897147) q[3];
ry(3.141311748405798) q[4];
cx q[3],q[4];
ry(-1.7115381898272943) q[4];
ry(1.4504298097649446) q[5];
cx q[4],q[5];
ry(-3.1413409170225433) q[4];
ry(1.74925496434788) q[5];
cx q[4],q[5];
ry(1.4228512481060571) q[5];
ry(2.4425218920675396) q[6];
cx q[5],q[6];
ry(-1.05145957943494) q[5];
ry(0.0677182121869464) q[6];
cx q[5],q[6];
ry(0.9247237177381934) q[6];
ry(-2.3737455066104234) q[7];
cx q[6],q[7];
ry(0.2819123196656656) q[6];
ry(0.6605598284580079) q[7];
cx q[6],q[7];
ry(2.0320362985531624) q[7];
ry(0.2696829464038357) q[8];
cx q[7],q[8];
ry(-1.6990518016819274) q[7];
ry(-1.1523298767470627) q[8];
cx q[7],q[8];
ry(2.7605886664518153) q[8];
ry(-0.36603705890989424) q[9];
cx q[8],q[9];
ry(0.008016938684825448) q[8];
ry(0.06422221642223742) q[9];
cx q[8],q[9];
ry(-1.5711733250226398) q[9];
ry(-0.45963333006916784) q[10];
cx q[9],q[10];
ry(0.06802817760984674) q[9];
ry(-2.0740177539704163) q[10];
cx q[9],q[10];
ry(-1.7007158505234532) q[10];
ry(2.0169026290344485) q[11];
cx q[10],q[11];
ry(-0.5789758311005413) q[10];
ry(-2.5573318239881173) q[11];
cx q[10],q[11];
ry(-2.122160767866367) q[11];
ry(-1.1872671658310732) q[12];
cx q[11],q[12];
ry(-1.035234892755959) q[11];
ry(-0.7893526882058743) q[12];
cx q[11],q[12];
ry(3.0387546345173395) q[12];
ry(-2.9659885336384058) q[13];
cx q[12],q[13];
ry(-3.139864857729389) q[12];
ry(2.868999925726716) q[13];
cx q[12],q[13];
ry(-1.7291227783588543) q[13];
ry(-0.6116190102569199) q[14];
cx q[13],q[14];
ry(-1.5346377112248453) q[13];
ry(-2.311197591088248) q[14];
cx q[13],q[14];
ry(-2.906590340294112) q[14];
ry(-2.6164583610568686) q[15];
cx q[14],q[15];
ry(-2.429861949025775) q[14];
ry(2.9044043482807687) q[15];
cx q[14],q[15];
ry(0.9225219727378948) q[15];
ry(-1.1024714919392515) q[16];
cx q[15],q[16];
ry(-3.129804595163163) q[15];
ry(-0.006127434741088677) q[16];
cx q[15],q[16];
ry(-1.9396114867374232) q[16];
ry(0.6892848200749905) q[17];
cx q[16],q[17];
ry(1.290995665702744) q[16];
ry(0.00148256085196547) q[17];
cx q[16],q[17];
ry(-1.5106721252374464) q[17];
ry(1.0820566574826032) q[18];
cx q[17],q[18];
ry(-0.20725216447916672) q[17];
ry(-3.027944564977074) q[18];
cx q[17],q[18];
ry(-1.6429453072173552) q[18];
ry(-1.1661268580080586) q[19];
cx q[18],q[19];
ry(-1.9019893362284925) q[18];
ry(2.07680056366045) q[19];
cx q[18],q[19];
ry(0.3573191067508911) q[0];
ry(2.167692478521174) q[1];
cx q[0],q[1];
ry(-0.09137325826605418) q[0];
ry(-0.06372352472830123) q[1];
cx q[0],q[1];
ry(1.6426197566209604) q[1];
ry(-2.298166092254606) q[2];
cx q[1],q[2];
ry(-2.5803421636883117) q[1];
ry(0.40713436088764965) q[2];
cx q[1],q[2];
ry(-2.190260234313002) q[2];
ry(-0.8174596089239753) q[3];
cx q[2],q[3];
ry(-3.1405911470788346) q[2];
ry(2.578424165274504) q[3];
cx q[2],q[3];
ry(-2.504071751289289) q[3];
ry(-2.38943207300477) q[4];
cx q[3],q[4];
ry(-3.0792210363494523) q[3];
ry(3.1393116231778575) q[4];
cx q[3],q[4];
ry(-0.07358690186455215) q[4];
ry(-0.18712190095067752) q[5];
cx q[4],q[5];
ry(3.125418429020892) q[4];
ry(-1.2084001469723455) q[5];
cx q[4],q[5];
ry(2.427302539415373) q[5];
ry(0.5271402160730405) q[6];
cx q[5],q[6];
ry(2.5258322759804104) q[5];
ry(-0.31013293734956154) q[6];
cx q[5],q[6];
ry(-0.8103615315070514) q[6];
ry(-0.3194626858714047) q[7];
cx q[6],q[7];
ry(3.118385663456782) q[6];
ry(-0.0018235341433081587) q[7];
cx q[6],q[7];
ry(0.1442301539504611) q[7];
ry(-2.5931347163019893) q[8];
cx q[7],q[8];
ry(-1.5281942104796533) q[7];
ry(2.176407937544255) q[8];
cx q[7],q[8];
ry(-1.6979736821573734) q[8];
ry(-1.598028780236361) q[9];
cx q[8],q[9];
ry(2.997322238179622) q[8];
ry(2.809701178020239) q[9];
cx q[8],q[9];
ry(1.5175937559413493) q[9];
ry(1.758761012860355) q[10];
cx q[9],q[10];
ry(-0.023844662361918827) q[9];
ry(-3.0306622209709535) q[10];
cx q[9],q[10];
ry(0.4344493324209883) q[10];
ry(1.5403595752630954) q[11];
cx q[10],q[11];
ry(0.1611163224109578) q[10];
ry(-3.1397083990236756) q[11];
cx q[10],q[11];
ry(-1.608845582020276) q[11];
ry(2.6841391462863036) q[12];
cx q[11],q[12];
ry(0.04065159727043799) q[11];
ry(-2.7837623304878387) q[12];
cx q[11],q[12];
ry(0.4261396618487376) q[12];
ry(-1.590341444798573) q[13];
cx q[12],q[13];
ry(3.067367166616674) q[12];
ry(-3.1370040686944995) q[13];
cx q[12],q[13];
ry(-1.5797809921405432) q[13];
ry(1.5236763678124925) q[14];
cx q[13],q[14];
ry(3.1306493349360283) q[13];
ry(0.38948351855985663) q[14];
cx q[13],q[14];
ry(-2.9338826729532195) q[14];
ry(-0.9825493443080173) q[15];
cx q[14],q[15];
ry(2.452825921137918) q[14];
ry(-2.8191073688073716) q[15];
cx q[14],q[15];
ry(-2.6392536348858866) q[15];
ry(2.6208376354387535) q[16];
cx q[15],q[16];
ry(-0.05076302795660226) q[15];
ry(0.4634349448399586) q[16];
cx q[15],q[16];
ry(1.5586346780907343) q[16];
ry(-0.8807994510585563) q[17];
cx q[16],q[17];
ry(-3.054730571165392) q[16];
ry(2.708056505848673) q[17];
cx q[16],q[17];
ry(0.6186463527060573) q[17];
ry(1.5446065837060914) q[18];
cx q[17],q[18];
ry(-3.0839273940150167) q[17];
ry(-0.44589917863688733) q[18];
cx q[17],q[18];
ry(-0.6229251146259678) q[18];
ry(0.49027221134560506) q[19];
cx q[18],q[19];
ry(1.957678045997277) q[18];
ry(-2.717481980884352) q[19];
cx q[18],q[19];
ry(-2.283232747793833) q[0];
ry(-0.03483021080795173) q[1];
ry(0.3921436648179153) q[2];
ry(0.2881212196746611) q[3];
ry(0.15713770002735714) q[4];
ry(0.16905747921412492) q[5];
ry(-1.9162336679186198) q[6];
ry(1.2581610991222887) q[7];
ry(3.1340788865994993) q[8];
ry(-0.01796499669997172) q[9];
ry(-0.9446879189401072) q[10];
ry(3.036438933867057) q[11];
ry(3.1137624181117536) q[12];
ry(3.1341002959748967) q[13];
ry(-2.047760539706659) q[14];
ry(-0.013655576859711083) q[15];
ry(-0.00019647759759688912) q[16];
ry(-3.1274274157978117) q[17];
ry(3.1382902207781833) q[18];
ry(-2.0118096627204336) q[19];