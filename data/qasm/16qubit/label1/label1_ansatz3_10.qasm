OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.882866568576711) q[0];
rz(0.19706730576310486) q[0];
ry(-3.1398431003886) q[1];
rz(1.9380024971909924) q[1];
ry(3.139236376423097) q[2];
rz(1.0780884735510048) q[2];
ry(2.229482930616829) q[3];
rz(-2.696594179286574) q[3];
ry(-0.39840854321247615) q[4];
rz(-2.9169488588447114) q[4];
ry(-0.645368940786588) q[5];
rz(-0.3397558241393377) q[5];
ry(-3.0769469715375783) q[6];
rz(-2.747392194043055) q[6];
ry(-0.7493668644571454) q[7];
rz(2.0864720347713863) q[7];
ry(-0.19267532686651276) q[8];
rz(0.07316774820240063) q[8];
ry(2.2741654689476176) q[9];
rz(3.1288872562928924) q[9];
ry(1.8985266134545606) q[10];
rz(-2.9794039264107166) q[10];
ry(-0.626835013729791) q[11];
rz(3.0622167398201174) q[11];
ry(-2.7077146025111434) q[12];
rz(-2.6323924996192436) q[12];
ry(0.03930439217901416) q[13];
rz(-2.2162832141904154) q[13];
ry(3.0753620188184576) q[14];
rz(-2.1800750996803018) q[14];
ry(2.846991542535273) q[15];
rz(-0.013459891716092283) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.9603117506416177) q[0];
rz(2.5910900262316985) q[0];
ry(-0.003787096813582025) q[1];
rz(1.412236110430383) q[1];
ry(3.1138600266414853) q[2];
rz(-0.24400834720003228) q[2];
ry(-1.5737933141938023) q[3];
rz(1.1111246056227042) q[3];
ry(2.705013495747445) q[4];
rz(-2.697936180065234) q[4];
ry(-1.6460302069208899) q[5];
rz(0.01618240104726976) q[5];
ry(-0.8698284028773752) q[6];
rz(-2.692477220250064) q[6];
ry(0.01614697118926858) q[7];
rz(-0.769721342755382) q[7];
ry(-0.24883934533842297) q[8];
rz(2.994516968216007) q[8];
ry(0.0015876572022275326) q[9];
rz(-2.290009458409447) q[9];
ry(-2.350729983803689) q[10];
rz(-0.3204256539813795) q[10];
ry(-2.057718120396402) q[11];
rz(-0.21564260383425676) q[11];
ry(-0.4825104589268265) q[12];
rz(-1.3828299297091502) q[12];
ry(-1.5703362309225024) q[13];
rz(1.5699111607448364) q[13];
ry(-1.4054223577358835) q[14];
rz(-1.2323990346326612) q[14];
ry(-0.30826536468583265) q[15];
rz(-0.19711777884381387) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-3.103184661320578) q[0];
rz(1.9545484910964595) q[0];
ry(-3.133737549651107) q[1];
rz(-2.632223898088096) q[1];
ry(-0.0021869294714775833) q[2];
rz(-1.516385039152819) q[2];
ry(-0.060306235265538355) q[3];
rz(1.6320694342691437) q[3];
ry(1.6059923986826972) q[4];
rz(-1.245443657121376) q[4];
ry(2.8051355005862155) q[5];
rz(-1.9739668295379007) q[5];
ry(0.10022115992268193) q[6];
rz(1.4089367909861605) q[6];
ry(-2.5423281259093016) q[7];
rz(2.879911398524383) q[7];
ry(-1.8961303667461578) q[8];
rz(0.03281789277021919) q[8];
ry(-0.3428302708289906) q[9];
rz(2.333202981424243) q[9];
ry(-0.485448835276878) q[10];
rz(1.8664337559498296) q[10];
ry(-3.0240724455230965) q[11];
rz(0.0083618701088648) q[11];
ry(-0.1321230352490003) q[12];
rz(-2.7923990234461415) q[12];
ry(-1.6568643266296745) q[13];
rz(-3.0452597875730136) q[13];
ry(-0.8493673039541345) q[14];
rz(0.6946564358879348) q[14];
ry(2.0975125159738433) q[15];
rz(2.5382212890597975) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.7900633135649064) q[0];
rz(-2.0124471300420397) q[0];
ry(-0.0082004675342775) q[1];
rz(0.13434802626969408) q[1];
ry(-3.133408565742925) q[2];
rz(0.92898983301861) q[2];
ry(1.064971300272373) q[3];
rz(-1.815843277286602) q[3];
ry(1.9771827458908648) q[4];
rz(2.58589469396366) q[4];
ry(1.33626141544621) q[5];
rz(-0.5358571933249063) q[5];
ry(3.1348143700717013) q[6];
rz(-0.8795285261683841) q[6];
ry(3.1210745200463887) q[7];
rz(1.589333728779038) q[7];
ry(2.953776037287291) q[8];
rz(-2.276209699528078) q[8];
ry(-3.1143682805038573) q[9];
rz(-1.1860490773490628) q[9];
ry(0.021655621763745362) q[10];
rz(1.2192717713714005) q[10];
ry(-1.093536044830412) q[11];
rz(0.014033359658180089) q[11];
ry(-0.9684140872340785) q[12];
rz(-0.8063058544361494) q[12];
ry(-3.0185113537187007) q[13];
rz(-0.769954903567518) q[13];
ry(-2.3059610557394037) q[14];
rz(-2.904598451152295) q[14];
ry(-2.6721858595850154) q[15];
rz(2.812080408962783) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.5380047470725785) q[0];
rz(0.7801299949630973) q[0];
ry(3.1112747220226695) q[1];
rz(-0.7848509294366197) q[1];
ry(0.012589052465195927) q[2];
rz(-0.1883838011990555) q[2];
ry(-1.2244434334650491) q[3];
rz(2.214047755242268) q[3];
ry(-1.3132269833344479) q[4];
rz(-0.39606210038427747) q[4];
ry(1.6312682618911358) q[5];
rz(-1.358336694875401) q[5];
ry(-3.078614824190412) q[6];
rz(0.5402443917665382) q[6];
ry(-1.6838775654551672) q[7];
rz(0.8847795428268155) q[7];
ry(2.6910118310130873) q[8];
rz(0.7901636309461074) q[8];
ry(-0.5119587477341168) q[9];
rz(1.6203667478911805) q[9];
ry(2.8063817076342783) q[10];
rz(-0.09066081691115001) q[10];
ry(3.0862587746406347) q[11];
rz(-2.847717809026647) q[11];
ry(-0.14894928899947044) q[12];
rz(-0.0579603187395028) q[12];
ry(3.130044270346874) q[13];
rz(-2.181589021448787) q[13];
ry(0.2681448475432493) q[14];
rz(2.8241788011481415) q[14];
ry(-3.006670954029924) q[15];
rz(2.0493998501972284) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.099856001852038) q[0];
rz(-2.3118106144726154) q[0];
ry(1.524343895525023) q[1];
rz(0.00026205072025309306) q[1];
ry(0.018271778482843892) q[2];
rz(2.252959837208179) q[2];
ry(0.3174104855465275) q[3];
rz(2.7627916886013786) q[3];
ry(0.05903389515355412) q[4];
rz(1.962624444564102) q[4];
ry(-2.31841763783201) q[5];
rz(-1.1575413075721988) q[5];
ry(3.1413837248436165) q[6];
rz(-0.10505891431169445) q[6];
ry(-0.006150942862019328) q[7];
rz(1.5673805559298752) q[7];
ry(-0.6557666993583818) q[8];
rz(3.1360198079475463) q[8];
ry(-0.012340483426068076) q[9];
rz(2.7989915155475815) q[9];
ry(3.126610673243485) q[10];
rz(1.0057859517386543) q[10];
ry(-2.7191538004944706) q[11];
rz(0.2587943962140343) q[11];
ry(2.039802414518582) q[12];
rz(-3.022438351172016) q[12];
ry(-2.441516955674328) q[13];
rz(2.9993647934203147) q[13];
ry(-0.8235701218246012) q[14];
rz(1.5916646373918573) q[14];
ry(1.3592695888405997) q[15];
rz(-0.18741869199700303) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(0.026864742493746374) q[0];
rz(0.03371904897758515) q[0];
ry(-1.5598345559447573) q[1];
rz(-3.138484520212549) q[1];
ry(0.10843481054256099) q[2];
rz(1.4790706892982746) q[2];
ry(-0.005099177289865732) q[3];
rz(2.610939000211791) q[3];
ry(0.5743573529121401) q[4];
rz(-0.1847118818497851) q[4];
ry(-1.915419138298958) q[5];
rz(-2.179654339953332) q[5];
ry(-0.052933563531524896) q[6];
rz(0.15015055731110397) q[6];
ry(1.483340890501288) q[7];
rz(1.7239284843403158) q[7];
ry(1.5658278813135107) q[8];
rz(0.9655435027941699) q[8];
ry(-0.7315966783136059) q[9];
rz(-3.11229063893313) q[9];
ry(2.9948526024855866) q[10];
rz(0.4657720379282481) q[10];
ry(0.06708311410662304) q[11];
rz(2.514562351871676) q[11];
ry(3.0071608857895455) q[12];
rz(2.208964288621334) q[12];
ry(-0.03584003761477721) q[13];
rz(1.7958633661913823) q[13];
ry(-2.2286845251989114) q[14];
rz(-0.269910446632716) q[14];
ry(-0.9036861121884633) q[15];
rz(-2.84695126685897) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.5682868646208954) q[0];
rz(-1.27185645244323) q[0];
ry(1.534423503343957) q[1];
rz(1.6370820755644226) q[1];
ry(-3.1199650739532534) q[2];
rz(-0.14884717927853058) q[2];
ry(-2.834781664088439) q[3];
rz(-2.9847901167550392) q[3];
ry(-1.4977470913289386) q[4];
rz(0.36651901068346154) q[4];
ry(-0.17378550336474596) q[5];
rz(1.9545700828990622) q[5];
ry(-1.0834340884935587) q[6];
rz(-2.655508113566362) q[6];
ry(1.5712057249279856) q[7];
rz(-0.2619135187131052) q[7];
ry(0.23021934197165006) q[8];
rz(2.024471273332849) q[8];
ry(0.6530264286557019) q[9];
rz(3.064686855780656) q[9];
ry(-0.3422653386954867) q[10];
rz(-1.0339330036475305) q[10];
ry(2.82897426849425) q[11];
rz(-1.3924433396715514) q[11];
ry(2.3158373223356357) q[12];
rz(-2.8224598796051907) q[12];
ry(-1.7618461054474084) q[13];
rz(-1.8029051234693947) q[13];
ry(-2.287695185562438) q[14];
rz(1.4490610874210814) q[14];
ry(1.1291163585577937) q[15];
rz(-1.1544505839320216) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-0.12215018540770295) q[0];
rz(1.5025779202293936) q[0];
ry(-0.80052282515625) q[1];
rz(2.676940385549867) q[1];
ry(1.7817503514290012) q[2];
rz(3.052830597569524) q[2];
ry(0.8096353675700483) q[3];
rz(0.7549033068545857) q[3];
ry(2.70690970709431) q[4];
rz(2.6761963043329793) q[4];
ry(-3.139807738989001) q[5];
rz(2.0448501510875827) q[5];
ry(-0.0005321588941802347) q[6];
rz(0.4776017030579134) q[6];
ry(0.01523659961781032) q[7];
rz(1.9601626818744948) q[7];
ry(-0.08126535824293918) q[8];
rz(-1.6266545202000122) q[8];
ry(-1.5831951131123783) q[9];
rz(3.121572762700945) q[9];
ry(-3.110776348848544) q[10];
rz(2.8464053281660906) q[10];
ry(-3.1260646665978205) q[11];
rz(0.6093475595728792) q[11];
ry(3.094561469125116) q[12];
rz(1.1476834594531562) q[12];
ry(-2.8671993470062) q[13];
rz(-0.7217189431572891) q[13];
ry(1.2410010124746114) q[14];
rz(2.306399919880537) q[14];
ry(0.7612773436261673) q[15];
rz(-2.490387329466806) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(2.901107876349811) q[0];
rz(-2.1392577018995027) q[0];
ry(-2.269339651862527) q[1];
rz(3.0147831942541163) q[1];
ry(-1.160185112613413) q[2];
rz(-3.060791057385439) q[2];
ry(-3.130827856821877) q[3];
rz(0.3282938868404252) q[3];
ry(-0.14568614898532195) q[4];
rz(-1.5583006759550466) q[4];
ry(-1.7186836194033992) q[5];
rz(-3.1049041456890696) q[5];
ry(-0.20979938473918214) q[6];
rz(2.2731065011220943) q[6];
ry(-1.3944460902332567) q[7];
rz(0.9321901208160498) q[7];
ry(1.5740320567928592) q[8];
rz(-0.21364067659681332) q[8];
ry(-2.245363997609405) q[9];
rz(1.5514229061046256) q[9];
ry(2.6825723782322934) q[10];
rz(2.9801827009911674) q[10];
ry(-0.01689786764053824) q[11];
rz(1.921123944304889) q[11];
ry(2.649819055942303) q[12];
rz(-0.297707471375964) q[12];
ry(0.847113130529701) q[13];
rz(2.720472067727117) q[13];
ry(1.3964180452465964) q[14];
rz(0.025273686971392092) q[14];
ry(2.4627230852728825) q[15];
rz(-0.09068301968239911) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.9555989179670341) q[0];
rz(-1.2441424107844128) q[0];
ry(-1.372234135475775) q[1];
rz(1.1299681776734978) q[1];
ry(1.3591825783555516) q[2];
rz(3.1188698668465173) q[2];
ry(3.1331727869139194) q[3];
rz(2.4072326910769286) q[3];
ry(0.0354989294601209) q[4];
rz(2.0998365814717794) q[4];
ry(-3.1409009946054582) q[5];
rz(-0.3135166129242964) q[5];
ry(3.092408730533128) q[6];
rz(-1.6657235086833075) q[6];
ry(2.540154333451) q[7];
rz(0.03884123282558339) q[7];
ry(-0.02024083774059088) q[8];
rz(0.17470997574346805) q[8];
ry(1.5726350981843025) q[9];
rz(-1.2507192746980458) q[9];
ry(3.133669013477455) q[10];
rz(-2.0379029464387335) q[10];
ry(-0.021883429683562296) q[11];
rz(-0.5719380078256435) q[11];
ry(0.10272528022763898) q[12];
rz(0.4221147659177902) q[12];
ry(-2.9238065107635287) q[13];
rz(1.0370367918859167) q[13];
ry(1.6534187774763718) q[14];
rz(3.078988653061754) q[14];
ry(2.4829121305626605) q[15];
rz(-1.150862131693999) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.558457013983011) q[0];
rz(-3.106704598484005) q[0];
ry(1.4125117250439505) q[1];
rz(1.6777912212527326) q[1];
ry(-0.16429408688327168) q[2];
rz(1.2686655653685737) q[2];
ry(2.980242080667485) q[3];
rz(3.136518130366687) q[3];
ry(-2.635919517821831) q[4];
rz(2.049963961357748) q[4];
ry(3.1183407191518655) q[5];
rz(-1.7766825359242635) q[5];
ry(-0.05908316706600853) q[6];
rz(-1.112327747653123) q[6];
ry(1.5858651171275087) q[7];
rz(2.9232549838258235) q[7];
ry(-1.5792582665434205) q[8];
rz(2.9646656672057254) q[8];
ry(0.08300331142952816) q[9];
rz(-0.2709969837448405) q[9];
ry(2.5716327870228874) q[10];
rz(-2.642904765946252) q[10];
ry(0.053685519970015416) q[11];
rz(1.0216458307246121) q[11];
ry(2.6534407178969643) q[12];
rz(-2.8839071517997197) q[12];
ry(1.1412627164502531) q[13];
rz(3.1249118519756403) q[13];
ry(2.217133523555878) q[14];
rz(1.3538282312130194) q[14];
ry(0.527638059031214) q[15];
rz(-0.5279642880926199) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(-1.4215384710005106) q[0];
rz(3.127892398119268) q[0];
ry(-1.5704701280589273) q[1];
rz(0.4266737106232572) q[1];
ry(-3.126374200066166) q[2];
rz(1.1090168167239522) q[2];
ry(3.1406538568417863) q[3];
rz(2.948988210536597) q[3];
ry(-3.105798770082644) q[4];
rz(-3.081785485058911) q[4];
ry(0.0207172209839408) q[5];
rz(3.078802491929123) q[5];
ry(-3.1309861112664623) q[6];
rz(-2.762913411106755) q[6];
ry(0.6071778360074279) q[7];
rz(-1.4134881136667812) q[7];
ry(-3.09993384778961) q[8];
rz(0.5317404932333236) q[8];
ry(1.5520787718433677) q[9];
rz(1.313237770248808) q[9];
ry(-0.04769620938460471) q[10];
rz(-0.5200330280263774) q[10];
ry(0.024038569710079262) q[11];
rz(0.5769744938657454) q[11];
ry(-3.0972105513874753) q[12];
rz(1.4464722885462535) q[12];
ry(-0.10668161010861807) q[13];
rz(1.8002970369840234) q[13];
ry(-1.6236141010021266) q[14];
rz(1.3435916116119708) q[14];
ry(-1.6366194427033443) q[15];
rz(0.17379385740435194) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
ry(1.554022207776943) q[0];
rz(-3.0955817101193683) q[0];
ry(0.10290927941132111) q[1];
rz(-0.39877726601686275) q[1];
ry(3.032777650807119) q[2];
rz(3.018640574260575) q[2];
ry(1.5589847673052066) q[3];
rz(1.5797134954062575) q[3];
ry(1.3709888692017032) q[4];
rz(1.1525086492518266) q[4];
ry(-0.5813764727497519) q[5];
rz(0.04851334493308941) q[5];
ry(-0.436681228805184) q[6];
rz(0.04360169042057293) q[6];
ry(-0.9832283104503848) q[7];
rz(-3.065664776501446) q[7];
ry(1.6598309254379986) q[8];
rz(-1.4520174653125306) q[8];
ry(-2.803773435118914) q[9];
rz(2.9617870594691214) q[9];
ry(-0.9038619690581228) q[10];
rz(2.133755097467854) q[10];
ry(1.624510393249059) q[11];
rz(-1.3049955203039096) q[11];
ry(1.3396253187359837) q[12];
rz(-1.553107794415339) q[12];
ry(1.710133893922056) q[13];
rz(2.697709485651306) q[13];
ry(0.8387005650796258) q[14];
rz(-2.4202209179354353) q[14];
ry(0.7104540711243006) q[15];
rz(1.6099996179433171) q[15];