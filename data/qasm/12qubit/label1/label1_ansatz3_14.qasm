OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.08153853697038546) q[0];
rz(-1.082387815319677) q[0];
ry(2.718646155664029) q[1];
rz(2.6463538068515677) q[1];
ry(1.8360122332792503) q[2];
rz(0.35440090306939354) q[2];
ry(3.090255393620209) q[3];
rz(2.3786963817221114) q[3];
ry(2.9364954430778227) q[4];
rz(0.489166998650064) q[4];
ry(3.1024042179725546) q[5];
rz(-0.59204550816915) q[5];
ry(-1.9112063653781644) q[6];
rz(-2.457037631828013) q[6];
ry(1.5222581985270522) q[7];
rz(-1.680591682268637) q[7];
ry(-0.08108559185321607) q[8];
rz(-0.9929664287036326) q[8];
ry(2.0649414230313106) q[9];
rz(-1.8653543597637405) q[9];
ry(0.46699753902908464) q[10];
rz(2.761854615301701) q[10];
ry(-3.1243848842282804) q[11];
rz(-1.4076981093113492) q[11];
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
ry(0.2411829630561435) q[0];
rz(-1.918824162462582) q[0];
ry(0.6316081089946017) q[1];
rz(-1.4359987895134383) q[1];
ry(3.0297767309206685) q[2];
rz(-1.860016192324612) q[2];
ry(-1.27807689969171) q[3];
rz(1.5622375085098104) q[3];
ry(-0.002346708070518319) q[4];
rz(-1.8869664151323438) q[4];
ry(-0.3955341108993551) q[5];
rz(-0.11294789367409275) q[5];
ry(2.191682742043975) q[6];
rz(1.9160639630952587) q[6];
ry(-2.5924353859987064) q[7];
rz(3.112927276132059) q[7];
ry(0.05329361850233938) q[8];
rz(-1.6990487096127769) q[8];
ry(-0.44460748664521504) q[9];
rz(-2.327684259422406) q[9];
ry(-0.23530545044590623) q[10];
rz(-3.0747634079129997) q[10];
ry(-2.738837386135279) q[11];
rz(2.0457892740495764) q[11];
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
ry(3.0166430919007006) q[0];
rz(0.8059262905841089) q[0];
ry(-2.5000346629205414) q[1];
rz(-0.6729833309227383) q[1];
ry(0.33047452895566387) q[2];
rz(-1.655145845410165) q[2];
ry(3.121135955661456) q[3];
rz(-0.4063032112674953) q[3];
ry(0.728238162914789) q[4];
rz(-0.9694089717859571) q[4];
ry(-0.006024569078516998) q[5];
rz(-2.1428608384595327) q[5];
ry(1.0605225252246715) q[6];
rz(1.8500597504420728) q[6];
ry(-1.1901553231973612) q[7];
rz(-1.3958695390582783) q[7];
ry(3.111196424853124) q[8];
rz(1.13787082346614) q[8];
ry(2.116196086390243) q[9];
rz(-2.4314162772061962) q[9];
ry(-0.874607330592348) q[10];
rz(3.1012348321083225) q[10];
ry(2.9888726821610536) q[11];
rz(2.334166475750094) q[11];
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
ry(-0.14861745327071524) q[0];
rz(-1.6425838810882278) q[0];
ry(2.3838576565503593) q[1];
rz(-0.23121326303350725) q[1];
ry(3.013018773592816) q[2];
rz(2.399899318620922) q[2];
ry(3.0005581636714975) q[3];
rz(-1.5983399408453465) q[3];
ry(2.446246751825472) q[4];
rz(-3.1282080859247707) q[4];
ry(0.60381141631661) q[5];
rz(2.144530763159297) q[5];
ry(1.07177775068795) q[6];
rz(-2.4365464524891927) q[6];
ry(-0.18884504505288113) q[7];
rz(-2.899688480945701) q[7];
ry(1.3684960055921804) q[8];
rz(-2.4698118692753432) q[8];
ry(0.05449777367394315) q[9];
rz(0.9580214404122713) q[9];
ry(-3.056369537288914) q[10];
rz(-0.2727851610760332) q[10];
ry(2.9166027241677037) q[11];
rz(0.8035707656764298) q[11];
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
ry(0.546723309806608) q[0];
rz(-2.9849753047794376) q[0];
ry(1.8618946947547295) q[1];
rz(-1.0477550891799234) q[1];
ry(1.845779977181615) q[2];
rz(-3.1048814523714614) q[2];
ry(0.021304249724910997) q[3];
rz(2.7212527627937804) q[3];
ry(1.2662112440363043) q[4];
rz(-0.041135275323657385) q[4];
ry(-3.133283884838981) q[5];
rz(3.0912541565898586) q[5];
ry(-3.137605705523197) q[6];
rz(-1.1305662982673215) q[6];
ry(0.05686035745679405) q[7];
rz(-3.0754082474761795) q[7];
ry(0.008969648426382771) q[8];
rz(1.2964071991037005) q[8];
ry(0.6395676427385999) q[9];
rz(0.3997077104260036) q[9];
ry(-2.8868887844172075) q[10];
rz(0.8283686411605355) q[10];
ry(2.542311684867055) q[11];
rz(1.267104778572396) q[11];
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
ry(0.1321626801830002) q[0];
rz(-0.20214967385317004) q[0];
ry(-2.893398994335727) q[1];
rz(1.5229509679022932) q[1];
ry(2.9320813142499538) q[2];
rz(-3.137848574244654) q[2];
ry(-1.1462409537772196) q[3];
rz(-2.4130496113066573) q[3];
ry(-1.4779902374746738) q[4];
rz(1.0350926469508837) q[4];
ry(-2.8229600359946687) q[5];
rz(-1.7625017796546327) q[5];
ry(-1.7553800848867875) q[6];
rz(0.6329871008948248) q[6];
ry(0.16706794308854978) q[7];
rz(-0.3983100397513111) q[7];
ry(0.8816061317815811) q[8];
rz(2.846493600282893) q[8];
ry(0.38375220919379416) q[9];
rz(-0.7815725788650658) q[9];
ry(-2.234656245562025) q[10];
rz(1.7639158685281) q[10];
ry(2.625825665635866) q[11];
rz(-0.977261626824027) q[11];
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
ry(-3.044062581013024) q[0];
rz(1.932284180220796) q[0];
ry(-0.5547074822551896) q[1];
rz(-2.528500160187264) q[1];
ry(-1.772325375853164) q[2];
rz(0.11173633754484946) q[2];
ry(-3.1131450917747983) q[3];
rz(1.0858575016660807) q[3];
ry(0.00013009080987150782) q[4];
rz(0.5345412100946132) q[4];
ry(-0.0037977020689106138) q[5];
rz(2.1123136361010606) q[5];
ry(3.141121511897283) q[6];
rz(-2.991946112605694) q[6];
ry(0.16587185409692928) q[7];
rz(-2.9661615153229257) q[7];
ry(3.141015037271128) q[8];
rz(0.17542248333593524) q[8];
ry(1.8044292685283134) q[9];
rz(2.864452888529564) q[9];
ry(-0.0005376078344738147) q[10];
rz(1.1750748184563398) q[10];
ry(-2.3970544013087896) q[11];
rz(0.05916623898058049) q[11];
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
ry(-0.10987628993712573) q[0];
rz(0.9788505002195504) q[0];
ry(1.7895773869991107) q[1];
rz(1.9420560603940844) q[1];
ry(-1.828318636137542) q[2];
rz(-1.6916482484993032) q[2];
ry(-1.2827560513865732) q[3];
rz(-2.3226942783080387) q[3];
ry(1.5260520804574256) q[4];
rz(-2.8048476433215956) q[4];
ry(-0.6411624966976626) q[5];
rz(1.255012361746891) q[5];
ry(-1.449411960853503) q[6];
rz(0.7923202731671385) q[6];
ry(-2.7638455929363857) q[7];
rz(1.748107588784491) q[7];
ry(1.836888433273594) q[8];
rz(-2.4461064375180057) q[8];
ry(2.8166362303838084) q[9];
rz(-2.263047605657933) q[9];
ry(0.5395100324767617) q[10];
rz(-0.5679535874540356) q[10];
ry(-0.20870431058907213) q[11];
rz(-2.3746868119031834) q[11];
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
ry(3.1412074141843824) q[0];
rz(-2.6348784844081) q[0];
ry(1.6792867110678447) q[1];
rz(-2.3687905621634573) q[1];
ry(-0.17214775082887562) q[2];
rz(0.15951302906771803) q[2];
ry(3.1351785615285137) q[3];
rz(-0.6152957131024823) q[3];
ry(-1.5612084481063997) q[4];
rz(1.6697236316598127) q[4];
ry(0.0030944334006255403) q[5];
rz(-2.4749744151390796) q[5];
ry(-3.141268754019813) q[6];
rz(-2.885414716093418) q[6];
ry(3.1328814291705616) q[7];
rz(1.8338194366518645) q[7];
ry(-3.12249203457567) q[8];
rz(-2.488653746303207) q[8];
ry(1.4081283797060937) q[9];
rz(-0.7887793267377603) q[9];
ry(1.1257902814628302) q[10];
rz(0.7483906836710944) q[10];
ry(-1.3688307302301204) q[11];
rz(-2.032730277624391) q[11];
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
ry(-0.08683704379391424) q[0];
rz(1.6535748680822158) q[0];
ry(-0.5153175621778069) q[1];
rz(2.1635215059421364) q[1];
ry(-1.571978467474305) q[2];
rz(-1.8071231426355494) q[2];
ry(1.6154137697515483) q[3];
rz(0.2562624142279937) q[3];
ry(-0.5087813994432233) q[4];
rz(-0.11915543529099361) q[4];
ry(3.136246405316971) q[5];
rz(2.7519711472843214) q[5];
ry(3.1262748815548482) q[6];
rz(1.0015739849407068) q[6];
ry(2.643000750618788) q[7];
rz(2.147093932123113) q[7];
ry(1.407130639817561) q[8];
rz(-0.8673087973173804) q[8];
ry(-2.324422993710726) q[9];
rz(1.1499602581212522) q[9];
ry(1.9012863693800612) q[10];
rz(-2.8402184464170075) q[10];
ry(0.04228189753633466) q[11];
rz(0.842188632823155) q[11];
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
ry(-0.0470438101500289) q[0];
rz(0.3568609441561339) q[0];
ry(-0.3914791302979985) q[1];
rz(-2.1575369551413166) q[1];
ry(1.603094137741242) q[2];
rz(-2.1101730474721294) q[2];
ry(-3.133959776833654) q[3];
rz(0.16924647347934837) q[3];
ry(1.5765116172872704) q[4];
rz(3.026570700614747) q[4];
ry(3.1409224594387744) q[5];
rz(-0.10836070490057192) q[5];
ry(-3.1412265069126413) q[6];
rz(3.0610331257034997) q[6];
ry(0.0034459212714539338) q[7];
rz(2.2176631895128764) q[7];
ry(-0.011336977147464289) q[8];
rz(-2.0518318830238886) q[8];
ry(0.2528281593178669) q[9];
rz(3.1367724149590424) q[9];
ry(-2.846699228553289) q[10];
rz(2.461841608054349) q[10];
ry(-0.40466725547146076) q[11];
rz(-1.6264344366140566) q[11];
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
ry(0.24112418093979215) q[0];
rz(2.6918030633257866) q[0];
ry(-0.9607355441298137) q[1];
rz(-1.8239028578754588) q[1];
ry(0.2485126899009913) q[2];
rz(0.8067080344725079) q[2];
ry(-1.5519085916812003) q[3];
rz(2.1517018197559086) q[3];
ry(-1.7890946888996473) q[4];
rz(2.082737562691652) q[4];
ry(0.472492977483574) q[5];
rz(0.9620243048324933) q[5];
ry(1.5949352585471104) q[6];
rz(-0.01443657151810889) q[6];
ry(-0.21022749178291814) q[7];
rz(-2.606061863969305) q[7];
ry(1.2718740798649542) q[8];
rz(-1.8600362056351198) q[8];
ry(2.434021843969708) q[9];
rz(-2.523710581865504) q[9];
ry(-0.36062885076216755) q[10];
rz(2.6826724628250034) q[10];
ry(1.4497988258897505) q[11];
rz(-1.8721575824690921) q[11];
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
ry(1.2754410431098024) q[0];
rz(0.43906530626847884) q[0];
ry(1.6098951973673385) q[1];
rz(0.5062553515477717) q[1];
ry(1.5441321824061036) q[2];
rz(-2.6639588081502126) q[2];
ry(-3.1266763493606717) q[3];
rz(-2.6461769623997475) q[3];
ry(3.1362531716090634) q[4];
rz(-1.0529414487855098) q[4];
ry(-0.0026438076575939014) q[5];
rz(0.2794943852295182) q[5];
ry(2.908717320959653e-05) q[6];
rz(-1.0512153587223225) q[6];
ry(-0.009308018006079244) q[7];
rz(-1.9099556610451423) q[7];
ry(0.010277384076428753) q[8];
rz(2.4930484115746747) q[8];
ry(-2.6867546790636476) q[9];
rz(2.9146342251352193) q[9];
ry(0.27808217675402913) q[10];
rz(0.2920298131640449) q[10];
ry(-2.2376674620447767) q[11];
rz(1.1396483027243969) q[11];
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
ry(-3.10838282187317) q[0];
rz(1.572543787719665) q[0];
ry(1.381878112252135) q[1];
rz(-2.7582124849059864) q[1];
ry(0.3961001413718374) q[2];
rz(-0.6658838522924635) q[2];
ry(-1.690150714208344) q[3];
rz(1.159450841008783) q[3];
ry(-1.4978864601583952) q[4];
rz(-2.5328300586842887) q[4];
ry(-0.9082738147890218) q[5];
rz(2.968062445688648) q[5];
ry(-0.9505696196203708) q[6];
rz(-0.6464095144759163) q[6];
ry(-0.5399773212062566) q[7];
rz(-2.6723967860045157) q[7];
ry(2.6839474653253914) q[8];
rz(1.260321604058153) q[8];
ry(-2.4438389564306786) q[9];
rz(2.8740218726605375) q[9];
ry(1.3747948068122664) q[10];
rz(-1.6196970243462936) q[10];
ry(-2.3234166024118323) q[11];
rz(-0.0185693849860753) q[11];
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
ry(-0.3448118825131674) q[0];
rz(-1.4744891666500957) q[0];
ry(-1.472132507845342) q[1];
rz(-0.43641138261584933) q[1];
ry(-2.1279599147732315) q[2];
rz(2.1174380983745587) q[2];
ry(3.1223113547227963) q[3];
rz(-0.019871239507085466) q[3];
ry(-0.0034201852856092785) q[4];
rz(-2.422374116532332) q[4];
ry(-3.1387570704437384) q[5];
rz(0.41711573137947516) q[5];
ry(-0.013380260959760193) q[6];
rz(-2.7277450870375346) q[6];
ry(0.00102160742154243) q[7];
rz(0.1978804160648595) q[7];
ry(0.004060501414396246) q[8];
rz(-1.6910833502804452) q[8];
ry(-0.578306439203) q[9];
rz(-1.5431801166926216) q[9];
ry(-2.2939385563617405) q[10];
rz(0.20912445600325216) q[10];
ry(-1.2504588135797496) q[11];
rz(1.4014408492168249) q[11];
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
ry(-3.0517337619224154) q[0];
rz(3.0024386564025067) q[0];
ry(1.4300130660224406) q[1];
rz(-2.572208404407726) q[1];
ry(2.6282857815676124) q[2];
rz(-1.0001250874655871) q[2];
ry(-2.947608508996971) q[3];
rz(1.9569035230152636) q[3];
ry(-2.849969410720051) q[4];
rz(-1.630038867563635) q[4];
ry(-2.3206996810644993) q[5];
rz(2.563683923516382) q[5];
ry(1.3162013163660387) q[6];
rz(3.097285325098011) q[6];
ry(-1.7810514030703324) q[7];
rz(2.5154674810433044) q[7];
ry(0.12395419580673797) q[8];
rz(-1.26979737650643) q[8];
ry(-0.11144884354219795) q[9];
rz(-2.7140534800571645) q[9];
ry(-1.628358472404697) q[10];
rz(2.123583616194991) q[10];
ry(0.975506496915613) q[11];
rz(-0.03232668686923207) q[11];
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
ry(1.3469998576882898) q[0];
rz(2.5316443204015964) q[0];
ry(1.464188222432159) q[1];
rz(0.7011942577221042) q[1];
ry(-2.455877112722742) q[2];
rz(-2.8119525490454547) q[2];
ry(-3.120580407359315) q[3];
rz(0.6331225477277949) q[3];
ry(3.122851093700425) q[4];
rz(1.7249027196922628) q[4];
ry(0.020565748046813326) q[5];
rz(-0.7253476879025644) q[5];
ry(-3.126168380424584) q[6];
rz(2.8242519729536637) q[6];
ry(-0.003900865346294878) q[7];
rz(2.310065967069583) q[7];
ry(-0.028585911350847076) q[8];
rz(-2.7791779725403942) q[8];
ry(-0.12680916824511598) q[9];
rz(1.4430193691889366) q[9];
ry(0.7819326033719683) q[10];
rz(-0.23131892399801152) q[10];
ry(-2.392273164446432) q[11];
rz(1.1740606249973975) q[11];
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
ry(-0.14393645814487555) q[0];
rz(0.3173074922386405) q[0];
ry(-2.886342592234977) q[1];
rz(2.0175447800170874) q[1];
ry(-0.3039656155912161) q[2];
rz(2.0019737500116332) q[2];
ry(1.7994330821040274) q[3];
rz(-0.7283594259627281) q[3];
ry(1.7262995889477954) q[4];
rz(2.3596713078834304) q[4];
ry(0.7954464601960558) q[5];
rz(-2.433486956470814) q[5];
ry(-1.1763056509432284) q[6];
rz(2.934643387854008) q[6];
ry(1.8935842509406482) q[7];
rz(0.10102929940297313) q[7];
ry(1.461797385100313) q[8];
rz(-0.24471709328436134) q[8];
ry(-1.783883566133154) q[9];
rz(-2.0306274555286343) q[9];
ry(1.2690646626737467) q[10];
rz(-2.054111245789292) q[10];
ry(-0.5080821266966056) q[11];
rz(-1.074484298553901) q[11];