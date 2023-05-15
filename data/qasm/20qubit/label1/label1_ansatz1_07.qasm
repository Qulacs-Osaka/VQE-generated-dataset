OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-0.7783061190278433) q[0];
rz(0.47600212895156346) q[0];
ry(-0.30045776790190537) q[1];
rz(-2.485868907729887) q[1];
ry(-2.851220209365482) q[2];
rz(-3.1070402894721836) q[2];
ry(0.12338542494901183) q[3];
rz(2.6586433891561883) q[3];
ry(-1.0617742443145486) q[4];
rz(2.7188940049583943) q[4];
ry(0.7505027595653215) q[5];
rz(-1.7924205318266777) q[5];
ry(1.5066013443688115) q[6];
rz(-1.7626197932746148) q[6];
ry(-2.733480568546606) q[7];
rz(-2.1859143232283533) q[7];
ry(-0.028196302026126812) q[8];
rz(-0.3084237932631935) q[8];
ry(0.9237047300778976) q[9];
rz(0.05657712072802286) q[9];
ry(-2.265984642696581) q[10];
rz(1.0243428247065243) q[10];
ry(0.7235782354544105) q[11];
rz(-2.7145664697400296) q[11];
ry(1.6402362358513045) q[12];
rz(1.037124418359475) q[12];
ry(2.4400181529817586) q[13];
rz(-2.606446185365711) q[13];
ry(2.276930756080387) q[14];
rz(0.024482548226255185) q[14];
ry(3.1129233008769233) q[15];
rz(-1.9501152684349083) q[15];
ry(-1.6723180127887272) q[16];
rz(-3.0356998010126413) q[16];
ry(-1.6326270403658567) q[17];
rz(-2.9917649647949682) q[17];
ry(1.6287808024153876) q[18];
rz(-2.2103335928379817) q[18];
ry(3.0147106260202308) q[19];
rz(-0.055223024323444385) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.906568158354432) q[0];
rz(1.3621513560440857) q[0];
ry(-0.605486575506144) q[1];
rz(0.6886089014016245) q[1];
ry(-2.870698492788041) q[2];
rz(-1.2488834414860888) q[2];
ry(1.412877885359257) q[3];
rz(-0.04476320992103754) q[3];
ry(1.536415892199165) q[4];
rz(2.2510443941267835) q[4];
ry(0.05207407232383728) q[5];
rz(2.603683322237041) q[5];
ry(2.268492787168517) q[6];
rz(-2.8176432727255065) q[6];
ry(-1.6563587406029248) q[7];
rz(1.9389651831795502) q[7];
ry(0.012085180929906753) q[8];
rz(-1.7118408152822786) q[8];
ry(0.5393590732653338) q[9];
rz(0.26773592990191997) q[9];
ry(0.41192936041168393) q[10];
rz(0.5065806593715019) q[10];
ry(2.1849934701555895) q[11];
rz(2.215013783163206) q[11];
ry(0.2552792818840972) q[12];
rz(0.42824595509719376) q[12];
ry(-2.178777430434459) q[13];
rz(0.004462836667696202) q[13];
ry(-2.4882454287117683) q[14];
rz(-0.4187919907141503) q[14];
ry(-1.1357084593804776) q[15];
rz(1.0269456399816876) q[15];
ry(2.819554489653362) q[16];
rz(-0.4902434242914922) q[16];
ry(0.028424293013565867) q[17];
rz(1.297107522819779) q[17];
ry(-2.0832683785500126) q[18];
rz(2.603019809579798) q[18];
ry(-0.07884892569684698) q[19];
rz(0.9441701690361469) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.24508633112820322) q[0];
rz(-0.1117492187969819) q[0];
ry(0.224789729482957) q[1];
rz(-1.04861730076321) q[1];
ry(2.0091762399441517) q[2];
rz(-3.0684998708725093) q[2];
ry(-1.2190428644593045) q[3];
rz(-0.8756365996329684) q[3];
ry(0.16307225319449348) q[4];
rz(-0.3690237779298861) q[4];
ry(1.7858256902393617) q[5];
rz(1.959887468679571) q[5];
ry(1.4255348239052876) q[6];
rz(-1.5609690636670113) q[6];
ry(1.0657274608996472) q[7];
rz(1.9645791898900564) q[7];
ry(2.614838472203294) q[8];
rz(1.1775395587256314) q[8];
ry(-0.559596919984846) q[9];
rz(-2.5559110219261454) q[9];
ry(-0.8396969249641946) q[10];
rz(0.14253632888976386) q[10];
ry(-1.7815878077382523) q[11];
rz(2.6608519897164813) q[11];
ry(-0.18562581686063828) q[12];
rz(1.3815796175026298) q[12];
ry(2.3147651814304164) q[13];
rz(-0.6687123494457305) q[13];
ry(2.287897293656204) q[14];
rz(-1.2333533724075705) q[14];
ry(-2.6735050816584325) q[15];
rz(3.0814415035634535) q[15];
ry(-3.1198398483918393) q[16];
rz(-0.6946593160188267) q[16];
ry(-0.09702216827625469) q[17];
rz(1.4436439534583094) q[17];
ry(2.2737589749364) q[18];
rz(1.7724136651082076) q[18];
ry(3.0469182882238504) q[19];
rz(2.895374483945031) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.845175997156874) q[0];
rz(-1.81178812565615) q[0];
ry(-1.4547639745422838) q[1];
rz(0.8921314294066621) q[1];
ry(2.6712491096908586) q[2];
rz(2.56689297137809) q[2];
ry(-0.014131878549800803) q[3];
rz(2.9268178297761964) q[3];
ry(1.3045152387361991) q[4];
rz(2.697296262986048) q[4];
ry(3.0423373171553516) q[5];
rz(-2.1401523510240517) q[5];
ry(1.5415713746477446) q[6];
rz(-2.9571620733122135) q[6];
ry(3.0402233855661565) q[7];
rz(-1.0280254949230851) q[7];
ry(-0.05867522577344413) q[8];
rz(2.6338902909772455) q[8];
ry(-2.0773477310647515) q[9];
rz(-0.2643922419899268) q[9];
ry(1.5737519399805553) q[10];
rz(-2.0040989255852657) q[10];
ry(1.0601211945382156) q[11];
rz(-0.24767134656534218) q[11];
ry(0.6711733599168266) q[12];
rz(2.9535521588910565) q[12];
ry(-0.5356957440004368) q[13];
rz(-2.726568833945887) q[13];
ry(-3.0551364420680645) q[14];
rz(-0.6152814372632689) q[14];
ry(2.0932990546726584) q[15];
rz(-0.5041988053162917) q[15];
ry(-0.42862513395352586) q[16];
rz(-2.895681539879366) q[16];
ry(0.07578350740412126) q[17];
rz(-0.3708870172950542) q[17];
ry(1.8814016071536592) q[18];
rz(1.5689725161033365) q[18];
ry(3.0655576212646776) q[19];
rz(-1.4940187799900793) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.0979066304676475) q[0];
rz(0.8953663029101111) q[0];
ry(2.166082710426125) q[1];
rz(0.4168093883672963) q[1];
ry(2.3457596658287607) q[2];
rz(1.481546886491459) q[2];
ry(-0.9442608100162957) q[3];
rz(-2.797959455650955) q[3];
ry(2.3894197606416188) q[4];
rz(1.86621172645508) q[4];
ry(3.051892871612081) q[5];
rz(-1.734576575938493) q[5];
ry(-1.861754062832936) q[6];
rz(-2.654478841946567) q[6];
ry(0.28791991850013404) q[7];
rz(-0.645446201626388) q[7];
ry(1.5885121966658622) q[8];
rz(1.799830321903225) q[8];
ry(-1.9341339699932996) q[9];
rz(-2.468749997906211) q[9];
ry(0.38139118777752634) q[10];
rz(-2.4467788904450405) q[10];
ry(2.464484893624391) q[11];
rz(-2.901521264357765) q[11];
ry(2.7906838048717546) q[12];
rz(0.44760352715787466) q[12];
ry(-0.2508393972757732) q[13];
rz(1.2497756953611014) q[13];
ry(1.5311943560406251) q[14];
rz(2.811359282248061) q[14];
ry(-2.3876476533803066) q[15];
rz(2.821815116294) q[15];
ry(-1.05622822438266) q[16];
rz(2.903942285268172) q[16];
ry(2.5190781841313568) q[17];
rz(-3.1044535599810774) q[17];
ry(-2.736389921885166) q[18];
rz(0.7910008210605657) q[18];
ry(-2.7144881453821688) q[19];
rz(2.598875801238316) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(2.6506969469287873) q[0];
rz(-2.2477378864743187) q[0];
ry(-0.6572151526778374) q[1];
rz(-2.4665538267124765) q[1];
ry(0.5693882507454261) q[2];
rz(-1.344056066237294) q[2];
ry(0.06230427912800625) q[3];
rz(-1.6158787941046926) q[3];
ry(0.2849480527294141) q[4];
rz(-1.425116562493939) q[4];
ry(-0.9497111461723999) q[5];
rz(3.0650962037662017) q[5];
ry(1.0582946703422034) q[6];
rz(2.379515057896391) q[6];
ry(-1.0164994460792878) q[7];
rz(-2.8403556495420195) q[7];
ry(-0.24528028968126048) q[8];
rz(2.9596285407081995) q[8];
ry(0.3638158375639691) q[9];
rz(-1.3271519137656838) q[9];
ry(0.4049349442250039) q[10];
rz(0.21692946764496043) q[10];
ry(-0.2317169638010794) q[11];
rz(0.3934915200580208) q[11];
ry(-0.6504116696985613) q[12];
rz(-0.9317409334985003) q[12];
ry(-0.4340154793016733) q[13];
rz(1.554609620400747) q[13];
ry(2.566809436469189) q[14];
rz(0.6707573275642043) q[14];
ry(-1.5495378650111853) q[15];
rz(-0.16035016938874339) q[15];
ry(0.17564416019223372) q[16];
rz(-0.3388877037740713) q[16];
ry(-2.387893198917221) q[17];
rz(0.8342651739548032) q[17];
ry(2.5968615615680126) q[18];
rz(2.297431038401927) q[18];
ry(-3.006965351371538) q[19];
rz(-0.537870707623653) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-1.5530917828314195) q[0];
rz(0.46694920965518527) q[0];
ry(-0.9199576356576279) q[1];
rz(-2.642344555388843) q[1];
ry(-2.1709490111041747) q[2];
rz(3.136889703901723) q[2];
ry(-1.0300235229488575) q[3];
rz(-1.2938105206999477) q[3];
ry(0.032595647886004775) q[4];
rz(1.7188133134445227) q[4];
ry(2.910743038501187) q[5];
rz(-0.10315476817206283) q[5];
ry(-0.249467783509988) q[6];
rz(-0.09441318552998904) q[6];
ry(-2.6598161009657626) q[7];
rz(0.3834972111794146) q[7];
ry(2.4436394506519923) q[8];
rz(-1.089113121415827) q[8];
ry(-0.7834177988773128) q[9];
rz(-0.8581593601301047) q[9];
ry(0.7662257015223464) q[10];
rz(2.0531736414076054) q[10];
ry(0.3156615535378638) q[11];
rz(3.0086248484176616) q[11];
ry(-2.65873031425014) q[12];
rz(1.1089097169819575) q[12];
ry(2.609272618270472) q[13];
rz(0.3719435087646463) q[13];
ry(-3.047661673708355) q[14];
rz(-1.9733694509220743) q[14];
ry(-2.333346921555977) q[15];
rz(2.158128926841111) q[15];
ry(-1.5241360233059678) q[16];
rz(-2.723364546864072) q[16];
ry(2.466876653262973) q[17];
rz(-2.3141905672229384) q[17];
ry(2.2847315379853566) q[18];
rz(0.5343883972950492) q[18];
ry(-0.3298469004619138) q[19];
rz(-2.784113044812402) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-0.8072306925043065) q[0];
rz(-0.417943918990277) q[0];
ry(0.3863738643811505) q[1];
rz(-1.2319695678613014) q[1];
ry(0.44649842968702) q[2];
rz(-2.493049203544442) q[2];
ry(0.1278026366025432) q[3];
rz(2.192499167562832) q[3];
ry(2.925662056556037) q[4];
rz(-0.6179914284864136) q[4];
ry(2.113107815182941) q[5];
rz(-2.361446330129598) q[5];
ry(-0.7457298104971528) q[6];
rz(-3.0461405366426018) q[6];
ry(-2.1477208504748093) q[7];
rz(2.8872594270343392) q[7];
ry(-0.37476010096647233) q[8];
rz(-1.3634752267474957) q[8];
ry(0.06322141199414165) q[9];
rz(2.7224125768387277) q[9];
ry(0.31480100756735574) q[10];
rz(-0.6468313586300091) q[10];
ry(0.7070975191800724) q[11];
rz(3.008785107617096) q[11];
ry(3.0735243523293927) q[12];
rz(2.706472756104446) q[12];
ry(2.8709200153529792) q[13];
rz(1.4885481876976443) q[13];
ry(1.68989158413841) q[14];
rz(-0.2846576528427429) q[14];
ry(0.08105526990361511) q[15];
rz(3.0983310414162424) q[15];
ry(-3.0492314800110862) q[16];
rz(1.6725605411055477) q[16];
ry(1.706112137775539) q[17];
rz(-0.06705906445267562) q[17];
ry(0.5733563588813305) q[18];
rz(-1.3894893151814411) q[18];
ry(-0.6205954872718777) q[19];
rz(-2.3963606116423737) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(-2.5682561894395124) q[0];
rz(-1.7020811408363272) q[0];
ry(-0.8106978230103452) q[1];
rz(-2.0276149853737118) q[1];
ry(-2.1707854966379188) q[2];
rz(2.917681206690851) q[2];
ry(0.4693914893074124) q[3];
rz(3.051408436119905) q[3];
ry(1.8734374818493484) q[4];
rz(2.171591769191407) q[4];
ry(2.8832618859649415) q[5];
rz(-2.747720170755092) q[5];
ry(1.5522181977334686) q[6];
rz(-1.1493208952503524) q[6];
ry(0.42152576568324895) q[7];
rz(-0.03403273077910374) q[7];
ry(2.897149771904174) q[8];
rz(-1.6216224657813463) q[8];
ry(3.0936862545951893) q[9];
rz(1.2146428922721684) q[9];
ry(-0.32418933232295954) q[10];
rz(1.5414197244050385) q[10];
ry(-1.4351454948405593) q[11];
rz(2.957913139229344) q[11];
ry(-0.14080513294212252) q[12];
rz(-2.3300317059016002) q[12];
ry(3.1312639473887347) q[13];
rz(1.2731166059003582) q[13];
ry(-2.8825213570684993) q[14];
rz(0.05571829115204441) q[14];
ry(-0.04761556599342165) q[15];
rz(-0.7746421878655179) q[15];
ry(-1.6786281163308012) q[16];
rz(-0.6046070820404517) q[16];
ry(2.1182648282094276) q[17];
rz(1.6049233824724574) q[17];
ry(-1.604157542213502) q[18];
rz(3.107823083969051) q[18];
ry(1.9487185470681423) q[19];
rz(1.9795726419652149) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.9605135837185088) q[0];
rz(0.7689489910298345) q[0];
ry(0.6094070620679115) q[1];
rz(0.7860475432260617) q[1];
ry(3.0356865242228404) q[2];
rz(-1.9462039652596674) q[2];
ry(0.044828008791107266) q[3];
rz(-0.9361901183862762) q[3];
ry(-2.9943824123907974) q[4];
rz(-2.502082137635949) q[4];
ry(-0.3010270510160697) q[5];
rz(1.390418930393622) q[5];
ry(-0.30265618839106356) q[6];
rz(-1.4473059871329603) q[6];
ry(-0.6065602068684699) q[7];
rz(-1.8462484911332133) q[7];
ry(-0.47985675047189164) q[8];
rz(0.9665110718409684) q[8];
ry(0.23166498540583133) q[9];
rz(-3.009617494468786) q[9];
ry(-2.8073654396376306) q[10];
rz(0.13233737836446868) q[10];
ry(-0.6145314498284495) q[11];
rz(1.8703583998260003) q[11];
ry(3.045264759351738) q[12];
rz(-2.2217464893363448) q[12];
ry(2.7230396049620285) q[13];
rz(2.168748328538974) q[13];
ry(-1.188201271530709) q[14];
rz(0.7836101126798096) q[14];
ry(-2.9893852457304435) q[15];
rz(3.12548587053368) q[15];
ry(3.1371217906039863) q[16];
rz(0.6134219080044101) q[16];
ry(3.0746955559679927) q[17];
rz(-3.0989455004254154) q[17];
ry(-2.848969818248433) q[18];
rz(-1.6275085786022858) q[18];
ry(-1.4859485428564767) q[19];
rz(-1.611507476532137) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
cz q[15],q[16];
cz q[17],q[18];
ry(0.2731808824131272) q[0];
rz(1.4582594716429842) q[0];
ry(-1.284692429459074) q[1];
rz(-0.42135574213853033) q[1];
ry(-1.355126478158587) q[2];
rz(-0.9935731616676469) q[2];
ry(2.141705723883004) q[3];
rz(-1.8720760751493941) q[3];
ry(-1.670452952513638) q[4];
rz(2.7102705021396476) q[4];
ry(-1.813600812105439) q[5];
rz(2.616313097060577) q[5];
ry(-1.4978488407470572) q[6];
rz(-2.180511988814292) q[6];
ry(1.8456795775394095) q[7];
rz(1.3362197556768964) q[7];
ry(1.6580964115881507) q[8];
rz(2.442050035200746) q[8];
ry(0.6573508825110824) q[9];
rz(1.181666582113495) q[9];
ry(-2.415725313426916) q[10];
rz(1.413702972024028) q[10];
ry(-1.694717057439653) q[11];
rz(-1.2472584712263695) q[11];
ry(-2.936104927041855) q[12];
rz(0.5056724740482529) q[12];
ry(3.0155252259662357) q[13];
rz(-2.9733509159550064) q[13];
ry(-1.6423530283824814) q[14];
rz(0.04327717629360856) q[14];
ry(-1.3367794747824142) q[15];
rz(0.04718036082654731) q[15];
ry(-2.7946309420587494) q[16];
rz(-0.3029951544280731) q[16];
ry(1.5678148471258835) q[17];
rz(-2.4886266514262) q[17];
ry(1.6118488237035629) q[18];
rz(-1.7438178385865546) q[18];
ry(1.6071606558021174) q[19];
rz(-0.9284994113315905) q[19];