OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.042831249265914965) q[0];
ry(-0.04981785367548025) q[1];
cx q[0],q[1];
ry(-0.4678283307600708) q[0];
ry(-1.7639987131289823) q[1];
cx q[0],q[1];
ry(-0.23458009667478374) q[2];
ry(1.0267669718078212) q[3];
cx q[2],q[3];
ry(-2.8352179825776758) q[2];
ry(1.4955824485194287) q[3];
cx q[2],q[3];
ry(0.30194301868328677) q[4];
ry(-2.1865216899581976) q[5];
cx q[4],q[5];
ry(-1.1728002777566018) q[4];
ry(-2.956068595481588) q[5];
cx q[4],q[5];
ry(-2.30029756832662) q[6];
ry(-1.4670164617963135) q[7];
cx q[6],q[7];
ry(-1.9081076401293648) q[6];
ry(-3.0570740593002466) q[7];
cx q[6],q[7];
ry(-2.587382279298003) q[8];
ry(2.429534634365675) q[9];
cx q[8],q[9];
ry(-0.9530049926766916) q[8];
ry(-2.032579506466747) q[9];
cx q[8],q[9];
ry(1.7821974476386417) q[10];
ry(1.9353028181823524) q[11];
cx q[10],q[11];
ry(-1.9393034748287734) q[10];
ry(-2.78347304981504) q[11];
cx q[10],q[11];
ry(2.2516462936689727) q[1];
ry(-2.853418803731992) q[2];
cx q[1],q[2];
ry(-3.140265868656137) q[1];
ry(-1.7237786094147536) q[2];
cx q[1],q[2];
ry(-0.5250960466068578) q[3];
ry(0.7735912842686425) q[4];
cx q[3],q[4];
ry(-3.1373984796727017) q[3];
ry(1.890461137124957) q[4];
cx q[3],q[4];
ry(-1.1377357338873688) q[5];
ry(-0.0823872121907117) q[6];
cx q[5],q[6];
ry(-0.2570703801758878) q[5];
ry(-0.453768032419264) q[6];
cx q[5],q[6];
ry(3.006376168861911) q[7];
ry(-1.907157538249928) q[8];
cx q[7],q[8];
ry(1.2815075368038626) q[7];
ry(-0.33189711153812507) q[8];
cx q[7],q[8];
ry(-0.47968246519470986) q[9];
ry(-3.0009234473675725) q[10];
cx q[9],q[10];
ry(-1.7551505021220573) q[9];
ry(-3.1152984186460695) q[10];
cx q[9],q[10];
ry(-0.7722547798016253) q[0];
ry(-1.3059969052368405) q[1];
cx q[0],q[1];
ry(0.785802638158204) q[0];
ry(-0.6906274410673062) q[1];
cx q[0],q[1];
ry(-2.5870086653757998) q[2];
ry(1.9914894182208265) q[3];
cx q[2],q[3];
ry(-1.2563227830842418) q[2];
ry(-1.1048674262167524) q[3];
cx q[2],q[3];
ry(-3.13631658321476) q[4];
ry(-0.43478123019094406) q[5];
cx q[4],q[5];
ry(-2.2332440409426173) q[4];
ry(1.688156318950869) q[5];
cx q[4],q[5];
ry(0.011653772351004399) q[6];
ry(-0.17683436695008584) q[7];
cx q[6],q[7];
ry(-3.130059726952664) q[6];
ry(-0.11203652197367703) q[7];
cx q[6],q[7];
ry(1.9119063124698767) q[8];
ry(1.2070492579827916) q[9];
cx q[8],q[9];
ry(3.121059800894805) q[8];
ry(1.1566131787487066) q[9];
cx q[8],q[9];
ry(-1.6875724872576747) q[10];
ry(-0.0916208213747467) q[11];
cx q[10],q[11];
ry(-2.1705697632697962) q[10];
ry(-2.4391725052918187) q[11];
cx q[10],q[11];
ry(1.7559101473290646) q[1];
ry(-3.1070874098896772) q[2];
cx q[1],q[2];
ry(-1.6774903259263318) q[1];
ry(-2.2801803253484234) q[2];
cx q[1],q[2];
ry(0.7177879790477085) q[3];
ry(-2.5964519604538574) q[4];
cx q[3],q[4];
ry(3.1388339005229304) q[3];
ry(-0.2136777936788592) q[4];
cx q[3],q[4];
ry(3.129321469591574) q[5];
ry(2.9952123697319526) q[6];
cx q[5],q[6];
ry(2.1267788450589866) q[5];
ry(-2.2994817099139495) q[6];
cx q[5],q[6];
ry(1.281911540010058) q[7];
ry(1.46751124655977) q[8];
cx q[7],q[8];
ry(1.2879515520969222) q[7];
ry(1.090196909537742) q[8];
cx q[7],q[8];
ry(-1.4482650256898866) q[9];
ry(-2.3392350883221864) q[10];
cx q[9],q[10];
ry(0.027949980381059933) q[9];
ry(2.193047117531777) q[10];
cx q[9],q[10];
ry(-0.3841079077517664) q[0];
ry(1.594155391638773) q[1];
cx q[0],q[1];
ry(2.6243795871458144) q[0];
ry(-1.3708519428539576) q[1];
cx q[0],q[1];
ry(0.9148075320877629) q[2];
ry(-0.8339842406127493) q[3];
cx q[2],q[3];
ry(1.7926760884375472) q[2];
ry(1.9880569096054694) q[3];
cx q[2],q[3];
ry(-0.9469922276316953) q[4];
ry(-1.9645204611683822) q[5];
cx q[4],q[5];
ry(-3.0518578666096547) q[4];
ry(-2.1796700126940176) q[5];
cx q[4],q[5];
ry(2.5090510364948004) q[6];
ry(-2.355837016070173) q[7];
cx q[6],q[7];
ry(-0.04097733307364937) q[6];
ry(-3.120579793404975) q[7];
cx q[6],q[7];
ry(1.1806401987051567) q[8];
ry(0.047570874865701995) q[9];
cx q[8],q[9];
ry(2.458223291787333) q[8];
ry(1.6369649827672155) q[9];
cx q[8],q[9];
ry(2.2315855362722052) q[10];
ry(-1.9951856595709287) q[11];
cx q[10],q[11];
ry(-0.17487008718971359) q[10];
ry(1.5159223195851963) q[11];
cx q[10],q[11];
ry(1.296521210565297) q[1];
ry(1.3645656565279491) q[2];
cx q[1],q[2];
ry(2.7459013401920416) q[1];
ry(-1.2936407988680794) q[2];
cx q[1],q[2];
ry(0.683473182290827) q[3];
ry(-0.8042356234108192) q[4];
cx q[3],q[4];
ry(-0.23302949245019638) q[3];
ry(1.3710462924706186) q[4];
cx q[3],q[4];
ry(2.908527201733305) q[5];
ry(-2.0529368879049485) q[6];
cx q[5],q[6];
ry(0.435236766955061) q[5];
ry(0.9715421572017229) q[6];
cx q[5],q[6];
ry(2.0934043552692376) q[7];
ry(2.738642251786248) q[8];
cx q[7],q[8];
ry(3.1313595890554273) q[7];
ry(-0.14419776887706082) q[8];
cx q[7],q[8];
ry(0.6373878630137476) q[9];
ry(-0.6177535762218697) q[10];
cx q[9],q[10];
ry(-2.4191208925636034) q[9];
ry(2.858554629428455) q[10];
cx q[9],q[10];
ry(-1.794187069476003) q[0];
ry(1.4483836330645097) q[1];
cx q[0],q[1];
ry(2.036447645010632) q[0];
ry(2.0155487854498837) q[1];
cx q[0],q[1];
ry(1.9286413366042465) q[2];
ry(2.1620118178370094) q[3];
cx q[2],q[3];
ry(0.9875029049159023) q[2];
ry(-2.2108578543940114) q[3];
cx q[2],q[3];
ry(-2.7283556188696125) q[4];
ry(1.164758692190022) q[5];
cx q[4],q[5];
ry(0.7296593674352421) q[4];
ry(0.042695571045861413) q[5];
cx q[4],q[5];
ry(-2.9295984651115723) q[6];
ry(-0.4465732578423115) q[7];
cx q[6],q[7];
ry(2.4340925265333935) q[6];
ry(-0.85689552257322) q[7];
cx q[6],q[7];
ry(-1.6802238256907627) q[8];
ry(2.954726096049854) q[9];
cx q[8],q[9];
ry(-2.589741552731033) q[8];
ry(-2.444354365195655) q[9];
cx q[8],q[9];
ry(1.0594287521209076) q[10];
ry(-1.2434707468983444) q[11];
cx q[10],q[11];
ry(1.271582468178587) q[10];
ry(-3.049750013350083) q[11];
cx q[10],q[11];
ry(2.8802514962140457) q[1];
ry(2.5111161435618063) q[2];
cx q[1],q[2];
ry(-0.4063506013965047) q[1];
ry(-2.0864877784970997) q[2];
cx q[1],q[2];
ry(0.5445120112899495) q[3];
ry(2.256591652157711) q[4];
cx q[3],q[4];
ry(-0.14152106946907167) q[3];
ry(2.512897907668637) q[4];
cx q[3],q[4];
ry(-2.30904384771305) q[5];
ry(0.3647198152906217) q[6];
cx q[5],q[6];
ry(-0.00428406035387304) q[5];
ry(-3.1249267810557675) q[6];
cx q[5],q[6];
ry(0.6646908723484799) q[7];
ry(-2.162662739063877) q[8];
cx q[7],q[8];
ry(-0.03990873036170317) q[7];
ry(3.1296562229969607) q[8];
cx q[7],q[8];
ry(-2.7183142746251523) q[9];
ry(-2.076286077498282) q[10];
cx q[9],q[10];
ry(0.25086881809079836) q[9];
ry(2.9566140949429425) q[10];
cx q[9],q[10];
ry(-2.8576238538149408) q[0];
ry(-2.005548196855612) q[1];
cx q[0],q[1];
ry(-1.71440168411252) q[0];
ry(1.617390144100156) q[1];
cx q[0],q[1];
ry(-2.3612270313446) q[2];
ry(0.17457015938326895) q[3];
cx q[2],q[3];
ry(1.8301380607840005) q[2];
ry(-1.3301059039663734) q[3];
cx q[2],q[3];
ry(-1.673490150657686) q[4];
ry(1.6096188492944654) q[5];
cx q[4],q[5];
ry(-0.7666494573445403) q[4];
ry(3.1396795745314368) q[5];
cx q[4],q[5];
ry(-0.35777639888503376) q[6];
ry(-1.7240349491500875) q[7];
cx q[6],q[7];
ry(0.5821952647579568) q[6];
ry(0.8706463163881555) q[7];
cx q[6],q[7];
ry(-2.347869108721845) q[8];
ry(2.667968573142792) q[9];
cx q[8],q[9];
ry(2.4434334732258867) q[8];
ry(1.8530267790671866) q[9];
cx q[8],q[9];
ry(-2.589354361170384) q[10];
ry(-2.1606811055821646) q[11];
cx q[10],q[11];
ry(-2.457386223278002) q[10];
ry(-1.0074105503710724) q[11];
cx q[10],q[11];
ry(1.3412733315495915) q[1];
ry(2.1497216610446115) q[2];
cx q[1],q[2];
ry(-2.142909345318205) q[1];
ry(-1.8044898479716265) q[2];
cx q[1],q[2];
ry(-2.538401064516434) q[3];
ry(2.6648930506469117) q[4];
cx q[3],q[4];
ry(2.962940068622721) q[3];
ry(0.7360919061813701) q[4];
cx q[3],q[4];
ry(2.2925784125002426) q[5];
ry(1.9501341657891196) q[6];
cx q[5],q[6];
ry(0.14811637987837678) q[5];
ry(-2.102717078493029) q[6];
cx q[5],q[6];
ry(-0.00909958927068009) q[7];
ry(-3.127440663816165) q[8];
cx q[7],q[8];
ry(-3.138957939673725) q[7];
ry(-1.7491755220010106) q[8];
cx q[7],q[8];
ry(-0.7498550492106579) q[9];
ry(-1.2886033849250182) q[10];
cx q[9],q[10];
ry(2.4741730363215315) q[9];
ry(-1.431888867256373) q[10];
cx q[9],q[10];
ry(2.020375202619883) q[0];
ry(1.49237612818827) q[1];
cx q[0],q[1];
ry(-2.2458032626411617) q[0];
ry(-0.2917664360320771) q[1];
cx q[0],q[1];
ry(1.6839373211199664) q[2];
ry(0.16561232023953257) q[3];
cx q[2],q[3];
ry(0.2663583264384233) q[2];
ry(-3.0697119036204312) q[3];
cx q[2],q[3];
ry(1.3454282115054346) q[4];
ry(-1.9748800604416221) q[5];
cx q[4],q[5];
ry(1.8056037331038164) q[4];
ry(3.0822049044057436) q[5];
cx q[4],q[5];
ry(-0.2682181719267869) q[6];
ry(1.1238466388052917) q[7];
cx q[6],q[7];
ry(1.8323042018806597) q[6];
ry(1.2476302980871363) q[7];
cx q[6],q[7];
ry(-0.40419649555007514) q[8];
ry(-1.0308663185625049) q[9];
cx q[8],q[9];
ry(2.3145752884970507) q[8];
ry(0.8310183633725972) q[9];
cx q[8],q[9];
ry(-2.815167241632971) q[10];
ry(-3.0213128530801168) q[11];
cx q[10],q[11];
ry(0.20178803208473983) q[10];
ry(-0.024270414441019742) q[11];
cx q[10],q[11];
ry(2.7576340249484357) q[1];
ry(-1.6408299586083643) q[2];
cx q[1],q[2];
ry(1.6235640144799683) q[1];
ry(-0.32755493232033395) q[2];
cx q[1],q[2];
ry(-0.6905718135839267) q[3];
ry(1.358380225923426) q[4];
cx q[3],q[4];
ry(-0.06626235819390408) q[3];
ry(-1.7983147158515074) q[4];
cx q[3],q[4];
ry(-0.17543280435755104) q[5];
ry(-2.5868193471307914) q[6];
cx q[5],q[6];
ry(-2.734609058780373) q[5];
ry(2.1425564227631706) q[6];
cx q[5],q[6];
ry(-1.7365429219169402) q[7];
ry(-1.5392153029181896) q[8];
cx q[7],q[8];
ry(3.1323226670857207) q[7];
ry(-3.0204466876271825) q[8];
cx q[7],q[8];
ry(2.2494683952033174) q[9];
ry(0.5028143260821532) q[10];
cx q[9],q[10];
ry(-1.8947074741310557) q[9];
ry(1.8719205669699872) q[10];
cx q[9],q[10];
ry(1.0899500726402627) q[0];
ry(-0.7923918587686174) q[1];
cx q[0],q[1];
ry(2.922700692407559) q[0];
ry(1.379985827160266) q[1];
cx q[0],q[1];
ry(-2.7332937025009736) q[2];
ry(-2.098815603274514) q[3];
cx q[2],q[3];
ry(0.1609556534202016) q[2];
ry(0.7041004036374039) q[3];
cx q[2],q[3];
ry(-1.3134584995287453) q[4];
ry(0.655594587386504) q[5];
cx q[4],q[5];
ry(-1.2310222640173356) q[4];
ry(1.1191933502100988) q[5];
cx q[4],q[5];
ry(0.9937825665748415) q[6];
ry(-1.876363610559105) q[7];
cx q[6],q[7];
ry(-2.385943152107082) q[6];
ry(-0.011566579897087318) q[7];
cx q[6],q[7];
ry(0.46029023083315135) q[8];
ry(-1.1403113928656516) q[9];
cx q[8],q[9];
ry(0.45710482104608996) q[8];
ry(2.8119381110584416) q[9];
cx q[8],q[9];
ry(-2.673582770436701) q[10];
ry(0.6599806276682774) q[11];
cx q[10],q[11];
ry(1.8297095518957611) q[10];
ry(-0.4232072255210691) q[11];
cx q[10],q[11];
ry(2.2619733220174933) q[1];
ry(-1.142385618030026) q[2];
cx q[1],q[2];
ry(1.8960772003456465) q[1];
ry(-1.0786849606970752) q[2];
cx q[1],q[2];
ry(2.7288850106340585) q[3];
ry(-2.487250715927527) q[4];
cx q[3],q[4];
ry(3.1386744418176784) q[3];
ry(-3.0355340936925534) q[4];
cx q[3],q[4];
ry(2.7164212439871) q[5];
ry(-1.764350504723472) q[6];
cx q[5],q[6];
ry(-1.352530541973454) q[5];
ry(0.05014720298048699) q[6];
cx q[5],q[6];
ry(2.8463429090832997) q[7];
ry(-2.35425002194546) q[8];
cx q[7],q[8];
ry(-0.21316787050767783) q[7];
ry(1.9693640505692818) q[8];
cx q[7],q[8];
ry(-0.13254799518194962) q[9];
ry(1.5202843290813222) q[10];
cx q[9],q[10];
ry(0.8089332438932023) q[9];
ry(-0.27377169112329724) q[10];
cx q[9],q[10];
ry(-0.9880518155320708) q[0];
ry(-1.2942990735240771) q[1];
cx q[0],q[1];
ry(-2.7922890157333824) q[0];
ry(-0.15017295448040135) q[1];
cx q[0],q[1];
ry(-0.7906933523358193) q[2];
ry(-0.9368702862982214) q[3];
cx q[2],q[3];
ry(-2.6617437668455177) q[2];
ry(-2.5117227523235157) q[3];
cx q[2],q[3];
ry(0.6583053810697088) q[4];
ry(2.1718116229512736) q[5];
cx q[4],q[5];
ry(-1.710489491701372) q[4];
ry(-2.034312923599746) q[5];
cx q[4],q[5];
ry(-1.0503898739277702) q[6];
ry(0.5073480523561688) q[7];
cx q[6],q[7];
ry(0.0015127040649365359) q[6];
ry(-0.0005676837803747503) q[7];
cx q[6],q[7];
ry(-3.018802678063316) q[8];
ry(1.3246725149639278) q[9];
cx q[8],q[9];
ry(2.316286811520465) q[8];
ry(0.3453178458900091) q[9];
cx q[8],q[9];
ry(1.8240148656925825) q[10];
ry(1.0201473692538423) q[11];
cx q[10],q[11];
ry(1.5396078550658319) q[10];
ry(2.852692487744186) q[11];
cx q[10],q[11];
ry(-0.008584990524377112) q[1];
ry(-2.59359199572571) q[2];
cx q[1],q[2];
ry(-2.4150366723588634) q[1];
ry(-0.2595022507199207) q[2];
cx q[1],q[2];
ry(1.3679050511533042) q[3];
ry(-0.8694651121732461) q[4];
cx q[3],q[4];
ry(-0.06549246241701212) q[3];
ry(2.2937086665827304) q[4];
cx q[3],q[4];
ry(-0.28756316251805836) q[5];
ry(1.0555581057867145) q[6];
cx q[5],q[6];
ry(-0.6572595442496444) q[5];
ry(-0.9382505596085728) q[6];
cx q[5],q[6];
ry(-0.17086558606596913) q[7];
ry(2.1903496312791955) q[8];
cx q[7],q[8];
ry(-2.965382336357298) q[7];
ry(-2.2271970128844556) q[8];
cx q[7],q[8];
ry(-2.303658838027585) q[9];
ry(-2.193398213382312) q[10];
cx q[9],q[10];
ry(-0.7406374407096585) q[9];
ry(-2.014777342825053) q[10];
cx q[9],q[10];
ry(-1.7700179253541135) q[0];
ry(-1.516530278884594) q[1];
cx q[0],q[1];
ry(-1.3916617715634008) q[0];
ry(1.9238601303984728) q[1];
cx q[0],q[1];
ry(-0.3032948466660059) q[2];
ry(1.3094653496228634) q[3];
cx q[2],q[3];
ry(-0.04535360573786473) q[2];
ry(-0.10035981354302681) q[3];
cx q[2],q[3];
ry(-0.34603578308171645) q[4];
ry(-1.7760127209676795) q[5];
cx q[4],q[5];
ry(0.05459232986673345) q[4];
ry(0.01619454701743268) q[5];
cx q[4],q[5];
ry(2.1657268947224804) q[6];
ry(1.6318752634527902) q[7];
cx q[6],q[7];
ry(0.02611338179767131) q[6];
ry(-0.002733715728814533) q[7];
cx q[6],q[7];
ry(-1.7390733561895608) q[8];
ry(1.9843900329124853) q[9];
cx q[8],q[9];
ry(2.356058864362239) q[8];
ry(2.98008367674443) q[9];
cx q[8],q[9];
ry(-1.2732358125092924) q[10];
ry(1.7155625404181958) q[11];
cx q[10],q[11];
ry(-0.7816658308523898) q[10];
ry(0.5508558347383189) q[11];
cx q[10],q[11];
ry(2.4957009009122353) q[1];
ry(-0.6374448102809183) q[2];
cx q[1],q[2];
ry(-2.089121023744159) q[1];
ry(0.5803090292307685) q[2];
cx q[1],q[2];
ry(-2.456880863628961) q[3];
ry(-0.33722351015287355) q[4];
cx q[3],q[4];
ry(3.0999326367995814) q[3];
ry(-0.6900791703512601) q[4];
cx q[3],q[4];
ry(2.3201375556575434) q[5];
ry(2.0082593004526723) q[6];
cx q[5],q[6];
ry(2.006123317187925) q[5];
ry(-0.47412728914976326) q[6];
cx q[5],q[6];
ry(1.9083332435814073) q[7];
ry(-2.088986760597831) q[8];
cx q[7],q[8];
ry(0.1488818948336544) q[7];
ry(3.000153538797587) q[8];
cx q[7],q[8];
ry(0.8555690160271539) q[9];
ry(2.422423277911845) q[10];
cx q[9],q[10];
ry(-1.7444041737991725) q[9];
ry(2.8940171767568534) q[10];
cx q[9],q[10];
ry(0.7541646568362951) q[0];
ry(-2.289965265646378) q[1];
cx q[0],q[1];
ry(-0.48604341015928504) q[0];
ry(0.08056758526627039) q[1];
cx q[0],q[1];
ry(0.11341959463525164) q[2];
ry(-1.9462308618220208) q[3];
cx q[2],q[3];
ry(-0.19299270485138473) q[2];
ry(1.5865094993474447) q[3];
cx q[2],q[3];
ry(-1.1779602822786153) q[4];
ry(0.5275678073505229) q[5];
cx q[4],q[5];
ry(2.1973621091465656) q[4];
ry(0.2897351900329683) q[5];
cx q[4],q[5];
ry(-2.8367610539550907) q[6];
ry(2.08559886690428) q[7];
cx q[6],q[7];
ry(-0.014183826056365272) q[6];
ry(-0.03302140806859821) q[7];
cx q[6],q[7];
ry(-1.903044491190437) q[8];
ry(-1.5512557670804616) q[9];
cx q[8],q[9];
ry(-2.747074030430589) q[8];
ry(-2.827409627990134) q[9];
cx q[8],q[9];
ry(-1.350911731348153) q[10];
ry(-1.5618430509220298) q[11];
cx q[10],q[11];
ry(-3.080409756189407) q[10];
ry(-1.693117187711258) q[11];
cx q[10],q[11];
ry(-2.078905628801587) q[1];
ry(1.0256357501049145) q[2];
cx q[1],q[2];
ry(1.5060261663010959) q[1];
ry(-2.3307360148972642) q[2];
cx q[1],q[2];
ry(-1.8120079110610674) q[3];
ry(-2.3823716205003205) q[4];
cx q[3],q[4];
ry(0.07594267852589075) q[3];
ry(0.3593395649944648) q[4];
cx q[3],q[4];
ry(1.3292476241755091) q[5];
ry(-1.3907634744759967) q[6];
cx q[5],q[6];
ry(-0.6024646451350573) q[5];
ry(-3.0288926535959857) q[6];
cx q[5],q[6];
ry(2.124896186131701) q[7];
ry(-1.3011484824200057) q[8];
cx q[7],q[8];
ry(2.8934825317781834) q[7];
ry(-1.3415408754750233) q[8];
cx q[7],q[8];
ry(-2.134307785457814) q[9];
ry(-0.995252955948554) q[10];
cx q[9],q[10];
ry(-2.2539952745097134) q[9];
ry(-2.737636553696851) q[10];
cx q[9],q[10];
ry(-1.3350813307279) q[0];
ry(-1.3303902663112375) q[1];
cx q[0],q[1];
ry(2.801278007351672) q[0];
ry(-2.48445567529706) q[1];
cx q[0],q[1];
ry(3.0250657631688744) q[2];
ry(-0.10586683530151486) q[3];
cx q[2],q[3];
ry(2.3810856761416095) q[2];
ry(-0.2409155429679224) q[3];
cx q[2],q[3];
ry(0.21525930135734453) q[4];
ry(0.8591269110955316) q[5];
cx q[4],q[5];
ry(-1.881281509291865) q[4];
ry(-3.105921277769213) q[5];
cx q[4],q[5];
ry(3.038626033692238) q[6];
ry(-0.8158955415254293) q[7];
cx q[6],q[7];
ry(-0.0009040937679474226) q[6];
ry(3.1414266809978475) q[7];
cx q[6],q[7];
ry(1.962842160422123) q[8];
ry(0.532038661143898) q[9];
cx q[8],q[9];
ry(2.2444526567340954) q[8];
ry(-0.18488252828427412) q[9];
cx q[8],q[9];
ry(-2.9334091148903956) q[10];
ry(-2.4709533846412453) q[11];
cx q[10],q[11];
ry(-2.1859185722749164) q[10];
ry(1.7282180111663281) q[11];
cx q[10],q[11];
ry(-2.152680130882726) q[1];
ry(-2.625606411048346) q[2];
cx q[1],q[2];
ry(-0.5102640909502174) q[1];
ry(-0.8750226842498591) q[2];
cx q[1],q[2];
ry(-0.3779455008636958) q[3];
ry(-0.8359915025028409) q[4];
cx q[3],q[4];
ry(0.7719850603150669) q[3];
ry(2.737596862610187) q[4];
cx q[3],q[4];
ry(2.8759643724991824) q[5];
ry(3.1170249621378763) q[6];
cx q[5],q[6];
ry(1.3723792674530726) q[5];
ry(-2.952399631223077) q[6];
cx q[5],q[6];
ry(2.792446123905372) q[7];
ry(-1.4265225132904433) q[8];
cx q[7],q[8];
ry(-2.0884861097256566) q[7];
ry(2.5611660860309615) q[8];
cx q[7],q[8];
ry(1.1331372302272422) q[9];
ry(0.31996776627851187) q[10];
cx q[9],q[10];
ry(0.03343857949080409) q[9];
ry(0.1730736392624017) q[10];
cx q[9],q[10];
ry(2.4593515353446835) q[0];
ry(1.2690097162760132) q[1];
cx q[0],q[1];
ry(2.66836993365487) q[0];
ry(1.9848631486926667) q[1];
cx q[0],q[1];
ry(-2.9268583270232376) q[2];
ry(-3.0388971661076996) q[3];
cx q[2],q[3];
ry(-0.006384620589877386) q[2];
ry(0.925291775720761) q[3];
cx q[2],q[3];
ry(0.8341955562747696) q[4];
ry(0.5733101078294272) q[5];
cx q[4],q[5];
ry(3.0501704493952477) q[4];
ry(3.1195904103842267) q[5];
cx q[4],q[5];
ry(2.318332049193858) q[6];
ry(-2.3706150931654326) q[7];
cx q[6],q[7];
ry(-2.515376298141296) q[6];
ry(-2.0572282012681793) q[7];
cx q[6],q[7];
ry(1.378186872916893) q[8];
ry(0.8394035756454787) q[9];
cx q[8],q[9];
ry(-1.0118953554720882) q[8];
ry(-0.20311362298660762) q[9];
cx q[8],q[9];
ry(-2.748622521467536) q[10];
ry(-0.1605240436885511) q[11];
cx q[10],q[11];
ry(1.5333032128412878) q[10];
ry(-1.1473263317959823) q[11];
cx q[10],q[11];
ry(1.9891436134696594) q[1];
ry(0.08030950574964102) q[2];
cx q[1],q[2];
ry(-0.15905400850800666) q[1];
ry(-1.9447417727760108) q[2];
cx q[1],q[2];
ry(2.7716933563196653) q[3];
ry(0.09018053236445757) q[4];
cx q[3],q[4];
ry(0.843910852033038) q[3];
ry(0.35624790094515735) q[4];
cx q[3],q[4];
ry(2.795675368593816) q[5];
ry(0.634257472615609) q[6];
cx q[5],q[6];
ry(-0.0028277629240215543) q[5];
ry(-0.0032925905141432286) q[6];
cx q[5],q[6];
ry(2.9068874993728095) q[7];
ry(1.8424175006422965) q[8];
cx q[7],q[8];
ry(0.5899324519540378) q[7];
ry(3.1224891646818227) q[8];
cx q[7],q[8];
ry(-1.89792732898939) q[9];
ry(-2.984540917337121) q[10];
cx q[9],q[10];
ry(2.539375218388348) q[9];
ry(1.841085273523795) q[10];
cx q[9],q[10];
ry(2.499129632430692) q[0];
ry(-0.2120638642218505) q[1];
cx q[0],q[1];
ry(1.108899430346981) q[0];
ry(1.7926679087155872) q[1];
cx q[0],q[1];
ry(-1.7074049365109158) q[2];
ry(2.82618328397757) q[3];
cx q[2],q[3];
ry(-0.18632511936319887) q[2];
ry(-0.5120304271064438) q[3];
cx q[2],q[3];
ry(-0.9103792980757701) q[4];
ry(-2.9962850315829925) q[5];
cx q[4],q[5];
ry(0.21055910252910606) q[4];
ry(0.5610832601377984) q[5];
cx q[4],q[5];
ry(-0.6230425616587647) q[6];
ry(1.6087375574972347) q[7];
cx q[6],q[7];
ry(-3.016267067871296) q[6];
ry(-2.02645767505886) q[7];
cx q[6],q[7];
ry(-1.0416924590028833) q[8];
ry(2.2379749950370362) q[9];
cx q[8],q[9];
ry(1.1497244367740003) q[8];
ry(0.2012493759293699) q[9];
cx q[8],q[9];
ry(-0.3980544571800211) q[10];
ry(-2.5978746907533945) q[11];
cx q[10],q[11];
ry(0.47043942081809204) q[10];
ry(-0.24716662619196378) q[11];
cx q[10],q[11];
ry(-0.27945447202674334) q[1];
ry(-0.5724230927823423) q[2];
cx q[1],q[2];
ry(0.2601536644100637) q[1];
ry(1.5261320324248726) q[2];
cx q[1],q[2];
ry(-2.5412516343089524) q[3];
ry(-0.19411711367133272) q[4];
cx q[3],q[4];
ry(-0.36313158340806395) q[3];
ry(-1.1110088660889448) q[4];
cx q[3],q[4];
ry(-2.4481477732402723) q[5];
ry(1.2568485728572036) q[6];
cx q[5],q[6];
ry(-0.0247076268332667) q[5];
ry(0.2065530561200708) q[6];
cx q[5],q[6];
ry(-0.7139114434221819) q[7];
ry(3.001579533420682) q[8];
cx q[7],q[8];
ry(3.070677738345399) q[7];
ry(-2.5187060045413676) q[8];
cx q[7],q[8];
ry(-2.1579752747293686) q[9];
ry(-2.102311896517838) q[10];
cx q[9],q[10];
ry(-1.6566343271855184) q[9];
ry(0.6058622420987544) q[10];
cx q[9],q[10];
ry(1.5502387835682292) q[0];
ry(0.388677815361767) q[1];
cx q[0],q[1];
ry(-2.6970215742704733) q[0];
ry(-0.9322480015014697) q[1];
cx q[0],q[1];
ry(-1.8461170912254878) q[2];
ry(-0.8301186686401714) q[3];
cx q[2],q[3];
ry(1.4147778536146602) q[2];
ry(1.6961548089880318) q[3];
cx q[2],q[3];
ry(0.3954404869693944) q[4];
ry(2.1720075969931307) q[5];
cx q[4],q[5];
ry(-0.17173202400578133) q[4];
ry(0.10282898666152017) q[5];
cx q[4],q[5];
ry(1.6671124021853831) q[6];
ry(-2.983551989301304) q[7];
cx q[6],q[7];
ry(0.6376574530296112) q[6];
ry(1.9059179454514907) q[7];
cx q[6],q[7];
ry(2.729124870008088) q[8];
ry(2.334320715427352) q[9];
cx q[8],q[9];
ry(-0.6459227858032985) q[8];
ry(-2.2207546124155595) q[9];
cx q[8],q[9];
ry(-2.513062983592165) q[10];
ry(0.053311822653845375) q[11];
cx q[10],q[11];
ry(0.6306753553771918) q[10];
ry(2.429146588381077) q[11];
cx q[10],q[11];
ry(-0.10375739941080117) q[1];
ry(-0.5272893688525321) q[2];
cx q[1],q[2];
ry(0.05317340545955142) q[1];
ry(-2.9208331491604063) q[2];
cx q[1],q[2];
ry(1.5411788065051641) q[3];
ry(-2.177923265068798) q[4];
cx q[3],q[4];
ry(-1.5405234200904643) q[3];
ry(1.5277430122001214) q[4];
cx q[3],q[4];
ry(-2.325972692460137) q[5];
ry(-1.4806368450523792) q[6];
cx q[5],q[6];
ry(-0.0020999151014979844) q[5];
ry(3.0849423373334504) q[6];
cx q[5],q[6];
ry(1.8126031279000494) q[7];
ry(0.8360737719547799) q[8];
cx q[7],q[8];
ry(0.0538393387893573) q[7];
ry(1.8444425318769941) q[8];
cx q[7],q[8];
ry(-2.2663932119754753) q[9];
ry(-0.7870844361798688) q[10];
cx q[9],q[10];
ry(1.9695022835169) q[9];
ry(1.9640348969378723) q[10];
cx q[9],q[10];
ry(-2.6840291189518193) q[0];
ry(1.2669585529758667) q[1];
cx q[0],q[1];
ry(0.3415220119211106) q[0];
ry(0.05761010374956469) q[1];
cx q[0],q[1];
ry(-2.480395822142164) q[2];
ry(-2.1029438341507136) q[3];
cx q[2],q[3];
ry(-2.9807742154735983) q[2];
ry(1.4811875619491581) q[3];
cx q[2],q[3];
ry(1.442490300326267) q[4];
ry(-1.0315875864764288) q[5];
cx q[4],q[5];
ry(-3.120208005352962) q[4];
ry(-0.014570974494230171) q[5];
cx q[4],q[5];
ry(-2.291812313629521) q[6];
ry(1.5937323973739252) q[7];
cx q[6],q[7];
ry(0.46140583017615006) q[6];
ry(-0.025210017677801146) q[7];
cx q[6],q[7];
ry(2.257777344928008) q[8];
ry(1.5276760141525592) q[9];
cx q[8],q[9];
ry(-0.31226240731192156) q[8];
ry(1.7238858251702927) q[9];
cx q[8],q[9];
ry(0.28184963506648675) q[10];
ry(3.0402809976200174) q[11];
cx q[10],q[11];
ry(2.1433190326466267) q[10];
ry(1.9979016990247513) q[11];
cx q[10],q[11];
ry(0.13116261184976175) q[1];
ry(1.3005403729313105) q[2];
cx q[1],q[2];
ry(-0.5053580002051954) q[1];
ry(-1.4328981333155955) q[2];
cx q[1],q[2];
ry(-2.1039903878827624) q[3];
ry(2.8329795281795223) q[4];
cx q[3],q[4];
ry(0.2425009610560167) q[3];
ry(-1.5466975795468416) q[4];
cx q[3],q[4];
ry(2.0473497975083443) q[5];
ry(-0.9501010129643336) q[6];
cx q[5],q[6];
ry(1.5993926430980343) q[5];
ry(-0.7534503450034565) q[6];
cx q[5],q[6];
ry(0.817735474330683) q[7];
ry(1.5516774540505227) q[8];
cx q[7],q[8];
ry(1.6870708054884467) q[7];
ry(0.03785770814028223) q[8];
cx q[7],q[8];
ry(-1.2315172121671318) q[9];
ry(-0.44350905954076847) q[10];
cx q[9],q[10];
ry(-0.004120225322185997) q[9];
ry(0.5485145160102238) q[10];
cx q[9],q[10];
ry(-2.9142118454366037) q[0];
ry(-0.7767741895111749) q[1];
cx q[0],q[1];
ry(1.5716968144706431) q[0];
ry(-2.4568712020009627) q[1];
cx q[0],q[1];
ry(-2.9789863613264904) q[2];
ry(-1.6029798729560023) q[3];
cx q[2],q[3];
ry(-1.5138667570052826) q[2];
ry(-2.9557627344263966) q[3];
cx q[2],q[3];
ry(-2.683066830077061) q[4];
ry(-1.5715282034546532) q[5];
cx q[4],q[5];
ry(0.46228664906223127) q[4];
ry(-2.766488419847406) q[5];
cx q[4],q[5];
ry(-1.5723651543375512) q[6];
ry(2.310984221607829) q[7];
cx q[6],q[7];
ry(-1.5696389874324375) q[6];
ry(2.037849190868153) q[7];
cx q[6],q[7];
ry(0.1846550590627034) q[8];
ry(1.5526616501655077) q[9];
cx q[8],q[9];
ry(0.9356683988195904) q[8];
ry(-3.1327637654325944) q[9];
cx q[8],q[9];
ry(2.5441385968480006) q[10];
ry(-2.292000516491431) q[11];
cx q[10],q[11];
ry(-0.31408894981450947) q[10];
ry(0.2413351429690005) q[11];
cx q[10],q[11];
ry(2.9006474316790953) q[1];
ry(2.076527566832077) q[2];
cx q[1],q[2];
ry(-0.03028183694518205) q[1];
ry(-2.861255787930873) q[2];
cx q[1],q[2];
ry(-1.5630401173681072) q[3];
ry(-1.5678301976837608) q[4];
cx q[3],q[4];
ry(-1.6268737833980662) q[3];
ry(2.9649505680169854) q[4];
cx q[3],q[4];
ry(-1.5630613326961598) q[5];
ry(-1.5732926396445617) q[6];
cx q[5],q[6];
ry(-2.03732656140771) q[5];
ry(-0.4676187251516992) q[6];
cx q[5],q[6];
ry(1.3149201240546766) q[7];
ry(-2.9706131364666173) q[8];
cx q[7],q[8];
ry(1.5630967967233458) q[7];
ry(-0.006077312767659756) q[8];
cx q[7],q[8];
ry(1.7183677693045638) q[9];
ry(2.9411963393102263) q[10];
cx q[9],q[10];
ry(-0.5383816742969643) q[9];
ry(2.9869374891269618) q[10];
cx q[9],q[10];
ry(-0.7432298520952285) q[0];
ry(1.2334794151979551) q[1];
cx q[0],q[1];
ry(-2.41733924400924) q[0];
ry(2.771758169070655) q[1];
cx q[0],q[1];
ry(-0.38406135542507613) q[2];
ry(-2.662569171335489) q[3];
cx q[2],q[3];
ry(-1.4191144224859624) q[2];
ry(-0.512735096762406) q[3];
cx q[2],q[3];
ry(1.5077173546406055) q[4];
ry(-1.5728603270955026) q[5];
cx q[4],q[5];
ry(0.17008509868581329) q[4];
ry(0.306080480369659) q[5];
cx q[4],q[5];
ry(-0.9513564058483607) q[6];
ry(1.825734221207549) q[7];
cx q[6],q[7];
ry(-2.9099259751367885) q[6];
ry(-3.138791965768151) q[7];
cx q[6],q[7];
ry(0.6402572595543461) q[8];
ry(1.6923875377460658) q[9];
cx q[8],q[9];
ry(-1.5625003781991473) q[8];
ry(0.0006266880903008188) q[9];
cx q[8],q[9];
ry(2.7787044869128965) q[10];
ry(-1.9609640022923225) q[11];
cx q[10],q[11];
ry(-2.370057275098895) q[10];
ry(0.3620458512703735) q[11];
cx q[10],q[11];
ry(-2.4067390327414646) q[1];
ry(-1.4177458300910117) q[2];
cx q[1],q[2];
ry(3.1390185576672334) q[1];
ry(-2.56695162108768) q[2];
cx q[1],q[2];
ry(1.590867387667462) q[3];
ry(3.025210117207115) q[4];
cx q[3],q[4];
ry(-0.00925983569559835) q[3];
ry(-0.00010852990866005996) q[4];
cx q[3],q[4];
ry(1.848343211380542) q[5];
ry(2.1886719465451057) q[6];
cx q[5],q[6];
ry(-2.0324873721510146) q[5];
ry(3.1339888002635683) q[6];
cx q[5],q[6];
ry(1.5660695321032925) q[7];
ry(2.5013659739392597) q[8];
cx q[7],q[8];
ry(2.1721285724362858) q[7];
ry(-0.2608334770208254) q[8];
cx q[7],q[8];
ry(-1.5649241714864501) q[9];
ry(3.0764167065706243) q[10];
cx q[9],q[10];
ry(1.559724860625988) q[9];
ry(-2.8299165575250216) q[10];
cx q[9],q[10];
ry(-1.2885768163233278) q[0];
ry(0.7313821870763473) q[1];
cx q[0],q[1];
ry(0.8602614561110604) q[0];
ry(1.5665566161277216) q[1];
cx q[0],q[1];
ry(-1.3537881464982409) q[2];
ry(3.0622484657223765) q[3];
cx q[2],q[3];
ry(2.0406858792580893) q[2];
ry(1.1447945695538015) q[3];
cx q[2],q[3];
ry(0.6347953050083956) q[4];
ry(1.7971691963263545) q[5];
cx q[4],q[5];
ry(3.078440840122293) q[4];
ry(0.04254547097948613) q[5];
cx q[4],q[5];
ry(1.5712199724734832) q[6];
ry(-1.5640648178875063) q[7];
cx q[6],q[7];
ry(1.1531796551739149) q[6];
ry(-1.5779852762997093) q[7];
cx q[6],q[7];
ry(1.5722137197630872) q[8];
ry(-1.564718664652992) q[9];
cx q[8],q[9];
ry(1.7283236246384155) q[8];
ry(1.566751744808788) q[9];
cx q[8],q[9];
ry(1.5695294341617996) q[10];
ry(-2.8492320995543103) q[11];
cx q[10],q[11];
ry(1.5730102104332246) q[10];
ry(-2.0554834546097784) q[11];
cx q[10],q[11];
ry(2.611320692054566) q[1];
ry(-1.5414974156970613) q[2];
cx q[1],q[2];
ry(1.6134869843264716) q[1];
ry(-0.020333819006266844) q[2];
cx q[1],q[2];
ry(1.240807122907521) q[3];
ry(-0.5134206423391952) q[4];
cx q[3],q[4];
ry(0.0012542722548438755) q[3];
ry(3.139214877291393) q[4];
cx q[3],q[4];
ry(1.515587801090199) q[5];
ry(1.5564525296602403) q[6];
cx q[5],q[6];
ry(-0.03497619036968658) q[5];
ry(-0.6825553604433531) q[6];
cx q[5],q[6];
ry(1.5700232717596707) q[7];
ry(-1.572021936946898) q[8];
cx q[7],q[8];
ry(1.4081769289544441) q[7];
ry(-1.5603762500051535) q[8];
cx q[7],q[8];
ry(1.5789087347234783) q[9];
ry(-1.4852910044300147) q[10];
cx q[9],q[10];
ry(-3.1403449814618978) q[9];
ry(0.050325802451705454) q[10];
cx q[9],q[10];
ry(-3.107639085087722) q[0];
ry(2.1851609013331297) q[1];
ry(-0.010342889102458841) q[2];
ry(3.0030260676591674) q[3];
ry(-1.5060908527952614) q[4];
ry(-1.5666149818311048) q[5];
ry(0.013732376807748083) q[6];
ry(1.5702929972811166) q[7];
ry(-0.0005591199336612007) q[8];
ry(1.5649752006396371) q[9];
ry(-3.054283564715821) q[10];
ry(1.5773988651895547) q[11];