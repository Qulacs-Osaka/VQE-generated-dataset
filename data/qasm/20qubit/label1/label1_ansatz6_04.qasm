OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(2.63780767619507) q[0];
ry(-2.986383219158479) q[1];
cx q[0],q[1];
ry(2.790410142623628) q[0];
ry(0.4477635703455155) q[1];
cx q[0],q[1];
ry(-0.7410588786832902) q[1];
ry(-1.4696559293515066) q[2];
cx q[1],q[2];
ry(2.122975647026635) q[1];
ry(-0.13739341190566864) q[2];
cx q[1],q[2];
ry(-2.0178975994966084) q[2];
ry(0.1367926240568125) q[3];
cx q[2],q[3];
ry(-1.5074182914013825) q[2];
ry(-0.577462983080926) q[3];
cx q[2],q[3];
ry(1.0030739945656566) q[3];
ry(2.4960117943926092) q[4];
cx q[3],q[4];
ry(1.1906656069972492) q[3];
ry(0.8304839718058714) q[4];
cx q[3],q[4];
ry(-1.0065806981092216) q[4];
ry(2.9822397258276436) q[5];
cx q[4],q[5];
ry(0.477587020555016) q[4];
ry(-2.769544669968501) q[5];
cx q[4],q[5];
ry(-2.0475019425588776) q[5];
ry(2.519693511774935) q[6];
cx q[5],q[6];
ry(-2.117123111564912) q[5];
ry(-2.1453606066144544) q[6];
cx q[5],q[6];
ry(0.3284921665278615) q[6];
ry(-2.375047677703729) q[7];
cx q[6],q[7];
ry(-1.1115886542555602) q[6];
ry(-1.1549536463281465) q[7];
cx q[6],q[7];
ry(-1.6900243814021358) q[7];
ry(0.6312750991018046) q[8];
cx q[7],q[8];
ry(-0.9162779356275603) q[7];
ry(2.1141109740664477) q[8];
cx q[7],q[8];
ry(-1.9186468581401837) q[8];
ry(-2.5385386850813387) q[9];
cx q[8],q[9];
ry(0.6272006446201506) q[8];
ry(-0.7488712538497956) q[9];
cx q[8],q[9];
ry(-2.5292469895827283) q[9];
ry(-2.7779476675340598) q[10];
cx q[9],q[10];
ry(0.44084064132170475) q[9];
ry(-0.5113291126965116) q[10];
cx q[9],q[10];
ry(2.7791719532643033) q[10];
ry(-0.4246347343129644) q[11];
cx q[10],q[11];
ry(-1.6349416537010815) q[10];
ry(2.033049593469414) q[11];
cx q[10],q[11];
ry(0.9263005016161934) q[11];
ry(2.514842332681648) q[12];
cx q[11],q[12];
ry(-1.2812444698488354) q[11];
ry(-1.5712772003886935) q[12];
cx q[11],q[12];
ry(-1.586349764431861) q[12];
ry(0.9798603606328157) q[13];
cx q[12],q[13];
ry(3.054870794642718) q[12];
ry(-3.1155243140683955) q[13];
cx q[12],q[13];
ry(2.569631392654195) q[13];
ry(0.9295313593654467) q[14];
cx q[13],q[14];
ry(-1.997584280555496) q[13];
ry(-1.803709469595988) q[14];
cx q[13],q[14];
ry(-3.0867208364505094) q[14];
ry(-1.8543586444382139) q[15];
cx q[14],q[15];
ry(2.5743434592807533) q[14];
ry(-2.2441020915683705) q[15];
cx q[14],q[15];
ry(0.15008904114766963) q[15];
ry(-2.3406082173180986) q[16];
cx q[15],q[16];
ry(0.7339610367908858) q[15];
ry(0.5213826841620044) q[16];
cx q[15],q[16];
ry(-0.5297817036066902) q[16];
ry(0.3683350773627137) q[17];
cx q[16],q[17];
ry(-3.045432879601503) q[16];
ry(-3.129309378937871) q[17];
cx q[16],q[17];
ry(0.004146391366414629) q[17];
ry(1.1693420696108054) q[18];
cx q[17],q[18];
ry(-3.1179509426259027) q[17];
ry(-0.030117462904951826) q[18];
cx q[17],q[18];
ry(0.8917410062235227) q[18];
ry(2.9141420639466267) q[19];
cx q[18],q[19];
ry(-0.9401973765662285) q[18];
ry(1.1844800776352153) q[19];
cx q[18],q[19];
ry(-0.9662792042362803) q[0];
ry(-0.3502081082584354) q[1];
cx q[0],q[1];
ry(-3.1136427380464515) q[0];
ry(0.06259033797146518) q[1];
cx q[0],q[1];
ry(0.21768499061778443) q[1];
ry(2.6412558887172533) q[2];
cx q[1],q[2];
ry(-0.01910480726062573) q[1];
ry(-0.7202773447991282) q[2];
cx q[1],q[2];
ry(3.014431243606606) q[2];
ry(-1.3665083153683195) q[3];
cx q[2],q[3];
ry(2.3249504618757073) q[2];
ry(2.954398039760719) q[3];
cx q[2],q[3];
ry(-1.6625395371437992) q[3];
ry(-1.9258351812152572) q[4];
cx q[3],q[4];
ry(1.8783244587817407) q[3];
ry(0.6722369559574632) q[4];
cx q[3],q[4];
ry(2.2487451439331947) q[4];
ry(0.8160900545052527) q[5];
cx q[4],q[5];
ry(0.9806290806334674) q[4];
ry(-0.3584721484387918) q[5];
cx q[4],q[5];
ry(1.778442887852865) q[5];
ry(2.0312139472639066) q[6];
cx q[5],q[6];
ry(-1.7991702473865676) q[5];
ry(0.2845809416092499) q[6];
cx q[5],q[6];
ry(-2.6364868702646964) q[6];
ry(-1.670462558501102) q[7];
cx q[6],q[7];
ry(1.579162050566238) q[6];
ry(0.8238424211189042) q[7];
cx q[6],q[7];
ry(-2.8692592133283594) q[7];
ry(-1.134464000992099) q[8];
cx q[7],q[8];
ry(1.2489086941037888) q[7];
ry(-2.815057081444988) q[8];
cx q[7],q[8];
ry(2.7259555843558854) q[8];
ry(-1.8731722149662264) q[9];
cx q[8],q[9];
ry(1.2554168459356876) q[8];
ry(-0.6230433850546622) q[9];
cx q[8],q[9];
ry(-3.0902209947551023) q[9];
ry(-1.6587237163026902) q[10];
cx q[9],q[10];
ry(1.1934775313722774) q[9];
ry(-0.3767197423801807) q[10];
cx q[9],q[10];
ry(-2.4105256719840082) q[10];
ry(2.4908659382170955) q[11];
cx q[10],q[11];
ry(-2.7967755007063384) q[10];
ry(0.5780651196353856) q[11];
cx q[10],q[11];
ry(-1.6260698444452046) q[11];
ry(-0.11384441768267539) q[12];
cx q[11],q[12];
ry(-0.9819976625443188) q[11];
ry(3.08184558240413) q[12];
cx q[11],q[12];
ry(0.5644268362167528) q[12];
ry(-1.246925183255554) q[13];
cx q[12],q[13];
ry(3.087530274392551) q[12];
ry(-0.008159153229979843) q[13];
cx q[12],q[13];
ry(-0.07698526348000444) q[13];
ry(2.6328072618463696) q[14];
cx q[13],q[14];
ry(0.6804885691517588) q[13];
ry(2.9610834667673513) q[14];
cx q[13],q[14];
ry(0.3887938981364071) q[14];
ry(0.4992784019369919) q[15];
cx q[14],q[15];
ry(0.012711259426903654) q[14];
ry(-3.085326636000649) q[15];
cx q[14],q[15];
ry(-0.44894200958147046) q[15];
ry(2.2710530345936766) q[16];
cx q[15],q[16];
ry(-1.3709481041018412) q[15];
ry(-0.5112808201419714) q[16];
cx q[15],q[16];
ry(2.032062113465196) q[16];
ry(1.474298190634598) q[17];
cx q[16],q[17];
ry(0.04724019892408649) q[16];
ry(2.9871006181579487) q[17];
cx q[16],q[17];
ry(2.6390654408447896) q[17];
ry(-2.7628092691502184) q[18];
cx q[17],q[18];
ry(0.8720158947808355) q[17];
ry(1.798281729642997) q[18];
cx q[17],q[18];
ry(-0.6420258620834858) q[18];
ry(-1.4722622490989168) q[19];
cx q[18],q[19];
ry(-1.9463746043981134) q[18];
ry(1.6205212835701976) q[19];
cx q[18],q[19];
ry(-2.2405962772402557) q[0];
ry(2.200536940467021) q[1];
cx q[0],q[1];
ry(-1.0306274695731523) q[0];
ry(2.482050922292081) q[1];
cx q[0],q[1];
ry(-2.226732484995297) q[1];
ry(-2.9378802762964877) q[2];
cx q[1],q[2];
ry(-0.03906059900715407) q[1];
ry(2.918157727295347) q[2];
cx q[1],q[2];
ry(0.23094503441628814) q[2];
ry(1.548624438603561) q[3];
cx q[2],q[3];
ry(1.0675385967922453) q[2];
ry(-1.6620283606650106) q[3];
cx q[2],q[3];
ry(0.6219427185359729) q[3];
ry(1.4119208141879651) q[4];
cx q[3],q[4];
ry(-0.3838446309025656) q[3];
ry(-0.8858337295205764) q[4];
cx q[3],q[4];
ry(-1.8755529028774243) q[4];
ry(-2.2521320432536234) q[5];
cx q[4],q[5];
ry(0.11714415663124456) q[4];
ry(-3.1142484758666265) q[5];
cx q[4],q[5];
ry(0.17804220410406096) q[5];
ry(0.060109498908467916) q[6];
cx q[5],q[6];
ry(-2.281795088540586) q[5];
ry(-2.9790512703467207) q[6];
cx q[5],q[6];
ry(-1.424010916555746) q[6];
ry(-0.6819932646692726) q[7];
cx q[6],q[7];
ry(-3.0009182506662713) q[6];
ry(2.9280365853976678) q[7];
cx q[6],q[7];
ry(-0.4323290612684157) q[7];
ry(0.9001032716928847) q[8];
cx q[7],q[8];
ry(-2.814073173370239) q[7];
ry(-0.19772631259831364) q[8];
cx q[7],q[8];
ry(1.152182260554861) q[8];
ry(1.3972006409677382) q[9];
cx q[8],q[9];
ry(-2.9507587457301963) q[8];
ry(-0.9871813862350098) q[9];
cx q[8],q[9];
ry(2.548120773033259) q[9];
ry(-2.5370744150977287) q[10];
cx q[9],q[10];
ry(3.021756078352362) q[9];
ry(-1.2185482275290127) q[10];
cx q[9],q[10];
ry(-2.415541551626944) q[10];
ry(-1.7907254889778323) q[11];
cx q[10],q[11];
ry(0.9059872597520426) q[10];
ry(0.7179837782987954) q[11];
cx q[10],q[11];
ry(0.39350609733188424) q[11];
ry(-1.4441718346780295) q[12];
cx q[11],q[12];
ry(-1.469666454946518) q[11];
ry(0.504895258630101) q[12];
cx q[11],q[12];
ry(2.628055612374275) q[12];
ry(-0.843066653136529) q[13];
cx q[12],q[13];
ry(-2.715944350414667) q[12];
ry(-0.706338251786943) q[13];
cx q[12],q[13];
ry(1.3029129679129037) q[13];
ry(-0.23896435527721402) q[14];
cx q[13],q[14];
ry(-2.253982234654985) q[13];
ry(1.863332939216168) q[14];
cx q[13],q[14];
ry(-0.12024109612390906) q[14];
ry(-0.10773952449417655) q[15];
cx q[14],q[15];
ry(-3.082548286337163) q[14];
ry(0.8809429307035401) q[15];
cx q[14],q[15];
ry(0.32180505472103693) q[15];
ry(1.6004130363604585) q[16];
cx q[15],q[16];
ry(1.3478060307590414) q[15];
ry(3.064224936442061) q[16];
cx q[15],q[16];
ry(-2.389887514328967) q[16];
ry(-1.2707427355819878) q[17];
cx q[16],q[17];
ry(1.876736465811918) q[16];
ry(-1.2422514250521706) q[17];
cx q[16],q[17];
ry(-0.3104487110537263) q[17];
ry(0.2653934375832572) q[18];
cx q[17],q[18];
ry(-3.1302607909967732) q[17];
ry(3.120869554410297) q[18];
cx q[17],q[18];
ry(-0.08938677698648068) q[18];
ry(1.7192759226789887) q[19];
cx q[18],q[19];
ry(1.180302464294293) q[18];
ry(-0.5300815629861599) q[19];
cx q[18],q[19];
ry(1.6978998963522582) q[0];
ry(1.271924005093114) q[1];
cx q[0],q[1];
ry(2.885589600276607) q[0];
ry(-1.9902455289020244) q[1];
cx q[0],q[1];
ry(-1.4789634027918126) q[1];
ry(-0.6343048425466202) q[2];
cx q[1],q[2];
ry(-2.8768922311374188) q[1];
ry(-3.068230922532781) q[2];
cx q[1],q[2];
ry(-0.5003344929146427) q[2];
ry(-1.2138492372087228) q[3];
cx q[2],q[3];
ry(1.5100417357769915) q[2];
ry(-1.6297825048709036) q[3];
cx q[2],q[3];
ry(3.045269394306936) q[3];
ry(-2.840746072520587) q[4];
cx q[3],q[4];
ry(-2.7684244241025397) q[3];
ry(-0.692071752928304) q[4];
cx q[3],q[4];
ry(2.464807289763829) q[4];
ry(1.3948604795660655) q[5];
cx q[4],q[5];
ry(-3.106979547718131) q[4];
ry(-3.002772661327325) q[5];
cx q[4],q[5];
ry(-0.5080882854479618) q[5];
ry(2.947269696331028) q[6];
cx q[5],q[6];
ry(0.7230157909035622) q[5];
ry(2.1194802540931565) q[6];
cx q[5],q[6];
ry(0.11388766905597603) q[6];
ry(-1.6708005377711501) q[7];
cx q[6],q[7];
ry(0.02923722180880972) q[6];
ry(2.94607022633504) q[7];
cx q[6],q[7];
ry(0.21094767113066465) q[7];
ry(1.759680656936414) q[8];
cx q[7],q[8];
ry(-0.6022914069451568) q[7];
ry(-0.07757948817907935) q[8];
cx q[7],q[8];
ry(-0.7181763734552771) q[8];
ry(-0.4090070210306225) q[9];
cx q[8],q[9];
ry(-3.034529038198116) q[8];
ry(-0.37056434741592487) q[9];
cx q[8],q[9];
ry(-0.5465912884891306) q[9];
ry(-2.120332149808801) q[10];
cx q[9],q[10];
ry(-0.26471060181437167) q[9];
ry(-0.3544570601042105) q[10];
cx q[9],q[10];
ry(2.9709946578120783) q[10];
ry(-0.9716287809263839) q[11];
cx q[10],q[11];
ry(0.5712906428738842) q[10];
ry(2.078936109930817) q[11];
cx q[10],q[11];
ry(-2.321894482860074) q[11];
ry(0.6294553448915537) q[12];
cx q[11],q[12];
ry(1.7057650041595371) q[11];
ry(2.559423130055975) q[12];
cx q[11],q[12];
ry(2.8775369244756894) q[12];
ry(2.8675515209117126) q[13];
cx q[12],q[13];
ry(-0.005221236447022662) q[12];
ry(-3.1216779341689156) q[13];
cx q[12],q[13];
ry(2.0537106241749488) q[13];
ry(-2.097130493413411) q[14];
cx q[13],q[14];
ry(-1.1369135299809539) q[13];
ry(-2.6301770993024176) q[14];
cx q[13],q[14];
ry(-0.44813354972060737) q[14];
ry(1.1164032099672363) q[15];
cx q[14],q[15];
ry(2.4671554157309954) q[14];
ry(0.9134369971625036) q[15];
cx q[14],q[15];
ry(-0.3428896654479107) q[15];
ry(0.9662565869486732) q[16];
cx q[15],q[16];
ry(1.4161508047018994) q[15];
ry(2.881805792104035) q[16];
cx q[15],q[16];
ry(1.7042247408392965) q[16];
ry(-2.3153827977243204) q[17];
cx q[16],q[17];
ry(0.07467242272455632) q[16];
ry(0.8157987719165992) q[17];
cx q[16],q[17];
ry(-1.778211287092181) q[17];
ry(0.365236231727468) q[18];
cx q[17],q[18];
ry(0.010780485424667342) q[17];
ry(0.2024068235114935) q[18];
cx q[17],q[18];
ry(3.0062079314161094) q[18];
ry(-2.8426924724955605) q[19];
cx q[18],q[19];
ry(-3.123752441794891) q[18];
ry(1.1943614938817058) q[19];
cx q[18],q[19];
ry(-0.2659487613812015) q[0];
ry(1.0870630096070881) q[1];
cx q[0],q[1];
ry(3.038385690316734) q[0];
ry(0.15953614826616125) q[1];
cx q[0],q[1];
ry(-2.1989545745685755) q[1];
ry(2.5224137451285102) q[2];
cx q[1],q[2];
ry(1.043841081388182) q[1];
ry(-2.1730356907959294) q[2];
cx q[1],q[2];
ry(-1.674701057601217) q[2];
ry(3.0273100608894907) q[3];
cx q[2],q[3];
ry(0.6278904794336446) q[2];
ry(2.5558105095666335) q[3];
cx q[2],q[3];
ry(-1.3599149271979123) q[3];
ry(0.199851507820507) q[4];
cx q[3],q[4];
ry(0.15219802789419035) q[3];
ry(-2.9391746871957305) q[4];
cx q[3],q[4];
ry(1.5929810273815808) q[4];
ry(1.2741729409117841) q[5];
cx q[4],q[5];
ry(-0.0513616430284749) q[4];
ry(-2.993253914881783) q[5];
cx q[4],q[5];
ry(0.5162688459202442) q[5];
ry(-1.7425971572587766) q[6];
cx q[5],q[6];
ry(-1.8254384089813662) q[5];
ry(-3.0680302944009377) q[6];
cx q[5],q[6];
ry(3.0684850057297637) q[6];
ry(-2.518536788713189) q[7];
cx q[6],q[7];
ry(-0.09688636143745642) q[6];
ry(2.887715868493203) q[7];
cx q[6],q[7];
ry(-1.1244423469988432) q[7];
ry(2.907645501884802) q[8];
cx q[7],q[8];
ry(-0.0043754177774427624) q[7];
ry(0.19328014754333278) q[8];
cx q[7],q[8];
ry(-1.791105708844726) q[8];
ry(2.8729939265405764) q[9];
cx q[8],q[9];
ry(-2.0858145555165573) q[8];
ry(1.8681431189530562) q[9];
cx q[8],q[9];
ry(2.5341724376044223) q[9];
ry(0.0829300597122975) q[10];
cx q[9],q[10];
ry(0.052370106816112615) q[9];
ry(3.0730571960407005) q[10];
cx q[9],q[10];
ry(1.9536996176195434) q[10];
ry(1.1652196255441964) q[11];
cx q[10],q[11];
ry(3.0354645936653744) q[10];
ry(-1.4662833222749878) q[11];
cx q[10],q[11];
ry(1.4489791229046285) q[11];
ry(-1.8992899369631697) q[12];
cx q[11],q[12];
ry(2.447406027490293) q[11];
ry(-0.3946913159684682) q[12];
cx q[11],q[12];
ry(2.9972738489392543) q[12];
ry(0.2250291434620051) q[13];
cx q[12],q[13];
ry(-1.5711568681884387) q[12];
ry(2.485104775130978) q[13];
cx q[12],q[13];
ry(-1.5556054198692513) q[13];
ry(-0.1457857069532051) q[14];
cx q[13],q[14];
ry(1.6231517233886907) q[13];
ry(-2.5603167564219906) q[14];
cx q[13],q[14];
ry(-3.028158968944034) q[14];
ry(2.208587092598999) q[15];
cx q[14],q[15];
ry(-1.6611733806053586) q[14];
ry(1.701282422251527) q[15];
cx q[14],q[15];
ry(-1.5600383241711682) q[15];
ry(-2.3815723787874097) q[16];
cx q[15],q[16];
ry(1.4600346013715477) q[15];
ry(1.3672647179903787) q[16];
cx q[15],q[16];
ry(-0.07421380530652932) q[16];
ry(2.0593559185005104) q[17];
cx q[16],q[17];
ry(0.2609148394621821) q[16];
ry(-3.062900589252137) q[17];
cx q[16],q[17];
ry(0.05924653023410515) q[17];
ry(-3.0633094727866883) q[18];
cx q[17],q[18];
ry(-1.0913097031317587) q[17];
ry(2.501319009429987) q[18];
cx q[17],q[18];
ry(2.8525661356612653) q[18];
ry(1.8522627075835931) q[19];
cx q[18],q[19];
ry(-0.4959569377108298) q[18];
ry(0.45788178280546216) q[19];
cx q[18],q[19];
ry(-0.4483802146906448) q[0];
ry(-1.83190843658312) q[1];
cx q[0],q[1];
ry(0.7134688920050944) q[0];
ry(-2.4649011340477247) q[1];
cx q[0],q[1];
ry(2.4227088420002185) q[1];
ry(1.5593376919382183) q[2];
cx q[1],q[2];
ry(-2.6597491680656233) q[1];
ry(0.31969932148091473) q[2];
cx q[1],q[2];
ry(-0.2403781256743995) q[2];
ry(0.9358938709752957) q[3];
cx q[2],q[3];
ry(-1.674321863586341) q[2];
ry(0.5765189269610029) q[3];
cx q[2],q[3];
ry(-0.4690160475899896) q[3];
ry(-2.8787013427102313) q[4];
cx q[3],q[4];
ry(0.019726203767814022) q[3];
ry(-0.0020574687792254973) q[4];
cx q[3],q[4];
ry(-2.9485438520210296) q[4];
ry(-0.6974683233538821) q[5];
cx q[4],q[5];
ry(-3.101691507459283) q[4];
ry(-3.1139384265343124) q[5];
cx q[4],q[5];
ry(0.9048945382639492) q[5];
ry(1.843804072970815) q[6];
cx q[5],q[6];
ry(1.3446797401596007) q[5];
ry(1.230345031122245) q[6];
cx q[5],q[6];
ry(-0.5785523692689392) q[6];
ry(1.8276291589692937) q[7];
cx q[6],q[7];
ry(2.852409668383975) q[6];
ry(0.30046819722792595) q[7];
cx q[6],q[7];
ry(0.7545956569726924) q[7];
ry(1.4557893997407039) q[8];
cx q[7],q[8];
ry(3.12584176169064) q[7];
ry(-3.0945265075774313) q[8];
cx q[7],q[8];
ry(2.2778061018707785) q[8];
ry(-3.120354035142712) q[9];
cx q[8],q[9];
ry(1.196469048423611) q[8];
ry(2.2498916897725887) q[9];
cx q[8],q[9];
ry(1.0816035719821508) q[9];
ry(1.5119034759409251) q[10];
cx q[9],q[10];
ry(3.0309931708475877) q[9];
ry(-3.1183145838730484) q[10];
cx q[9],q[10];
ry(-1.6807098520097659) q[10];
ry(1.2947572665236278) q[11];
cx q[10],q[11];
ry(3.1395205426643122) q[10];
ry(-0.9684096872699092) q[11];
cx q[10],q[11];
ry(-1.6990905210390777) q[11];
ry(1.5261122063958776) q[12];
cx q[11],q[12];
ry(-0.7460220882343942) q[11];
ry(-0.2164618166431385) q[12];
cx q[11],q[12];
ry(-2.9510906114539477) q[12];
ry(0.4788169798380358) q[13];
cx q[12],q[13];
ry(3.0299476177340674) q[12];
ry(-1.5026706421432547) q[13];
cx q[12],q[13];
ry(-2.359620471421381) q[13];
ry(0.16097523017808313) q[14];
cx q[13],q[14];
ry(-0.010757459532122482) q[13];
ry(-0.04536403642681679) q[14];
cx q[13],q[14];
ry(0.9396541198993015) q[14];
ry(-2.807446821239596) q[15];
cx q[14],q[15];
ry(-3.0722205759904475) q[14];
ry(-2.152550520807741) q[15];
cx q[14],q[15];
ry(-2.627038787880761) q[15];
ry(1.9907391872539582) q[16];
cx q[15],q[16];
ry(-0.0684387380368614) q[15];
ry(-1.5442769713772808) q[16];
cx q[15],q[16];
ry(3.0675210690993358) q[16];
ry(1.9184878364682074) q[17];
cx q[16],q[17];
ry(-1.8578211254475159) q[16];
ry(0.8275385585166282) q[17];
cx q[16],q[17];
ry(-0.3671280633292709) q[17];
ry(-1.9563615821639209) q[18];
cx q[17],q[18];
ry(-1.5144058312968989) q[17];
ry(0.9559003710925049) q[18];
cx q[17],q[18];
ry(0.8583299018126284) q[18];
ry(-0.38713577468448457) q[19];
cx q[18],q[19];
ry(2.7109857559594612) q[18];
ry(-0.19594177001562763) q[19];
cx q[18],q[19];
ry(0.9799492867448367) q[0];
ry(1.6255079052891976) q[1];
cx q[0],q[1];
ry(-0.060499113527059833) q[0];
ry(-3.054347709320965) q[1];
cx q[0],q[1];
ry(2.782863865549224) q[1];
ry(0.23344882183778423) q[2];
cx q[1],q[2];
ry(-0.27659073689299246) q[1];
ry(2.797989981827348) q[2];
cx q[1],q[2];
ry(0.1977494597587974) q[2];
ry(0.4334082882158574) q[3];
cx q[2],q[3];
ry(-1.178669850884905) q[2];
ry(0.7158441845909836) q[3];
cx q[2],q[3];
ry(0.8217473991383466) q[3];
ry(0.8117347762092748) q[4];
cx q[3],q[4];
ry(-0.21609280365069064) q[3];
ry(-2.9777558019589696) q[4];
cx q[3],q[4];
ry(1.6578142693909264) q[4];
ry(-2.4164213833954653) q[5];
cx q[4],q[5];
ry(3.0378961092087677) q[4];
ry(2.9405434135056416) q[5];
cx q[4],q[5];
ry(0.1929057526477695) q[5];
ry(-2.452964021865294) q[6];
cx q[5],q[6];
ry(0.42408222053127886) q[5];
ry(0.33043892050418755) q[6];
cx q[5],q[6];
ry(1.1607847588217322) q[6];
ry(1.0624448101922852) q[7];
cx q[6],q[7];
ry(-2.662239567721323) q[6];
ry(2.9064469733490927) q[7];
cx q[6],q[7];
ry(2.1434824652603774) q[7];
ry(0.9431627760256429) q[8];
cx q[7],q[8];
ry(-3.124419533301695) q[7];
ry(-2.961444118751647) q[8];
cx q[7],q[8];
ry(2.7535226904187615) q[8];
ry(-2.0170972112496917) q[9];
cx q[8],q[9];
ry(1.1271514007380898) q[8];
ry(1.6398414545231166) q[9];
cx q[8],q[9];
ry(-2.175823540952568) q[9];
ry(-2.8214595427935873) q[10];
cx q[9],q[10];
ry(3.034645297884868) q[9];
ry(-0.08027425376973252) q[10];
cx q[9],q[10];
ry(1.6343416807242672) q[10];
ry(-1.5735555640632024) q[11];
cx q[10],q[11];
ry(-0.10799794551259279) q[10];
ry(-2.9210659055652304) q[11];
cx q[10],q[11];
ry(0.14419005924754913) q[11];
ry(-1.8734490100630439) q[12];
cx q[11],q[12];
ry(0.05301830857872503) q[11];
ry(-0.07321366901360823) q[12];
cx q[11],q[12];
ry(-2.3188827269123973) q[12];
ry(0.6273293469833865) q[13];
cx q[12],q[13];
ry(2.949736458332823) q[12];
ry(-1.4220750578201988) q[13];
cx q[12],q[13];
ry(2.523460244947527) q[13];
ry(-1.1272575673445537) q[14];
cx q[13],q[14];
ry(0.0005750588468782422) q[13];
ry(-3.1137371787911525) q[14];
cx q[13],q[14];
ry(-2.5133913811458695) q[14];
ry(1.5985288247870437) q[15];
cx q[14],q[15];
ry(-0.8728538404321746) q[14];
ry(1.7778199758789794) q[15];
cx q[14],q[15];
ry(2.7802646726882307) q[15];
ry(1.583586813895753) q[16];
cx q[15],q[16];
ry(-0.4300340773251205) q[15];
ry(-0.006134080673112085) q[16];
cx q[15],q[16];
ry(-0.5097279387256868) q[16];
ry(-2.3203112492653153) q[17];
cx q[16],q[17];
ry(-0.8677149008451089) q[16];
ry(-0.28610497689687886) q[17];
cx q[16],q[17];
ry(1.126254679909228) q[17];
ry(2.297423758988745) q[18];
cx q[17],q[18];
ry(-0.4255863034740699) q[17];
ry(0.6921745800782926) q[18];
cx q[17],q[18];
ry(-1.754975301850891) q[18];
ry(-1.6929936967236319) q[19];
cx q[18],q[19];
ry(1.9129018152651198) q[18];
ry(-2.4286216668440157) q[19];
cx q[18],q[19];
ry(1.822970413303361) q[0];
ry(0.6686200580532242) q[1];
ry(3.0403667688996756) q[2];
ry(1.8013861502627042) q[3];
ry(-2.112398879165398) q[4];
ry(-0.15330930727710612) q[5];
ry(1.2934111340236782) q[6];
ry(1.6191345457604633) q[7];
ry(-1.5972760636175898) q[8];
ry(0.5075344534648855) q[9];
ry(1.544875456329497) q[10];
ry(-0.13162605196403745) q[11];
ry(-0.026877247531631964) q[12];
ry(3.1069550461888205) q[13];
ry(-0.21461993627035358) q[14];
ry(0.33080742074694847) q[15];
ry(-2.314552666689588) q[16];
ry(-0.5593504838856456) q[17];
ry(1.5453361587419874) q[18];
ry(0.8897952411969943) q[19];