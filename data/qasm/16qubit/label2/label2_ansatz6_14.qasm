OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.11719030706491498) q[0];
ry(-2.8851651922526664) q[1];
cx q[0],q[1];
ry(2.174776214593546) q[0];
ry(1.399964095076579) q[1];
cx q[0],q[1];
ry(0.9419676580372683) q[1];
ry(-0.07724931879367603) q[2];
cx q[1],q[2];
ry(-2.595151893165845) q[1];
ry(2.4404570007911577) q[2];
cx q[1],q[2];
ry(3.1108199329070203) q[2];
ry(0.6485387521318513) q[3];
cx q[2],q[3];
ry(-0.7690856748085036) q[2];
ry(-1.4831338032238863) q[3];
cx q[2],q[3];
ry(-2.9392636614943433) q[3];
ry(1.0299296057995928) q[4];
cx q[3],q[4];
ry(-2.0157306002463593) q[3];
ry(0.9952738290344892) q[4];
cx q[3],q[4];
ry(-0.9740768232871339) q[4];
ry(2.0031602198149847) q[5];
cx q[4],q[5];
ry(-0.031913988797479025) q[4];
ry(0.9536201757049045) q[5];
cx q[4],q[5];
ry(-2.103648889969664) q[5];
ry(1.5512848045893861) q[6];
cx q[5],q[6];
ry(1.4604046687523127) q[5];
ry(-3.114418622206985) q[6];
cx q[5],q[6];
ry(-1.2972537959485777) q[6];
ry(-2.2966196458885824) q[7];
cx q[6],q[7];
ry(3.138524257518806) q[6];
ry(3.1293865913124432) q[7];
cx q[6],q[7];
ry(-1.6525463865564494) q[7];
ry(1.2998428121054602) q[8];
cx q[7],q[8];
ry(-0.526852907226611) q[7];
ry(1.9668306525846624) q[8];
cx q[7],q[8];
ry(-1.2205621784518583) q[8];
ry(1.6159244803841262) q[9];
cx q[8],q[9];
ry(-1.2126660583676356) q[8];
ry(0.037714888287679216) q[9];
cx q[8],q[9];
ry(-0.6768201083539243) q[9];
ry(-1.0598226384975176) q[10];
cx q[9],q[10];
ry(-0.47396409842764176) q[9];
ry(3.0336914686208836) q[10];
cx q[9],q[10];
ry(-2.297983980812411) q[10];
ry(1.5261330040467804) q[11];
cx q[10],q[11];
ry(0.37590356670386527) q[10];
ry(-3.051013042127327) q[11];
cx q[10],q[11];
ry(-1.1189568462867783) q[11];
ry(-1.5820568083519437) q[12];
cx q[11],q[12];
ry(2.3103497580691084) q[11];
ry(-0.008429749033339817) q[12];
cx q[11],q[12];
ry(-0.5376693621078913) q[12];
ry(2.6129455140385436) q[13];
cx q[12],q[13];
ry(2.293806748573361) q[12];
ry(2.8494190211955566) q[13];
cx q[12],q[13];
ry(2.02591520449925) q[13];
ry(2.2684731489160592) q[14];
cx q[13],q[14];
ry(-0.9985158334957581) q[13];
ry(-0.0990213306908025) q[14];
cx q[13],q[14];
ry(2.8937742512710902) q[14];
ry(1.5079620015717712) q[15];
cx q[14],q[15];
ry(1.238158161889502) q[14];
ry(-0.9920773549998059) q[15];
cx q[14],q[15];
ry(0.7512953488688732) q[0];
ry(0.7604482752832871) q[1];
cx q[0],q[1];
ry(-1.5835131769317794) q[0];
ry(0.11254014539993996) q[1];
cx q[0],q[1];
ry(1.451895657833909) q[1];
ry(1.0397517938100456) q[2];
cx q[1],q[2];
ry(0.12701950322541677) q[1];
ry(1.7891691247322608) q[2];
cx q[1],q[2];
ry(1.2984125367291908) q[2];
ry(1.3529915085318835) q[3];
cx q[2],q[3];
ry(0.3205552402023605) q[2];
ry(1.523631111923474) q[3];
cx q[2],q[3];
ry(2.2829367250193977) q[3];
ry(1.8615461675171812) q[4];
cx q[3],q[4];
ry(2.2056870953325207) q[3];
ry(-2.093806689396199) q[4];
cx q[3],q[4];
ry(0.10575353519509371) q[4];
ry(-1.3797063584665292) q[5];
cx q[4],q[5];
ry(1.6161358102701682) q[4];
ry(1.3929386069932157) q[5];
cx q[4],q[5];
ry(-2.1833384756518424) q[5];
ry(-1.328622054032537) q[6];
cx q[5],q[6];
ry(3.0269473513992002) q[5];
ry(3.1412569955296705) q[6];
cx q[5],q[6];
ry(-1.3576390148686919) q[6];
ry(-1.9902426704393559) q[7];
cx q[6],q[7];
ry(-3.132433281481196) q[6];
ry(-0.00912851044941734) q[7];
cx q[6],q[7];
ry(-1.2159214305698418) q[7];
ry(-0.10248869006746375) q[8];
cx q[7],q[8];
ry(1.4409394701563905) q[7];
ry(2.1638412805764897) q[8];
cx q[7],q[8];
ry(1.8268630222259796) q[8];
ry(-0.32311799580169687) q[9];
cx q[8],q[9];
ry(0.15379490150995956) q[8];
ry(-1.881137785388675) q[9];
cx q[8],q[9];
ry(1.4134666765448536) q[9];
ry(0.7201439270584178) q[10];
cx q[9],q[10];
ry(-1.6517311639883445) q[9];
ry(-1.323143602339707) q[10];
cx q[9],q[10];
ry(1.8546880317664598) q[10];
ry(0.6194714137536713) q[11];
cx q[10],q[11];
ry(0.13883200378776023) q[10];
ry(-3.075454548317394) q[11];
cx q[10],q[11];
ry(-1.7171257688714423) q[11];
ry(-3.020781074464476) q[12];
cx q[11],q[12];
ry(-3.141512622418466) q[11];
ry(-0.0005877371863931558) q[12];
cx q[11],q[12];
ry(1.6727261406307556) q[12];
ry(1.8433721359115207) q[13];
cx q[12],q[13];
ry(3.0593114990438344) q[12];
ry(-0.6488581630221574) q[13];
cx q[12],q[13];
ry(0.25635497627492904) q[13];
ry(1.4538661726032889) q[14];
cx q[13],q[14];
ry(1.0438363748125807) q[13];
ry(-2.896455191643914) q[14];
cx q[13],q[14];
ry(1.9194233822996187) q[14];
ry(-2.1105267683249727) q[15];
cx q[14],q[15];
ry(2.370020749531165) q[14];
ry(2.565267215953675) q[15];
cx q[14],q[15];
ry(2.0693258585143512) q[0];
ry(1.5242675825839136) q[1];
cx q[0],q[1];
ry(-2.2414528962707747) q[0];
ry(2.8052834683875987) q[1];
cx q[0],q[1];
ry(2.0814188587952884) q[1];
ry(-1.8417056360070347) q[2];
cx q[1],q[2];
ry(-1.8733558730286768) q[1];
ry(-0.2469360444204156) q[2];
cx q[1],q[2];
ry(1.7156319755762524) q[2];
ry(2.952128045002542) q[3];
cx q[2],q[3];
ry(-0.8948427879519575) q[2];
ry(1.3011099134037487) q[3];
cx q[2],q[3];
ry(-0.623155051748113) q[3];
ry(1.060226887096774) q[4];
cx q[3],q[4];
ry(-2.879377829370077) q[3];
ry(-0.21279450933201538) q[4];
cx q[3],q[4];
ry(-2.0640956162306967) q[4];
ry(-1.8350614197364594) q[5];
cx q[4],q[5];
ry(-1.407991603417711) q[4];
ry(1.4989940736924092) q[5];
cx q[4],q[5];
ry(-0.18370792709722839) q[5];
ry(0.9022294683355208) q[6];
cx q[5],q[6];
ry(0.004921684359857181) q[5];
ry(-0.003313776768707477) q[6];
cx q[5],q[6];
ry(1.5902219453535795) q[6];
ry(0.7166314988901004) q[7];
cx q[6],q[7];
ry(-1.8813106928654768) q[6];
ry(-2.8741983176306767) q[7];
cx q[6],q[7];
ry(-0.7723878125065246) q[7];
ry(-1.7675016197962898) q[8];
cx q[7],q[8];
ry(0.0019072585492230232) q[7];
ry(1.4039065274171783) q[8];
cx q[7],q[8];
ry(-0.7341307969907778) q[8];
ry(-0.42827163111359917) q[9];
cx q[8],q[9];
ry(2.007241348981484) q[8];
ry(3.129557784842214) q[9];
cx q[8],q[9];
ry(-0.5607732619363297) q[9];
ry(-0.604425656030816) q[10];
cx q[9],q[10];
ry(2.1753263601994184) q[9];
ry(-1.534199352279343) q[10];
cx q[9],q[10];
ry(3.1324006832664693) q[10];
ry(-1.0928429729935951) q[11];
cx q[10],q[11];
ry(0.8810990683894114) q[10];
ry(-2.2930767112028416) q[11];
cx q[10],q[11];
ry(0.22198898792447386) q[11];
ry(-0.40999034876687485) q[12];
cx q[11],q[12];
ry(0.01637058567763816) q[11];
ry(3.140407059333933) q[12];
cx q[11],q[12];
ry(1.5342166012330776) q[12];
ry(-1.2723104896731314) q[13];
cx q[12],q[13];
ry(-2.79514308132013) q[12];
ry(-0.025281241449144315) q[13];
cx q[12],q[13];
ry(1.8632816578975808) q[13];
ry(2.0392325628662267) q[14];
cx q[13],q[14];
ry(-2.739880282392066) q[13];
ry(1.1718225013956056) q[14];
cx q[13],q[14];
ry(-2.3050395574743034) q[14];
ry(-2.5292387492341466) q[15];
cx q[14],q[15];
ry(-1.3360851532285478) q[14];
ry(-1.616491405449055) q[15];
cx q[14],q[15];
ry(-1.8482567355425497) q[0];
ry(1.3232951649998679) q[1];
cx q[0],q[1];
ry(-1.3219183037765778) q[0];
ry(-1.519450867283971) q[1];
cx q[0],q[1];
ry(1.7279434271513205) q[1];
ry(-1.4784326303125468) q[2];
cx q[1],q[2];
ry(-0.00701666477872929) q[1];
ry(1.9376021386724913) q[2];
cx q[1],q[2];
ry(-0.6717745246586371) q[2];
ry(2.75551296140703) q[3];
cx q[2],q[3];
ry(1.3901204061052086) q[2];
ry(3.127253013574785) q[3];
cx q[2],q[3];
ry(-2.504916451640211) q[3];
ry(-0.18167845413700864) q[4];
cx q[3],q[4];
ry(-1.3128453186050408) q[3];
ry(-0.31120931202611946) q[4];
cx q[3],q[4];
ry(2.1760776914169013) q[4];
ry(-3.1176207983329816) q[5];
cx q[4],q[5];
ry(0.1801533982448442) q[4];
ry(0.08820921099284007) q[5];
cx q[4],q[5];
ry(-1.0540686514014102) q[5];
ry(-2.686862489388511) q[6];
cx q[5],q[6];
ry(0.5891188668546441) q[5];
ry(1.5547739119482387) q[6];
cx q[5],q[6];
ry(-0.5867238701036763) q[6];
ry(-0.9662658123981509) q[7];
cx q[6],q[7];
ry(0.011856859413224896) q[6];
ry(3.141589592617312) q[7];
cx q[6],q[7];
ry(1.0486586915565688) q[7];
ry(-2.473619603097294) q[8];
cx q[7],q[8];
ry(0.00037941402325536444) q[7];
ry(-1.4156207277944226) q[8];
cx q[7],q[8];
ry(-1.957614195519627) q[8];
ry(0.9431235645936935) q[9];
cx q[8],q[9];
ry(-3.00448243241682) q[8];
ry(1.40250488892955) q[9];
cx q[8],q[9];
ry(3.136923336564867) q[9];
ry(-2.2667176164452627) q[10];
cx q[9],q[10];
ry(-0.7456024708997937) q[9];
ry(-0.05511070540657226) q[10];
cx q[9],q[10];
ry(2.8434030729334006) q[10];
ry(2.8535750160352196) q[11];
cx q[10],q[11];
ry(-1.2869973383640048) q[10];
ry(-1.140492350837218) q[11];
cx q[10],q[11];
ry(-2.131140214698836) q[11];
ry(-2.1716902826780498) q[12];
cx q[11],q[12];
ry(-3.1409688925175527) q[11];
ry(-3.1412173828423473) q[12];
cx q[11],q[12];
ry(-2.7184169451769145) q[12];
ry(-2.941813523717104) q[13];
cx q[12],q[13];
ry(-1.4903438496232608) q[12];
ry(-1.8708577777471929) q[13];
cx q[12],q[13];
ry(-2.763661059285228) q[13];
ry(2.3963250089519237) q[14];
cx q[13],q[14];
ry(0.6610384567079649) q[13];
ry(0.6481715129297116) q[14];
cx q[13],q[14];
ry(1.8424538323567612) q[14];
ry(-0.758278660665198) q[15];
cx q[14],q[15];
ry(-0.16120279909748916) q[14];
ry(3.0341956965425423) q[15];
cx q[14],q[15];
ry(-1.061697862914194) q[0];
ry(1.8417961374428735) q[1];
cx q[0],q[1];
ry(1.6178670795931678) q[0];
ry(-1.0407735152884783) q[1];
cx q[0],q[1];
ry(1.8542508357584075) q[1];
ry(0.3011424219685512) q[2];
cx q[1],q[2];
ry(-2.8844042943502486) q[1];
ry(-1.911742099438257) q[2];
cx q[1],q[2];
ry(0.8422395324378716) q[2];
ry(0.8083606462449948) q[3];
cx q[2],q[3];
ry(1.389131107454432) q[2];
ry(3.1304804046824035) q[3];
cx q[2],q[3];
ry(0.8764371132711427) q[3];
ry(0.41829902427806515) q[4];
cx q[3],q[4];
ry(-1.1878673728789664) q[3];
ry(-2.470913958095025) q[4];
cx q[3],q[4];
ry(1.145856112341204) q[4];
ry(-2.963372478069857) q[5];
cx q[4],q[5];
ry(-8.381596107209788e-05) q[4];
ry(0.00010435557491586489) q[5];
cx q[4],q[5];
ry(-0.1787741116034276) q[5];
ry(2.9880277354923934) q[6];
cx q[5],q[6];
ry(0.09550748024931364) q[5];
ry(-1.5536123180561925) q[6];
cx q[5],q[6];
ry(0.06525851841801096) q[6];
ry(3.068432892993277) q[7];
cx q[6],q[7];
ry(-0.39413871530095346) q[6];
ry(3.1379777934132096) q[7];
cx q[6],q[7];
ry(1.427216526455176) q[7];
ry(2.4694042181927465) q[8];
cx q[7],q[8];
ry(3.1414090463100166) q[7];
ry(-0.326733941352237) q[8];
cx q[7],q[8];
ry(-2.3327573615120007) q[8];
ry(2.6044629635986962) q[9];
cx q[8],q[9];
ry(2.1487746275719877) q[8];
ry(0.20423530609751353) q[9];
cx q[8],q[9];
ry(-1.462980251748757) q[9];
ry(1.0071293085108328) q[10];
cx q[9],q[10];
ry(-3.1097836820340667) q[9];
ry(-3.0911827751067222) q[10];
cx q[9],q[10];
ry(1.7423388024481146) q[10];
ry(1.0596049983242786) q[11];
cx q[10],q[11];
ry(1.2423717812350752) q[10];
ry(1.6475701349105765) q[11];
cx q[10],q[11];
ry(0.9524768555956535) q[11];
ry(-0.43968155869307585) q[12];
cx q[11],q[12];
ry(1.6281949040713268) q[11];
ry(3.129384189352398) q[12];
cx q[11],q[12];
ry(-2.3951733674088787) q[12];
ry(2.682466739358367) q[13];
cx q[12],q[13];
ry(-2.324383616910389) q[12];
ry(1.7759306671452821) q[13];
cx q[12],q[13];
ry(1.7829134974042722) q[13];
ry(-0.5869097863288815) q[14];
cx q[13],q[14];
ry(1.1524382665055528) q[13];
ry(-2.049327540377377) q[14];
cx q[13],q[14];
ry(0.4277692172858024) q[14];
ry(1.3795758374889395) q[15];
cx q[14],q[15];
ry(1.5414271635535766) q[14];
ry(1.4428952342410792) q[15];
cx q[14],q[15];
ry(2.0964778455159316) q[0];
ry(1.6567811015546696) q[1];
cx q[0],q[1];
ry(-2.9665732258811923) q[0];
ry(-1.0389641549437112) q[1];
cx q[0],q[1];
ry(-0.32782385731337627) q[1];
ry(-2.630597520312723) q[2];
cx q[1],q[2];
ry(-0.34369658064763353) q[1];
ry(0.4079171813316509) q[2];
cx q[1],q[2];
ry(1.0921289861622414) q[2];
ry(-1.3634270243283269) q[3];
cx q[2],q[3];
ry(-0.6706639095530642) q[2];
ry(-1.1581565330759542) q[3];
cx q[2],q[3];
ry(-0.5788101629245936) q[3];
ry(0.9899846598032651) q[4];
cx q[3],q[4];
ry(0.019222154858530495) q[3];
ry(2.075829243974009) q[4];
cx q[3],q[4];
ry(-2.939599689592596) q[4];
ry(2.3440672208664175) q[5];
cx q[4],q[5];
ry(-0.00019718018372173241) q[4];
ry(3.0692813390273153) q[5];
cx q[4],q[5];
ry(-2.4742117527253424) q[5];
ry(-0.7491433739926059) q[6];
cx q[5],q[6];
ry(-3.0997821644990644) q[5];
ry(-3.137709027569073) q[6];
cx q[5],q[6];
ry(-2.861174715376265) q[6];
ry(-1.5664378026825851) q[7];
cx q[6],q[7];
ry(0.38828894750948884) q[6];
ry(1.4844132930703946) q[7];
cx q[6],q[7];
ry(1.2708631102696342) q[7];
ry(-0.5619386913232027) q[8];
cx q[7],q[8];
ry(0.42988743078292924) q[7];
ry(-0.045227816222219275) q[8];
cx q[7],q[8];
ry(0.654224819896327) q[8];
ry(-0.9381278729458957) q[9];
cx q[8],q[9];
ry(-0.046682495112466096) q[8];
ry(0.46989727787925606) q[9];
cx q[8],q[9];
ry(-0.8171557218340721) q[9];
ry(3.0636874897681996) q[10];
cx q[9],q[10];
ry(0.15660786462998513) q[9];
ry(0.9048678407345534) q[10];
cx q[9],q[10];
ry(-3.101887040712716) q[10];
ry(-0.3879388257068572) q[11];
cx q[10],q[11];
ry(-0.016800520885415615) q[10];
ry(2.9708602411382072) q[11];
cx q[10],q[11];
ry(-2.1238709105062417) q[11];
ry(-0.6756692614466724) q[12];
cx q[11],q[12];
ry(-0.00849887176332409) q[11];
ry(-3.1362101604817965) q[12];
cx q[11],q[12];
ry(-2.375576955150174) q[12];
ry(-0.7083539140335553) q[13];
cx q[12],q[13];
ry(-0.4061428903970441) q[12];
ry(-2.5373262950740485) q[13];
cx q[12],q[13];
ry(-0.21349281955963395) q[13];
ry(2.0714311455600853) q[14];
cx q[13],q[14];
ry(-0.20402556043898623) q[13];
ry(-2.953409932544373) q[14];
cx q[13],q[14];
ry(1.9769037937363398) q[14];
ry(-0.6768945227431771) q[15];
cx q[14],q[15];
ry(0.18028733683337173) q[14];
ry(-0.6589439161191428) q[15];
cx q[14],q[15];
ry(0.4479660329103199) q[0];
ry(-1.1347719917632162) q[1];
cx q[0],q[1];
ry(-1.5147486808382813) q[0];
ry(-0.7171037998173073) q[1];
cx q[0],q[1];
ry(1.563091880654735) q[1];
ry(1.4442411722520108) q[2];
cx q[1],q[2];
ry(3.115187799252351) q[1];
ry(-1.775090221613059) q[2];
cx q[1],q[2];
ry(2.1594279357722583) q[2];
ry(-1.9154313915633565) q[3];
cx q[2],q[3];
ry(-2.2213234356022906) q[2];
ry(2.4798995043418133) q[3];
cx q[2],q[3];
ry(-1.0844600385135763) q[3];
ry(-2.9892045396907863) q[4];
cx q[3],q[4];
ry(1.605220117890597) q[3];
ry(-0.5704270148678461) q[4];
cx q[3],q[4];
ry(-1.8455815917915859) q[4];
ry(-1.325117712696609) q[5];
cx q[4],q[5];
ry(3.1412351205746987) q[4];
ry(-2.68319791533344) q[5];
cx q[4],q[5];
ry(-1.9195829635674437) q[5];
ry(-1.568589889255041) q[6];
cx q[5],q[6];
ry(-2.4616916811796994) q[5];
ry(-0.006018573904639777) q[6];
cx q[5],q[6];
ry(-1.5720142174474656) q[6];
ry(-1.4886365455123545) q[7];
cx q[6],q[7];
ry(-2.0599829608370195) q[6];
ry(-1.5351835679575407) q[7];
cx q[6],q[7];
ry(2.4206553399816726) q[7];
ry(0.6482101803712759) q[8];
cx q[7],q[8];
ry(-0.9603499151195685) q[7];
ry(-0.006364531601089478) q[8];
cx q[7],q[8];
ry(2.617744270803628) q[8];
ry(-1.349116886287109) q[9];
cx q[8],q[9];
ry(1.7681851023474549) q[8];
ry(-2.289163009815814) q[9];
cx q[8],q[9];
ry(-2.2656497339477824) q[9];
ry(1.2874348396135775) q[10];
cx q[9],q[10];
ry(1.4771581526247486) q[9];
ry(-0.5729010986865709) q[10];
cx q[9],q[10];
ry(-1.7407869316827997) q[10];
ry(0.5342927189868885) q[11];
cx q[10],q[11];
ry(-0.8111451382470574) q[10];
ry(-1.9004514175877758) q[11];
cx q[10],q[11];
ry(2.35789405909483) q[11];
ry(1.5222707673751863) q[12];
cx q[11],q[12];
ry(-3.141464725239162) q[11];
ry(3.1392446063471118) q[12];
cx q[11],q[12];
ry(-2.5653902211479926) q[12];
ry(-0.6085621219217184) q[13];
cx q[12],q[13];
ry(1.7451439247953422) q[12];
ry(3.049454259488063) q[13];
cx q[12],q[13];
ry(-1.8713694140990214) q[13];
ry(-0.6181436688248502) q[14];
cx q[13],q[14];
ry(-2.417531374264877) q[13];
ry(0.29482052359258937) q[14];
cx q[13],q[14];
ry(1.9227595715172003) q[14];
ry(-1.742883143669073) q[15];
cx q[14],q[15];
ry(2.261274238447327) q[14];
ry(-1.9026534286418346) q[15];
cx q[14],q[15];
ry(-0.8350302586012974) q[0];
ry(1.4300439632156203) q[1];
cx q[0],q[1];
ry(-1.614915255832712) q[0];
ry(0.10817177273352684) q[1];
cx q[0],q[1];
ry(-2.5309051275224426) q[1];
ry(1.2241650471516898) q[2];
cx q[1],q[2];
ry(-1.044522223381765) q[1];
ry(0.9850190638386467) q[2];
cx q[1],q[2];
ry(-2.5795265373020153) q[2];
ry(0.46495317132293484) q[3];
cx q[2],q[3];
ry(-0.4654127113312145) q[2];
ry(0.4705287538263514) q[3];
cx q[2],q[3];
ry(-0.4521971921152135) q[3];
ry(1.6457239701433417) q[4];
cx q[3],q[4];
ry(-1.0180596893851308) q[3];
ry(-1.0673611329919028) q[4];
cx q[3],q[4];
ry(0.7368256838583163) q[4];
ry(2.623593258862266) q[5];
cx q[4],q[5];
ry(-0.03946574223256771) q[4];
ry(3.096474993317643) q[5];
cx q[4],q[5];
ry(-2.2546926291126015) q[5];
ry(-2.2068014561902354) q[6];
cx q[5],q[6];
ry(3.1384575740013614) q[5];
ry(-3.127738020255356) q[6];
cx q[5],q[6];
ry(2.529278863471569) q[6];
ry(1.4947034759483793) q[7];
cx q[6],q[7];
ry(1.6929461075453514) q[6];
ry(0.12406375162901093) q[7];
cx q[6],q[7];
ry(2.2961726834212906) q[7];
ry(-2.3573858793774742) q[8];
cx q[7],q[8];
ry(-0.43348020656296216) q[7];
ry(0.8317062522368008) q[8];
cx q[7],q[8];
ry(-2.064078746880188) q[8];
ry(-2.0719709633433965) q[9];
cx q[8],q[9];
ry(2.0875923648896775) q[8];
ry(0.09333262409145071) q[9];
cx q[8],q[9];
ry(0.42864688519800137) q[9];
ry(0.424385943216115) q[10];
cx q[9],q[10];
ry(2.995067782847631) q[9];
ry(-0.15369075008377117) q[10];
cx q[9],q[10];
ry(0.3691698553283249) q[10];
ry(2.2138570753012266) q[11];
cx q[10],q[11];
ry(-1.8860048011257518) q[10];
ry(3.1410300991442583) q[11];
cx q[10],q[11];
ry(2.457261876525999) q[11];
ry(1.5749160604031003) q[12];
cx q[11],q[12];
ry(0.0006954977950917858) q[11];
ry(3.138180365678515) q[12];
cx q[11],q[12];
ry(-1.946510013091641) q[12];
ry(1.4197477054967924) q[13];
cx q[12],q[13];
ry(2.8978202158666155) q[12];
ry(0.23754520955911396) q[13];
cx q[12],q[13];
ry(-1.1567405324221112) q[13];
ry(-1.39259483537454) q[14];
cx q[13],q[14];
ry(2.252589054030597) q[13];
ry(-2.216461128754304) q[14];
cx q[13],q[14];
ry(-2.066618454145016) q[14];
ry(-1.7273142001619994) q[15];
cx q[14],q[15];
ry(-0.39468061031946783) q[14];
ry(2.044118282432872) q[15];
cx q[14],q[15];
ry(1.7606103029419649) q[0];
ry(-1.7220085009171537) q[1];
cx q[0],q[1];
ry(-2.8561416376122364) q[0];
ry(2.7491354126203387) q[1];
cx q[0],q[1];
ry(-2.6237369649595683) q[1];
ry(-0.09230679924244267) q[2];
cx q[1],q[2];
ry(0.2913294106732507) q[1];
ry(1.6815455599720321) q[2];
cx q[1],q[2];
ry(0.7621026288989725) q[2];
ry(-1.5068040187695972) q[3];
cx q[2],q[3];
ry(1.515961825757936) q[2];
ry(2.932723618121121) q[3];
cx q[2],q[3];
ry(-1.262193636846737) q[3];
ry(0.6061956077818621) q[4];
cx q[3],q[4];
ry(-1.0977717556337476) q[3];
ry(1.2711465206319796) q[4];
cx q[3],q[4];
ry(2.3140284882677946) q[4];
ry(-1.6394566500163665) q[5];
cx q[4],q[5];
ry(-0.05281022237867372) q[4];
ry(3.107258015495195) q[5];
cx q[4],q[5];
ry(0.6824474158941402) q[5];
ry(-1.8397303797628863) q[6];
cx q[5],q[6];
ry(-0.06837382241954747) q[5];
ry(-2.709307657434865) q[6];
cx q[5],q[6];
ry(-1.8435168929406573) q[6];
ry(0.5319714196419204) q[7];
cx q[6],q[7];
ry(-3.1415775910442534) q[6];
ry(-0.000892085303035195) q[7];
cx q[6],q[7];
ry(-2.452502333906949) q[7];
ry(-0.9123741984255943) q[8];
cx q[7],q[8];
ry(-0.06630005399247718) q[7];
ry(-0.922761116046272) q[8];
cx q[7],q[8];
ry(1.9421240217694549) q[8];
ry(-1.4262842602823715) q[9];
cx q[8],q[9];
ry(-2.474459499509668) q[8];
ry(0.4690056099444701) q[9];
cx q[8],q[9];
ry(-2.4162205459487374) q[9];
ry(2.6041144695312894) q[10];
cx q[9],q[10];
ry(0.011544061464605177) q[9];
ry(3.0768304031768468) q[10];
cx q[9],q[10];
ry(-2.9056143497939786) q[10];
ry(2.4648026786818527) q[11];
cx q[10],q[11];
ry(1.882250875823486) q[10];
ry(-1.6077594602157104) q[11];
cx q[10],q[11];
ry(2.7278860890034546) q[11];
ry(-0.7057294385779939) q[12];
cx q[11],q[12];
ry(0.0014424111486395221) q[11];
ry(-3.124945903625284) q[12];
cx q[11],q[12];
ry(0.21563959576687972) q[12];
ry(-1.1395227830922954) q[13];
cx q[12],q[13];
ry(2.2846893987052015) q[12];
ry(-2.8925000153573928) q[13];
cx q[12],q[13];
ry(-0.6250013123113058) q[13];
ry(2.7327447265098916) q[14];
cx q[13],q[14];
ry(2.1242454399416744) q[13];
ry(0.5000707580142896) q[14];
cx q[13],q[14];
ry(-0.9908928660070364) q[14];
ry(-0.6092863547152509) q[15];
cx q[14],q[15];
ry(1.9128908368795772) q[14];
ry(-2.892791766997124) q[15];
cx q[14],q[15];
ry(0.9902505571607394) q[0];
ry(1.767067293296213) q[1];
cx q[0],q[1];
ry(-0.15608178077280055) q[0];
ry(2.5330546466982313) q[1];
cx q[0],q[1];
ry(0.1652308094268875) q[1];
ry(-0.6657074207744474) q[2];
cx q[1],q[2];
ry(0.9901550979829818) q[1];
ry(-2.3488456491969494) q[2];
cx q[1],q[2];
ry(0.3179119427036117) q[2];
ry(0.6969591194996353) q[3];
cx q[2],q[3];
ry(-1.7189813483912575) q[2];
ry(-2.500165519198156) q[3];
cx q[2],q[3];
ry(-1.3591318229859546) q[3];
ry(-1.2832232410375246) q[4];
cx q[3],q[4];
ry(0.2489913952017826) q[3];
ry(0.5468617562524468) q[4];
cx q[3],q[4];
ry(1.2846674567898522) q[4];
ry(2.6264515279242873) q[5];
cx q[4],q[5];
ry(-0.5358477216303711) q[4];
ry(-0.7167867446509488) q[5];
cx q[4],q[5];
ry(1.9952058564802517) q[5];
ry(-1.7333014998142202) q[6];
cx q[5],q[6];
ry(3.1277375080468843) q[5];
ry(-0.24881512086374366) q[6];
cx q[5],q[6];
ry(-0.0004338731477147399) q[6];
ry(0.5553906605701319) q[7];
cx q[6],q[7];
ry(-0.03255434582645833) q[6];
ry(-3.1318036425048317) q[7];
cx q[6],q[7];
ry(-0.9155866434046455) q[7];
ry(-0.3267486776378356) q[8];
cx q[7],q[8];
ry(0.027703226376426242) q[7];
ry(-0.12575352347992913) q[8];
cx q[7],q[8];
ry(1.813690720046867) q[8];
ry(0.22712254996552517) q[9];
cx q[8],q[9];
ry(-0.25859643295401735) q[8];
ry(2.55916102972296) q[9];
cx q[8],q[9];
ry(-1.3227935058075335) q[9];
ry(2.339799161254191) q[10];
cx q[9],q[10];
ry(-2.635034469901894) q[9];
ry(1.9997116210563646) q[10];
cx q[9],q[10];
ry(-1.2258338675422726) q[10];
ry(-1.176939641044732) q[11];
cx q[10],q[11];
ry(2.9552406822171244) q[10];
ry(-1.3225245414420534) q[11];
cx q[10],q[11];
ry(-1.453566356983984) q[11];
ry(-1.369411797939966) q[12];
cx q[11],q[12];
ry(1.4447943250187292) q[11];
ry(0.02512009986902175) q[12];
cx q[11],q[12];
ry(-0.09927052020181013) q[12];
ry(0.5046668831860989) q[13];
cx q[12],q[13];
ry(-3.135600630079293) q[12];
ry(-3.1370004435727505) q[13];
cx q[12],q[13];
ry(-2.617848499126749) q[13];
ry(-2.8866700936375542) q[14];
cx q[13],q[14];
ry(1.2088124108273153) q[13];
ry(-0.7600015842212474) q[14];
cx q[13],q[14];
ry(2.528799182989561) q[14];
ry(0.4941101763265132) q[15];
cx q[14],q[15];
ry(-1.008872446238675) q[14];
ry(1.9876108468429479) q[15];
cx q[14],q[15];
ry(-0.14760354530957726) q[0];
ry(-2.89894275421779) q[1];
cx q[0],q[1];
ry(0.8758526911231598) q[0];
ry(-1.5681254383198642) q[1];
cx q[0],q[1];
ry(0.49139190851253556) q[1];
ry(-2.7705073225848906) q[2];
cx q[1],q[2];
ry(-0.32040806382139847) q[1];
ry(0.11357662501586407) q[2];
cx q[1],q[2];
ry(0.03233582692988128) q[2];
ry(-1.3159413066962924) q[3];
cx q[2],q[3];
ry(-1.631248324805589) q[2];
ry(-2.605012914145784) q[3];
cx q[2],q[3];
ry(1.8169390424898544) q[3];
ry(1.926866124863821) q[4];
cx q[3],q[4];
ry(-0.014008223347892645) q[3];
ry(3.1304885361050703) q[4];
cx q[3],q[4];
ry(-0.6299631115325036) q[4];
ry(1.0332771207453098) q[5];
cx q[4],q[5];
ry(0.7863596434653566) q[4];
ry(2.4151222052694608) q[5];
cx q[4],q[5];
ry(-2.46985897232991) q[5];
ry(2.9180064847402853) q[6];
cx q[5],q[6];
ry(1.919411931088535) q[5];
ry(-2.8538449853113472) q[6];
cx q[5],q[6];
ry(1.204788542672011) q[6];
ry(1.876290687464276) q[7];
cx q[6],q[7];
ry(2.222902041867702) q[6];
ry(0.009096233998739045) q[7];
cx q[6],q[7];
ry(0.09820374815015853) q[7];
ry(2.762960889903011) q[8];
cx q[7],q[8];
ry(3.1413405696701577) q[7];
ry(0.002955707971333932) q[8];
cx q[7],q[8];
ry(1.354940151614829) q[8];
ry(1.3682134622633517) q[9];
cx q[8],q[9];
ry(-0.6570724195403993) q[8];
ry(0.27702961680846805) q[9];
cx q[8],q[9];
ry(-1.9757871057189185) q[9];
ry(-1.489675498290655) q[10];
cx q[9],q[10];
ry(-1.8556562051671124) q[9];
ry(-0.2936231434757843) q[10];
cx q[9],q[10];
ry(-1.8462734547300044) q[10];
ry(2.1736980677021425) q[11];
cx q[10],q[11];
ry(3.081202149389247) q[10];
ry(-0.7689405761838474) q[11];
cx q[10],q[11];
ry(-0.9942569104854667) q[11];
ry(1.249006903981728) q[12];
cx q[11],q[12];
ry(2.0528068448229337) q[11];
ry(0.07291123872285254) q[12];
cx q[11],q[12];
ry(-1.0445624451277986) q[12];
ry(-0.6262641602505266) q[13];
cx q[12],q[13];
ry(-0.4841452138579738) q[12];
ry(-2.008469707480504) q[13];
cx q[12],q[13];
ry(-3.022596362128903) q[13];
ry(-2.148278446132401) q[14];
cx q[13],q[14];
ry(-2.595926736907166) q[13];
ry(0.010096965803187499) q[14];
cx q[13],q[14];
ry(-0.3359285084639573) q[14];
ry(1.6718670693116948) q[15];
cx q[14],q[15];
ry(-2.2459653270576423) q[14];
ry(-2.3985125235754707) q[15];
cx q[14],q[15];
ry(2.502178965824099) q[0];
ry(-3.130430215962959) q[1];
cx q[0],q[1];
ry(-1.7317575577737871) q[0];
ry(1.715550113202549) q[1];
cx q[0],q[1];
ry(-1.0579313080052932) q[1];
ry(-2.176831904404834) q[2];
cx q[1],q[2];
ry(0.3549451482165757) q[1];
ry(0.24165497201894492) q[2];
cx q[1],q[2];
ry(-0.3794602971882587) q[2];
ry(0.01290998771229468) q[3];
cx q[2],q[3];
ry(0.19882064068554023) q[2];
ry(-3.0414121319054432) q[3];
cx q[2],q[3];
ry(3.0350911860465466) q[3];
ry(0.8150855997281408) q[4];
cx q[3],q[4];
ry(0.009299014569124074) q[3];
ry(0.0028886834750814216) q[4];
cx q[3],q[4];
ry(-3.0994240700281965) q[4];
ry(1.4501693610201116) q[5];
cx q[4],q[5];
ry(-3.1105941035910907) q[4];
ry(-2.0642798249881835) q[5];
cx q[4],q[5];
ry(-2.6720600781320867) q[5];
ry(-1.2738099198040382) q[6];
cx q[5],q[6];
ry(-1.6420936660979952) q[5];
ry(-2.992106757449961) q[6];
cx q[5],q[6];
ry(-0.5441812464613092) q[6];
ry(3.0457462859801105) q[7];
cx q[6],q[7];
ry(1.2583272560138414) q[6];
ry(0.014286613504851609) q[7];
cx q[6],q[7];
ry(-0.38437512126281437) q[7];
ry(2.7577425893226035) q[8];
cx q[7],q[8];
ry(3.1340629290960855) q[7];
ry(1.5088504152454387) q[8];
cx q[7],q[8];
ry(0.1199594715258879) q[8];
ry(0.5232325712483252) q[9];
cx q[8],q[9];
ry(-1.2228101645649259) q[8];
ry(-3.134230680928294) q[9];
cx q[8],q[9];
ry(-1.166416659111352) q[9];
ry(-0.6958890383538133) q[10];
cx q[9],q[10];
ry(-0.0465939711358797) q[9];
ry(-0.8999919410841158) q[10];
cx q[9],q[10];
ry(-0.3799588121158609) q[10];
ry(0.4362959221336311) q[11];
cx q[10],q[11];
ry(1.3528272473984178) q[10];
ry(-0.05223109961572179) q[11];
cx q[10],q[11];
ry(-0.1479103287549138) q[11];
ry(2.1349663383657225) q[12];
cx q[11],q[12];
ry(-1.9159958266620798) q[11];
ry(0.05807720410316897) q[12];
cx q[11],q[12];
ry(2.0466540542098546) q[12];
ry(1.1726645804394438) q[13];
cx q[12],q[13];
ry(-3.1308660412547566) q[12];
ry(-0.30022655496738526) q[13];
cx q[12],q[13];
ry(0.9558134508618783) q[13];
ry(-2.7816824807401273) q[14];
cx q[13],q[14];
ry(-0.2095806503547335) q[13];
ry(0.05186061979110949) q[14];
cx q[13],q[14];
ry(2.561412888712633) q[14];
ry(-0.5302423257586533) q[15];
cx q[14],q[15];
ry(0.8394360058608932) q[14];
ry(-2.8592367928643116) q[15];
cx q[14],q[15];
ry(1.2019704691306128) q[0];
ry(-1.8801485539260046) q[1];
cx q[0],q[1];
ry(2.1388908449953017) q[0];
ry(2.629396404977356) q[1];
cx q[0],q[1];
ry(-2.1427058808206425) q[1];
ry(0.22935316035985487) q[2];
cx q[1],q[2];
ry(-2.416273817147784) q[1];
ry(-2.843563258933885) q[2];
cx q[1],q[2];
ry(-2.232095142487312) q[2];
ry(1.483507282189624) q[3];
cx q[2],q[3];
ry(1.4623733276146083) q[2];
ry(2.929413536261606) q[3];
cx q[2],q[3];
ry(1.0142080855789484) q[3];
ry(-1.209105151372598) q[4];
cx q[3],q[4];
ry(0.006808649547029866) q[3];
ry(3.1363859002322725) q[4];
cx q[3],q[4];
ry(-0.015301555040429626) q[4];
ry(2.724659008009983) q[5];
cx q[4],q[5];
ry(2.9579988713936864) q[4];
ry(-1.3770617300253871) q[5];
cx q[4],q[5];
ry(0.8840751160588978) q[5];
ry(0.21627105124089047) q[6];
cx q[5],q[6];
ry(-2.539938721746632) q[5];
ry(2.858718876537617) q[6];
cx q[5],q[6];
ry(0.728678491581249) q[6];
ry(-0.5902723840845505) q[7];
cx q[6],q[7];
ry(3.1150388719176667) q[6];
ry(0.0165526462829213) q[7];
cx q[6],q[7];
ry(-0.958800167774454) q[7];
ry(-1.4971150326563718) q[8];
cx q[7],q[8];
ry(-3.087561339782097) q[7];
ry(-2.7492964978996257) q[8];
cx q[7],q[8];
ry(0.4442164057022264) q[8];
ry(1.3566355898882787) q[9];
cx q[8],q[9];
ry(3.015453235748365) q[8];
ry(3.1286390790564953) q[9];
cx q[8],q[9];
ry(-1.6632932548091819) q[9];
ry(0.9683923059683588) q[10];
cx q[9],q[10];
ry(3.122338313608798) q[9];
ry(2.4775942142961154) q[10];
cx q[9],q[10];
ry(2.47092506552947) q[10];
ry(0.056176990742845184) q[11];
cx q[10],q[11];
ry(-2.5005852489675946) q[10];
ry(3.02505896339835) q[11];
cx q[10],q[11];
ry(1.3008158365873068) q[11];
ry(-1.9087193224866184) q[12];
cx q[11],q[12];
ry(-1.2017516664745376) q[11];
ry(0.2201435444592391) q[12];
cx q[11],q[12];
ry(-0.29036005202721277) q[12];
ry(2.618673269444814) q[13];
cx q[12],q[13];
ry(-0.11998027893678653) q[12];
ry(0.2871360087102648) q[13];
cx q[12],q[13];
ry(-2.666994432171364) q[13];
ry(-2.773587930874156) q[14];
cx q[13],q[14];
ry(-0.1264615103387738) q[13];
ry(-3.0759485518666447) q[14];
cx q[13],q[14];
ry(0.3626236008847856) q[14];
ry(-0.7343212088372656) q[15];
cx q[14],q[15];
ry(1.5847036837136836) q[14];
ry(-1.3127119566429997) q[15];
cx q[14],q[15];
ry(0.5446883108554499) q[0];
ry(2.483892815196075) q[1];
cx q[0],q[1];
ry(-1.4616239183139497) q[0];
ry(2.9503343933973207) q[1];
cx q[0],q[1];
ry(2.0268727417557244) q[1];
ry(-2.1388408568433004) q[2];
cx q[1],q[2];
ry(-3.136521902050319) q[1];
ry(0.1273890664221066) q[2];
cx q[1],q[2];
ry(1.2684160494156682) q[2];
ry(-0.5607153107180167) q[3];
cx q[2],q[3];
ry(2.2263326619967163) q[2];
ry(1.3986720639185382) q[3];
cx q[2],q[3];
ry(2.5142445090305814) q[3];
ry(-1.9386226710105643) q[4];
cx q[3],q[4];
ry(-0.008066422791408913) q[3];
ry(2.6861349761782085) q[4];
cx q[3],q[4];
ry(0.6372316602009005) q[4];
ry(-2.1981470960229728) q[5];
cx q[4],q[5];
ry(-1.2082301573925085) q[4];
ry(0.465564774291801) q[5];
cx q[4],q[5];
ry(-1.0707638623288318) q[5];
ry(-2.643517377936877) q[6];
cx q[5],q[6];
ry(-2.967480546259029) q[5];
ry(0.0005233724078790304) q[6];
cx q[5],q[6];
ry(2.788895660943273) q[6];
ry(-2.257018485853496) q[7];
cx q[6],q[7];
ry(0.017105144170903586) q[6];
ry(0.007381325596937493) q[7];
cx q[6],q[7];
ry(-1.168549611635733) q[7];
ry(0.6929850070901343) q[8];
cx q[7],q[8];
ry(3.109376513558036) q[7];
ry(-2.5151676090466286) q[8];
cx q[7],q[8];
ry(2.18808315108969) q[8];
ry(1.8653812324019645) q[9];
cx q[8],q[9];
ry(-0.8735848283426151) q[8];
ry(-3.1127457088618162) q[9];
cx q[8],q[9];
ry(-0.39090116758013155) q[9];
ry(-2.792913427513228) q[10];
cx q[9],q[10];
ry(3.1189911546399545) q[9];
ry(-2.937739833477872) q[10];
cx q[9],q[10];
ry(-1.0939983609500552) q[10];
ry(-0.5970759427499495) q[11];
cx q[10],q[11];
ry(2.7730670681104876) q[10];
ry(2.135801774865779) q[11];
cx q[10],q[11];
ry(-2.2807155947381252) q[11];
ry(-0.1304027653240185) q[12];
cx q[11],q[12];
ry(3.090382461566129) q[11];
ry(0.0872829051595163) q[12];
cx q[11],q[12];
ry(1.6618208848494875) q[12];
ry(0.9769472771097892) q[13];
cx q[12],q[13];
ry(-0.43866641162329234) q[12];
ry(1.368733123154139) q[13];
cx q[12],q[13];
ry(-1.9849321924197085) q[13];
ry(-2.802907262837313) q[14];
cx q[13],q[14];
ry(1.0568558189776824) q[13];
ry(-0.042121481032046056) q[14];
cx q[13],q[14];
ry(-1.9835161398833843) q[14];
ry(-1.7164790983017006) q[15];
cx q[14],q[15];
ry(-2.876256294771042) q[14];
ry(-0.7267520026321277) q[15];
cx q[14],q[15];
ry(-0.3003526468613531) q[0];
ry(1.1475306761725945) q[1];
cx q[0],q[1];
ry(-1.780487443477468) q[0];
ry(2.229783368341836) q[1];
cx q[0],q[1];
ry(-1.75629118780888) q[1];
ry(-3.0307419188851794) q[2];
cx q[1],q[2];
ry(3.0799152989723892) q[1];
ry(0.026255331826191834) q[2];
cx q[1],q[2];
ry(1.8463598174234965) q[2];
ry(-1.2590181629277613) q[3];
cx q[2],q[3];
ry(-0.028809785005521782) q[2];
ry(-1.0276525121680535) q[3];
cx q[2],q[3];
ry(-2.7619539962561803) q[3];
ry(1.2242450879825542) q[4];
cx q[3],q[4];
ry(-3.141334332657775) q[3];
ry(-0.0650804530729232) q[4];
cx q[3],q[4];
ry(-1.7322749429115294) q[4];
ry(1.1708588864999683) q[5];
cx q[4],q[5];
ry(-2.541625269731728) q[4];
ry(-0.7554146412853235) q[5];
cx q[4],q[5];
ry(2.6837121734841625) q[5];
ry(2.0498937451764023) q[6];
cx q[5],q[6];
ry(-1.5801640190829696) q[5];
ry(-3.1215403716497927) q[6];
cx q[5],q[6];
ry(-1.6993287112150102) q[6];
ry(-1.10665192085244) q[7];
cx q[6],q[7];
ry(0.04348684133903813) q[6];
ry(3.128356986222815) q[7];
cx q[6],q[7];
ry(-1.9337537497621904) q[7];
ry(-2.759301629316215) q[8];
cx q[7],q[8];
ry(-3.1388426349829217) q[7];
ry(-0.5881784210288413) q[8];
cx q[7],q[8];
ry(-0.2861011380390952) q[8];
ry(0.03912408774675449) q[9];
cx q[8],q[9];
ry(-0.3533138599407461) q[8];
ry(-2.594849396089134) q[9];
cx q[8],q[9];
ry(-0.831096481820392) q[9];
ry(-0.8543890377195057) q[10];
cx q[9],q[10];
ry(0.003945769143220091) q[9];
ry(0.0015225276311090497) q[10];
cx q[9],q[10];
ry(2.926456664451556) q[10];
ry(0.7279556656931172) q[11];
cx q[10],q[11];
ry(-0.06257330640919923) q[10];
ry(-1.7527079308063396) q[11];
cx q[10],q[11];
ry(2.998338272378341) q[11];
ry(1.60927317128254) q[12];
cx q[11],q[12];
ry(3.1129178203704546) q[11];
ry(0.024212458285680805) q[12];
cx q[11],q[12];
ry(2.446836858101275) q[12];
ry(1.9316401701845318) q[13];
cx q[12],q[13];
ry(-1.0390359196053112) q[12];
ry(-2.3722670855382226) q[13];
cx q[12],q[13];
ry(0.10207886612372813) q[13];
ry(1.7522609835868126) q[14];
cx q[13],q[14];
ry(-2.4085561635232775) q[13];
ry(-2.1906691974210446) q[14];
cx q[13],q[14];
ry(-1.2436227913724986) q[14];
ry(-0.024311344115244434) q[15];
cx q[14],q[15];
ry(2.5679364740260113) q[14];
ry(3.099366323356349) q[15];
cx q[14],q[15];
ry(-2.5407705712630313) q[0];
ry(-1.5582370806363528) q[1];
cx q[0],q[1];
ry(-2.4945660180607447) q[0];
ry(1.0737941604515298) q[1];
cx q[0],q[1];
ry(-1.0190886319157932) q[1];
ry(-1.1734420726015573) q[2];
cx q[1],q[2];
ry(-0.08241854420500427) q[1];
ry(0.004264268288941827) q[2];
cx q[1],q[2];
ry(-2.8900563008545337) q[2];
ry(1.1433242902480432) q[3];
cx q[2],q[3];
ry(0.43948285494737593) q[2];
ry(3.0800913276982214) q[3];
cx q[2],q[3];
ry(1.527929210141612) q[3];
ry(2.789267194181088) q[4];
cx q[3],q[4];
ry(-3.1049531364913467) q[3];
ry(-0.527765255781512) q[4];
cx q[3],q[4];
ry(-0.1890135158799375) q[4];
ry(2.7983935978392442) q[5];
cx q[4],q[5];
ry(-3.051382599051634) q[4];
ry(1.48518949041015) q[5];
cx q[4],q[5];
ry(2.590409187724348) q[5];
ry(1.2216776981497817) q[6];
cx q[5],q[6];
ry(1.1706586803449621) q[5];
ry(3.1390748945525573) q[6];
cx q[5],q[6];
ry(-2.801464506148181) q[6];
ry(2.983378120039959) q[7];
cx q[6],q[7];
ry(0.32728646920453575) q[6];
ry(1.4270633360336278) q[7];
cx q[6],q[7];
ry(1.818406748226236) q[7];
ry(-0.048657222818855816) q[8];
cx q[7],q[8];
ry(3.079885312066629) q[7];
ry(-0.010914669514051845) q[8];
cx q[7],q[8];
ry(2.8158574939637884) q[8];
ry(-0.2542273136083839) q[9];
cx q[8],q[9];
ry(-1.766782689775674) q[8];
ry(-2.654597340426033) q[9];
cx q[8],q[9];
ry(-2.645163922468835) q[9];
ry(-1.6344660460082885) q[10];
cx q[9],q[10];
ry(3.0832851329326463) q[9];
ry(0.0006491894422223864) q[10];
cx q[9],q[10];
ry(-0.23734821136260736) q[10];
ry(-1.7886439780508465) q[11];
cx q[10],q[11];
ry(-2.294723772220255) q[10];
ry(-1.3808353319877162) q[11];
cx q[10],q[11];
ry(2.108489661523984) q[11];
ry(-2.002472418876925) q[12];
cx q[11],q[12];
ry(3.076750412470878) q[11];
ry(3.1240844107388788) q[12];
cx q[11],q[12];
ry(-1.4509522000446882) q[12];
ry(0.3816761742401412) q[13];
cx q[12],q[13];
ry(0.0327216672066859) q[12];
ry(-3.0884567093948503) q[13];
cx q[12],q[13];
ry(2.776181505571244) q[13];
ry(0.30771783101215405) q[14];
cx q[13],q[14];
ry(0.4537580190721116) q[13];
ry(-2.152140069087689) q[14];
cx q[13],q[14];
ry(2.8149730307052647) q[14];
ry(-2.043071415365314) q[15];
cx q[14],q[15];
ry(-3.036548969403131) q[14];
ry(-2.529442075543475) q[15];
cx q[14],q[15];
ry(-2.5126114223659153) q[0];
ry(0.40892591416897695) q[1];
cx q[0],q[1];
ry(-1.8389104498861422) q[0];
ry(-0.0013669536985982944) q[1];
cx q[0],q[1];
ry(1.7896754917577062) q[1];
ry(-0.6321682177198742) q[2];
cx q[1],q[2];
ry(3.089910788986782) q[1];
ry(3.0927878042766364) q[2];
cx q[1],q[2];
ry(-0.1442265568607164) q[2];
ry(3.0183325992234535) q[3];
cx q[2],q[3];
ry(1.1634117327331681) q[2];
ry(-0.5139338510764093) q[3];
cx q[2],q[3];
ry(0.16843159198791025) q[3];
ry(0.9129190961969843) q[4];
cx q[3],q[4];
ry(-3.1362898957340866) q[3];
ry(3.093872519332581) q[4];
cx q[3],q[4];
ry(-0.8184067979471019) q[4];
ry(1.4398408567374998) q[5];
cx q[4],q[5];
ry(-0.14044901992278677) q[4];
ry(-1.2459531700535829) q[5];
cx q[4],q[5];
ry(-0.5331601278434652) q[5];
ry(2.04496547481816) q[6];
cx q[5],q[6];
ry(-3.135489861516921) q[5];
ry(3.083119601302731) q[6];
cx q[5],q[6];
ry(0.4310302330665854) q[6];
ry(0.6267576455943838) q[7];
cx q[6],q[7];
ry(3.114369742497691) q[6];
ry(-1.5722989665966578) q[7];
cx q[6],q[7];
ry(-2.793740515414771) q[7];
ry(-1.7893899462007008) q[8];
cx q[7],q[8];
ry(-0.010669945175291806) q[7];
ry(0.0733335652259482) q[8];
cx q[7],q[8];
ry(0.010549158574440746) q[8];
ry(-1.1408929116916282) q[9];
cx q[8],q[9];
ry(2.7146392647607205) q[8];
ry(1.627625257874585) q[9];
cx q[8],q[9];
ry(1.0570033348645609) q[9];
ry(2.594299953047815) q[10];
cx q[9],q[10];
ry(0.0062893366893616645) q[9];
ry(3.0459501051335156) q[10];
cx q[9],q[10];
ry(0.07295775020602946) q[10];
ry(3.134802833359503) q[11];
cx q[10],q[11];
ry(-2.0333130934496593) q[10];
ry(-1.6242639113404211) q[11];
cx q[10],q[11];
ry(2.430991978633421) q[11];
ry(-1.4637201976073413) q[12];
cx q[11],q[12];
ry(-0.01587923407069939) q[11];
ry(-0.06533651433029113) q[12];
cx q[11],q[12];
ry(1.4724687156966) q[12];
ry(-1.269476294828865) q[13];
cx q[12],q[13];
ry(-2.897942932159138) q[12];
ry(2.5023643152362767) q[13];
cx q[12],q[13];
ry(1.1553391123218395) q[13];
ry(-0.7514732990858061) q[14];
cx q[13],q[14];
ry(-0.15541738065518906) q[13];
ry(3.1237338727956416) q[14];
cx q[13],q[14];
ry(-0.9778540979977721) q[14];
ry(-0.9257566530639161) q[15];
cx q[14],q[15];
ry(-1.172738082232672) q[14];
ry(-0.4956203143176072) q[15];
cx q[14],q[15];
ry(-3.07155441408483) q[0];
ry(0.6716667035546877) q[1];
ry(-2.6489536015119906) q[2];
ry(-0.1353256515198753) q[3];
ry(0.6417485615103768) q[4];
ry(-0.07450948106120588) q[5];
ry(0.1323464974679105) q[6];
ry(0.3091638788221287) q[7];
ry(2.995219711146637) q[8];
ry(-0.22329274354093812) q[9];
ry(0.4832594871459275) q[10];
ry(1.7195425470460712) q[11];
ry(2.8517455528810185) q[12];
ry(0.5513822635578993) q[13];
ry(-0.7666747759597854) q[14];
ry(0.3891862742912444) q[15];