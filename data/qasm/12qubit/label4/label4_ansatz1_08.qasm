OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.5419778575231469) q[0];
rz(1.4563914934043434) q[0];
ry(-3.0956620822915863) q[1];
rz(-2.5697218477453494) q[1];
ry(1.5754898078935398) q[2];
rz(-2.054944038113619) q[2];
ry(2.098723864700337) q[3];
rz(-0.00522139036406255) q[3];
ry(1.5715501194004826) q[4];
rz(-1.4603430869011467) q[4];
ry(-3.1121707750353953) q[5];
rz(2.6301865545883523) q[5];
ry(-1.5681968990379112) q[6];
rz(-0.7671508756150542) q[6];
ry(0.0018777386763604298) q[7];
rz(1.7716009770201842) q[7];
ry(-0.015422648323954569) q[8];
rz(1.445614374467348) q[8];
ry(-0.19345113045633378) q[9];
rz(0.003998709352232997) q[9];
ry(0.01461790594443151) q[10];
rz(-0.8121192909604573) q[10];
ry(1.5113684899922886) q[11];
rz(-0.5620755274154217) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.2507998935900648) q[0];
rz(-3.027155011654516) q[0];
ry(1.5703554083869278) q[1];
rz(0.009589147031322298) q[1];
ry(-3.1372402571540423) q[2];
rz(0.6311521686732448) q[2];
ry(-1.5709296911890807) q[3];
rz(0.03937583396261246) q[3];
ry(-0.014521960058776706) q[4];
rz(-0.027505857391184385) q[4];
ry(1.5715532596772315) q[5];
rz(-2.317016722118431) q[5];
ry(0.0036415432988503586) q[6];
rz(-0.8032602297236027) q[6];
ry(-3.0904420716082313) q[7];
rz(-0.692631589088653) q[7];
ry(-1.4980877100966385) q[8];
rz(-1.902545355723058) q[8];
ry(0.14809751398712276) q[9];
rz(-2.013886388333481) q[9];
ry(-1.5596282219875883) q[10];
rz(-1.8452783646498438) q[10];
ry(-2.8252571754751963) q[11];
rz(-0.04675142092651274) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5629922407410524) q[0];
rz(-1.3542317498994756) q[0];
ry(1.5291008802583168) q[1];
rz(0.4351800693739957) q[1];
ry(1.5720321301489009) q[2];
rz(-0.0015194211101734465) q[2];
ry(-2.6914334570122738) q[3];
rz(-2.8352979350534993) q[3];
ry(1.4997200671128024) q[4];
rz(-2.707498355467095) q[4];
ry(-1.6053894215874716) q[5];
rz(-1.1175339078945974) q[5];
ry(-0.8312841693772466) q[6];
rz(-0.00018467846898495566) q[6];
ry(-3.1081018464007797) q[7];
rz(-0.246320368469078) q[7];
ry(0.018780325222414622) q[8];
rz(-3.0606802618736197) q[8];
ry(0.3828727348814409) q[9];
rz(2.9461431344035267) q[9];
ry(-3.0730779066042007) q[10];
rz(-2.9738199903012914) q[10];
ry(2.482673287224612) q[11];
rz(1.490534353163718) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.091178728001805) q[0];
rz(-1.2341742286497102) q[0];
ry(1.4903284577787943) q[1];
rz(-3.0696865847202477) q[1];
ry(1.7171509813200831) q[2];
rz(1.0282788313642852) q[2];
ry(0.0018628089006968106) q[3];
rz(2.8693883871158574) q[3];
ry(-0.005358461749933662) q[4];
rz(1.8269353755063875) q[4];
ry(0.03691676679101353) q[5];
rz(0.9257223456301444) q[5];
ry(-1.551223324127979) q[6];
rz(-0.0007571825332360265) q[6];
ry(-1.6012575374363005) q[7];
rz(-1.5673779758816335) q[7];
ry(-2.8034582715424743) q[8];
rz(2.8929039901904354) q[8];
ry(-1.5966878773146096) q[9];
rz(0.5352201692614221) q[9];
ry(-2.8543018894525916) q[10];
rz(-1.588330080016213) q[10];
ry(-2.7776178101377673) q[11];
rz(1.50673714182542) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.1395791291359663) q[0];
rz(1.3770519923630764) q[0];
ry(-3.059898019148501) q[1];
rz(-1.457563734547979) q[1];
ry(-0.00048168181023885515) q[2];
rz(-1.0297867883748095) q[2];
ry(1.308414581531581) q[3];
rz(3.1326219871491605) q[3];
ry(0.1086971918425478) q[4];
rz(-0.7629342286441783) q[4];
ry(3.0856148412354663) q[5];
rz(-0.21325894521014455) q[5];
ry(1.536329158242317) q[6];
rz(0.0001573948752819884) q[6];
ry(-0.47859934376632834) q[7];
rz(-3.1308362889157837) q[7];
ry(0.3743677554532596) q[8];
rz(1.1600756783742785) q[8];
ry(-0.78118583292677) q[9];
rz(-1.9502325386645403) q[9];
ry(0.4632143018298125) q[10];
rz(0.8927814923062298) q[10];
ry(-2.271272951580917) q[11];
rz(1.612254542372625) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.9748036241300264) q[0];
rz(2.348912508285357) q[0];
ry(0.4236471311573604) q[1];
rz(-1.9048939907743512) q[1];
ry(-2.203128299664269) q[2];
rz(0.0059551104401340295) q[2];
ry(0.4684975637653297) q[3];
rz(0.060214683743758234) q[3];
ry(3.1247355095748417) q[4];
rz(1.579535363248491) q[4];
ry(-2.7632502697872976) q[5];
rz(0.32137738595975396) q[5];
ry(1.4693902749531569) q[6];
rz(3.122371617793632) q[6];
ry(-1.5562374178975718) q[7];
rz(2.7464188188880905) q[7];
ry(0.00024603499185360533) q[8];
rz(1.9815244576895532) q[8];
ry(-3.140943864677219) q[9];
rz(2.177209778722832) q[9];
ry(-3.1409832918688796) q[10];
rz(1.031844506325974) q[10];
ry(-1.5856245330177003) q[11];
rz(1.5228230257910207) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.1389375733571843) q[0];
rz(-0.7593342314614988) q[0];
ry(-0.08291578135514754) q[1];
rz(1.7867542827598313) q[1];
ry(2.942325207250622) q[2];
rz(0.01485711641457681) q[2];
ry(-0.19590892577253305) q[3];
rz(-3.097780427664302) q[3];
ry(1.6319056389908333) q[4];
rz(-0.13969401289599048) q[4];
ry(0.035343477773219334) q[5];
rz(-0.5156292757886108) q[5];
ry(1.5679066280898324) q[6];
rz(-1.2407914665257447) q[6];
ry(0.001461082608215104) q[7];
rz(2.12638842846951) q[7];
ry(-1.5730284120491875) q[8];
rz(-3.1415251732693004) q[8];
ry(2.487274590632151) q[9];
rz(-2.530347441530446) q[9];
ry(3.0252193909241214) q[10];
rz(-3.0680875041250304) q[10];
ry(-1.713473773678371) q[11];
rz(2.6075692346330306) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.0594315182135667) q[0];
rz(1.484569427460222) q[0];
ry(-0.8874426633277135) q[1];
rz(-1.5756907275226695) q[1];
ry(0.125203417985207) q[2];
rz(-3.108396252748509) q[2];
ry(3.1410859262773583) q[3];
rz(-2.583356758906648) q[3];
ry(-3.0948648386786415) q[4];
rz(-1.3704985928486102) q[4];
ry(0.0013369074402589014) q[5];
rz(-2.5305036502822116) q[5];
ry(-0.0004596997269192827) q[6];
rz(-0.4516299570714919) q[6];
ry(-8.3713059371604e-05) q[7];
rz(-1.73122632226845) q[7];
ry(1.448712663099009) q[8];
rz(-3.141461468769078) q[8];
ry(-1.5641884385272498) q[9];
rz(-0.0012477738579717013) q[9];
ry(0.00020838575926963784) q[10];
rz(3.108399724541893) q[10];
ry(-0.27834183155722153) q[11];
rz(0.04204257951166111) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.08381172540599115) q[0];
rz(0.8428144167790155) q[0];
ry(1.4392317102081718) q[1];
rz(-1.8418143282812294) q[1];
ry(1.6761933577672987) q[2];
rz(-1.5174860754363786) q[2];
ry(0.03169217762649596) q[3];
rz(0.44352118659915446) q[3];
ry(0.14536790568622013) q[4];
rz(-1.935591852230857) q[4];
ry(3.112737975913698) q[5];
rz(-1.6275709347393956) q[5];
ry(-0.019374529543947357) q[6];
rz(-1.449281745498861) q[6];
ry(0.3774075967891848) q[7];
rz(-3.13104685074702) q[7];
ry(-1.571121193550944) q[8];
rz(-0.4730088083586885) q[8];
ry(-1.5764698977340208) q[9];
rz(0.5479072138845399) q[9];
ry(-3.140376867603064) q[10];
rz(1.5066033341879703) q[10];
ry(2.937037064366176) q[11];
rz(-2.6549625623544344) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.0011616957111026949) q[0];
rz(-1.1458830906778066) q[0];
ry(0.040111709911873496) q[1];
rz(-1.8906168992542078) q[1];
ry(0.00023044349394130137) q[2];
rz(-2.7137863505538053) q[2];
ry(3.141302834905151) q[3];
rz(-2.2423567788308345) q[3];
ry(0.32870933233978317) q[4];
rz(-1.6999420723311887) q[4];
ry(-0.00021254517718372057) q[5];
rz(-1.094369480097594) q[5];
ry(1.456198471771056) q[6];
rz(1.870036557936344) q[6];
ry(0.25153743110917715) q[7];
rz(-3.102523219963353) q[7];
ry(3.1062781495951883) q[8];
rz(-1.9704757886993967) q[8];
ry(0.0038353588076072776) q[9];
rz(-0.9428842072105925) q[9];
ry(2.987434344170865) q[10];
rz(1.5862897215710268) q[10];
ry(-1.2949835581902303) q[11];
rz(-2.7701340508861443) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.492913426548752) q[0];
rz(-1.5457690087051001) q[0];
ry(0.14455370960734876) q[1];
rz(-0.9848892659172057) q[1];
ry(-3.09551895161467) q[2];
rz(0.4755185045200872) q[2];
ry(-1.539067250163676) q[3];
rz(-1.5731859052297927) q[3];
ry(-0.06338954176019602) q[4];
rz(0.1498027317704693) q[4];
ry(1.5716069980945393) q[5];
rz(1.5693491251153195) q[5];
ry(-3.141418043738048) q[6];
rz(-2.8426995716930703) q[6];
ry(-0.022000680680898027) q[7];
rz(1.521133943663008) q[7];
ry(-3.1414291579801263) q[8];
rz(0.07285913410014011) q[8];
ry(0.004210456944515606) q[9];
rz(1.9657924029947726) q[9];
ry(-1.571116590912828) q[10];
rz(-1.570831240858552) q[10];
ry(-1.5591216471278917) q[11];
rz(1.255960612210699) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5699645161409785) q[0];
rz(-3.059222566597309) q[0];
ry(-1.573913547332718) q[1];
rz(-2.923578994146687) q[1];
ry(1.570349020775498) q[2];
rz(-3.0484009119874895) q[2];
ry(1.5710970245514104) q[3];
rz(-2.9241435064951666) q[3];
ry(-1.570641246708908) q[4];
rz(1.6637617571899543) q[4];
ry(-1.570637805292793) q[5];
rz(-1.3546647599004569) q[5];
ry(1.5707671133201886) q[6];
rz(1.6497852788836882) q[6];
ry(1.5735718844155413) q[7];
rz(-1.100797245333525) q[7];
ry(1.5546547319363948) q[8];
rz(0.10042298737764721) q[8];
ry(-1.5706404997165908) q[9];
rz(0.21884215582709735) q[9];
ry(-1.5707793443376976) q[10];
rz(0.09724539607690197) q[10];
ry(0.0036669201597874235) q[11];
rz(0.5925685879031816) q[11];