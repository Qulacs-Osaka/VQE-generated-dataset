OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-2.300147828761473) q[0];
ry(2.2291110587679706) q[1];
cx q[0],q[1];
ry(1.7750486230360476) q[0];
ry(1.063628464449283) q[1];
cx q[0],q[1];
ry(-2.4010418230646198) q[1];
ry(2.3541233129240013) q[2];
cx q[1],q[2];
ry(-3.092530083439257) q[1];
ry(3.0193670227940905) q[2];
cx q[1],q[2];
ry(-2.2317177676048106) q[2];
ry(1.1377075842907713) q[3];
cx q[2],q[3];
ry(2.528142863849498) q[2];
ry(1.1420125008790734) q[3];
cx q[2],q[3];
ry(1.573097906990979) q[3];
ry(1.9401385505292787) q[4];
cx q[3],q[4];
ry(-0.23923171455785486) q[3];
ry(1.0561906616248256) q[4];
cx q[3],q[4];
ry(1.2938169840790632) q[4];
ry(1.431053903075776) q[5];
cx q[4],q[5];
ry(-2.825106321144069) q[4];
ry(0.15757016175156302) q[5];
cx q[4],q[5];
ry(2.9651494400135423) q[5];
ry(0.922668387478639) q[6];
cx q[5],q[6];
ry(0.8449621186216392) q[5];
ry(0.09346462040894998) q[6];
cx q[5],q[6];
ry(1.033114734226162) q[6];
ry(3.0992792085462564) q[7];
cx q[6],q[7];
ry(1.1965001109656574) q[6];
ry(0.5089073867335524) q[7];
cx q[6],q[7];
ry(1.7108276603808499) q[0];
ry(-1.169518879061191) q[1];
cx q[0],q[1];
ry(-0.14541666879491455) q[0];
ry(-2.461676854672472) q[1];
cx q[0],q[1];
ry(0.8701191402946833) q[1];
ry(-0.023662586604320884) q[2];
cx q[1],q[2];
ry(-0.6877809478400492) q[1];
ry(1.4560825782745568) q[2];
cx q[1],q[2];
ry(-3.094309345268675) q[2];
ry(-0.5610150182604795) q[3];
cx q[2],q[3];
ry(2.412674042946997) q[2];
ry(-0.2015866566451173) q[3];
cx q[2],q[3];
ry(-0.17300462606142) q[3];
ry(0.4216634643011949) q[4];
cx q[3],q[4];
ry(0.05448811773671616) q[3];
ry(-2.5462629944705957) q[4];
cx q[3],q[4];
ry(2.1435245324476164) q[4];
ry(0.0625991634839642) q[5];
cx q[4],q[5];
ry(-1.200039159290733) q[4];
ry(2.65271796630168) q[5];
cx q[4],q[5];
ry(-1.883576237917361) q[5];
ry(1.58006744223423) q[6];
cx q[5],q[6];
ry(1.710243132184338) q[5];
ry(-2.8364170143152716) q[6];
cx q[5],q[6];
ry(-1.7748488183375208) q[6];
ry(0.033708585197386745) q[7];
cx q[6],q[7];
ry(2.4975133740501105) q[6];
ry(2.9595547895756926) q[7];
cx q[6],q[7];
ry(1.481979248646541) q[0];
ry(-0.18895780816101) q[1];
cx q[0],q[1];
ry(-1.4197747565180423) q[0];
ry(1.1423320575216325) q[1];
cx q[0],q[1];
ry(2.5829400328303445) q[1];
ry(0.9398855727650524) q[2];
cx q[1],q[2];
ry(0.9464239939599075) q[1];
ry(0.4084170594282914) q[2];
cx q[1],q[2];
ry(-1.145545046802524) q[2];
ry(1.574479201642468) q[3];
cx q[2],q[3];
ry(-2.5665132231889407) q[2];
ry(1.2107825758100228) q[3];
cx q[2],q[3];
ry(-2.3557226976658603) q[3];
ry(0.4138888832248302) q[4];
cx q[3],q[4];
ry(1.1486726414862496) q[3];
ry(1.0627176487125036) q[4];
cx q[3],q[4];
ry(1.2150157668136616) q[4];
ry(0.2730273524255926) q[5];
cx q[4],q[5];
ry(0.48331713400965004) q[4];
ry(0.7476417388122427) q[5];
cx q[4],q[5];
ry(2.0315780508311323) q[5];
ry(-0.5058628117758935) q[6];
cx q[5],q[6];
ry(-1.3712328888071692) q[5];
ry(-1.7577905887826402) q[6];
cx q[5],q[6];
ry(1.6208598132277334) q[6];
ry(-0.7098410784814017) q[7];
cx q[6],q[7];
ry(-2.8214406569724706) q[6];
ry(-3.101933290219598) q[7];
cx q[6],q[7];
ry(2.4526987721345734) q[0];
ry(1.8896634472058111) q[1];
cx q[0],q[1];
ry(-2.6392937187299497) q[0];
ry(2.745154856678949) q[1];
cx q[0],q[1];
ry(3.0391265474161386) q[1];
ry(2.789506445805363) q[2];
cx q[1],q[2];
ry(-0.24123083148871646) q[1];
ry(0.18656195101191564) q[2];
cx q[1],q[2];
ry(2.079453971276667) q[2];
ry(0.9181057308131598) q[3];
cx q[2],q[3];
ry(-0.31868231904597316) q[2];
ry(0.0031044609141650525) q[3];
cx q[2],q[3];
ry(0.9418465509821757) q[3];
ry(1.1923165825902942) q[4];
cx q[3],q[4];
ry(1.5883638218787381) q[3];
ry(1.6997160251719636) q[4];
cx q[3],q[4];
ry(-2.3190324338711212) q[4];
ry(-2.32728711783509) q[5];
cx q[4],q[5];
ry(-3.068456002936841) q[4];
ry(1.1416527247932595) q[5];
cx q[4],q[5];
ry(-0.7499273318151866) q[5];
ry(0.4027806721019408) q[6];
cx q[5],q[6];
ry(2.3124102320846287) q[5];
ry(-0.20698485491873875) q[6];
cx q[5],q[6];
ry(0.773115287446732) q[6];
ry(0.7409221116285734) q[7];
cx q[6],q[7];
ry(-1.7292662111264976) q[6];
ry(-1.3720357282688929) q[7];
cx q[6],q[7];
ry(-2.6975404057653805) q[0];
ry(2.3944948164419966) q[1];
cx q[0],q[1];
ry(2.3301468325473302) q[0];
ry(-1.5428295090526314) q[1];
cx q[0],q[1];
ry(1.9661233611205517) q[1];
ry(-1.5811432348096186) q[2];
cx q[1],q[2];
ry(1.1649674134370345) q[1];
ry(-1.6958748119535985) q[2];
cx q[1],q[2];
ry(-0.05849342099156881) q[2];
ry(0.4648073950668694) q[3];
cx q[2],q[3];
ry(-2.932296685190082) q[2];
ry(-0.16027589226701522) q[3];
cx q[2],q[3];
ry(-0.1281513803358873) q[3];
ry(-0.49849469199912394) q[4];
cx q[3],q[4];
ry(2.9862214482681297) q[3];
ry(1.048821992043285) q[4];
cx q[3],q[4];
ry(1.8020085274580122) q[4];
ry(3.0736677680352282) q[5];
cx q[4],q[5];
ry(2.080901650985527) q[4];
ry(-2.892498214789652) q[5];
cx q[4],q[5];
ry(0.26712658917257154) q[5];
ry(-2.9115313761144397) q[6];
cx q[5],q[6];
ry(-2.46347911208493) q[5];
ry(-1.1583590771003134) q[6];
cx q[5],q[6];
ry(-1.6841298799339883) q[6];
ry(-1.2362342889736189) q[7];
cx q[6],q[7];
ry(-1.630161048133874) q[6];
ry(0.23922492647613394) q[7];
cx q[6],q[7];
ry(-0.2153447511823003) q[0];
ry(1.9395772857075653) q[1];
cx q[0],q[1];
ry(-0.5672306891172842) q[0];
ry(2.4186918190926443) q[1];
cx q[0],q[1];
ry(2.2306105265600493) q[1];
ry(0.7084931615376853) q[2];
cx q[1],q[2];
ry(-1.4697758526279217) q[1];
ry(2.8166426025897033) q[2];
cx q[1],q[2];
ry(-3.0972940897406676) q[2];
ry(-0.38373991636702526) q[3];
cx q[2],q[3];
ry(1.6254765153558657) q[2];
ry(3.050424811987198) q[3];
cx q[2],q[3];
ry(-0.40375646533984266) q[3];
ry(2.8435893288023797) q[4];
cx q[3],q[4];
ry(-1.401176208955044) q[3];
ry(-0.21133035105399212) q[4];
cx q[3],q[4];
ry(-1.7588025142764216) q[4];
ry(2.0306102605240226) q[5];
cx q[4],q[5];
ry(-1.0391527796777147) q[4];
ry(1.3073942184745904) q[5];
cx q[4],q[5];
ry(0.4026884553750864) q[5];
ry(-3.0562376627004273) q[6];
cx q[5],q[6];
ry(1.4581167413718674) q[5];
ry(-0.654718821563783) q[6];
cx q[5],q[6];
ry(2.012460792944246) q[6];
ry(0.9122741134288564) q[7];
cx q[6],q[7];
ry(1.661462281279734) q[6];
ry(-2.684082748446279) q[7];
cx q[6],q[7];
ry(-1.5776866926581552) q[0];
ry(2.7469002424731292) q[1];
cx q[0],q[1];
ry(0.3459989734584976) q[0];
ry(2.936436303293025) q[1];
cx q[0],q[1];
ry(-0.8499765989101881) q[1];
ry(-3.0733448673846437) q[2];
cx q[1],q[2];
ry(2.3295590847300227) q[1];
ry(2.226574381507944) q[2];
cx q[1],q[2];
ry(0.43021360221897775) q[2];
ry(2.5567942553254155) q[3];
cx q[2],q[3];
ry(0.10832499096939951) q[2];
ry(-1.8313445124144623) q[3];
cx q[2],q[3];
ry(1.533145117907826) q[3];
ry(-1.3584321405963413) q[4];
cx q[3],q[4];
ry(-0.4039544078615121) q[3];
ry(1.595849435431167) q[4];
cx q[3],q[4];
ry(0.846530157922606) q[4];
ry(-2.248646743666795) q[5];
cx q[4],q[5];
ry(0.37595588012006687) q[4];
ry(-1.839356367237409) q[5];
cx q[4],q[5];
ry(3.051204137843044) q[5];
ry(2.3999574078722428) q[6];
cx q[5],q[6];
ry(-1.180080300543679) q[5];
ry(1.8974732741718559) q[6];
cx q[5],q[6];
ry(-2.1471455265567014) q[6];
ry(2.905405390693725) q[7];
cx q[6],q[7];
ry(-1.8997672608209095) q[6];
ry(-2.646565405486152) q[7];
cx q[6],q[7];
ry(-0.3498183220529304) q[0];
ry(0.10077135801979864) q[1];
cx q[0],q[1];
ry(2.9838650407399867) q[0];
ry(2.5100499534737906) q[1];
cx q[0],q[1];
ry(0.13848232793218465) q[1];
ry(2.3963079441128143) q[2];
cx q[1],q[2];
ry(0.2528537178865879) q[1];
ry(2.2110966328996877) q[2];
cx q[1],q[2];
ry(0.11305777724820575) q[2];
ry(-1.6888600733130747) q[3];
cx q[2],q[3];
ry(-1.3096884334677572) q[2];
ry(0.08972013233118006) q[3];
cx q[2],q[3];
ry(-1.3381952365664862) q[3];
ry(0.1253953632560068) q[4];
cx q[3],q[4];
ry(-0.008962117627032804) q[3];
ry(-1.8734757279433278) q[4];
cx q[3],q[4];
ry(2.2425700935510813) q[4];
ry(2.2657892228593246) q[5];
cx q[4],q[5];
ry(1.6837977276044938) q[4];
ry(-0.4415058625676618) q[5];
cx q[4],q[5];
ry(2.1168600153215094) q[5];
ry(0.5553980047187307) q[6];
cx q[5],q[6];
ry(0.11129185653828522) q[5];
ry(3.107301820183196) q[6];
cx q[5],q[6];
ry(-3.0940545879817023) q[6];
ry(-1.7559727306711088) q[7];
cx q[6],q[7];
ry(0.46921113023834327) q[6];
ry(-2.023061099760067) q[7];
cx q[6],q[7];
ry(-0.4234495676479738) q[0];
ry(-0.5455868547810201) q[1];
cx q[0],q[1];
ry(-2.9767986524035095) q[0];
ry(2.0439568065161033) q[1];
cx q[0],q[1];
ry(2.0819391660368542) q[1];
ry(1.2369720231915078) q[2];
cx q[1],q[2];
ry(1.25155392965698) q[1];
ry(-2.652946461535587) q[2];
cx q[1],q[2];
ry(2.3095171226178093) q[2];
ry(-0.6589098807659018) q[3];
cx q[2],q[3];
ry(-1.1927874504923226) q[2];
ry(1.9809509542963122) q[3];
cx q[2],q[3];
ry(1.18606348885188) q[3];
ry(2.6599662484008153) q[4];
cx q[3],q[4];
ry(-0.6190019111766621) q[3];
ry(-0.2891860739770804) q[4];
cx q[3],q[4];
ry(-2.891166060524591) q[4];
ry(-0.2658527798358755) q[5];
cx q[4],q[5];
ry(2.73290849706987) q[4];
ry(-1.3383650489793109) q[5];
cx q[4],q[5];
ry(-0.5241770361787321) q[5];
ry(2.165635279738521) q[6];
cx q[5],q[6];
ry(-1.3388910079394032) q[5];
ry(-1.6014039506487172) q[6];
cx q[5],q[6];
ry(-2.558197870887417) q[6];
ry(0.28216326946464926) q[7];
cx q[6],q[7];
ry(0.6236862376627448) q[6];
ry(3.12820335453244) q[7];
cx q[6],q[7];
ry(-3.109242296205597) q[0];
ry(-1.4331793003881488) q[1];
cx q[0],q[1];
ry(-2.8402961830721947) q[0];
ry(-0.12938261211697452) q[1];
cx q[0],q[1];
ry(-2.314134561624232) q[1];
ry(2.417436253836222) q[2];
cx q[1],q[2];
ry(-2.2699213909968057) q[1];
ry(0.4499927587615987) q[2];
cx q[1],q[2];
ry(-3.137818115346608) q[2];
ry(1.2278255198123258) q[3];
cx q[2],q[3];
ry(-2.883411200459247) q[2];
ry(-2.285207496149164) q[3];
cx q[2],q[3];
ry(-1.0594239083195516) q[3];
ry(2.2826388252953205) q[4];
cx q[3],q[4];
ry(2.0156029780745026) q[3];
ry(2.419342844642378) q[4];
cx q[3],q[4];
ry(-1.3427100511013013) q[4];
ry(-1.9558513889744435) q[5];
cx q[4],q[5];
ry(2.3860226986645845) q[4];
ry(-0.5799398703626304) q[5];
cx q[4],q[5];
ry(-2.7170177901480073) q[5];
ry(3.137345386412539) q[6];
cx q[5],q[6];
ry(3.076537984246942) q[5];
ry(2.971986589817711) q[6];
cx q[5],q[6];
ry(1.7614409988452335) q[6];
ry(0.7070577245541221) q[7];
cx q[6],q[7];
ry(-0.4467614845493209) q[6];
ry(-1.6098068522426578) q[7];
cx q[6],q[7];
ry(-0.4987028801939344) q[0];
ry(0.6040880206707735) q[1];
cx q[0],q[1];
ry(-1.3747289380050007) q[0];
ry(0.11493341576108432) q[1];
cx q[0],q[1];
ry(1.8641822867184228) q[1];
ry(1.2442858965309511) q[2];
cx q[1],q[2];
ry(3.0722113114176786) q[1];
ry(-2.3220916649858703) q[2];
cx q[1],q[2];
ry(-2.387704621444458) q[2];
ry(-1.823277722194659) q[3];
cx q[2],q[3];
ry(2.254327696601102) q[2];
ry(-1.4662679116118538) q[3];
cx q[2],q[3];
ry(-2.367441265761362) q[3];
ry(-1.7354936131142313) q[4];
cx q[3],q[4];
ry(1.8980940921372893) q[3];
ry(2.7011969133126565) q[4];
cx q[3],q[4];
ry(-0.9569054190118145) q[4];
ry(2.3953649373147656) q[5];
cx q[4],q[5];
ry(-0.2953375469586945) q[4];
ry(-1.304365554382278) q[5];
cx q[4],q[5];
ry(-1.0968024266447696) q[5];
ry(1.9026313464337319) q[6];
cx q[5],q[6];
ry(2.0164953630770457) q[5];
ry(-3.047682079932433) q[6];
cx q[5],q[6];
ry(-2.9621115707522736) q[6];
ry(1.745234504027675) q[7];
cx q[6],q[7];
ry(-2.535284125896069) q[6];
ry(2.7967684909123123) q[7];
cx q[6],q[7];
ry(0.6143787405703547) q[0];
ry(-2.1800233489947645) q[1];
cx q[0],q[1];
ry(0.20025379408216448) q[0];
ry(-2.672897695898671) q[1];
cx q[0],q[1];
ry(-1.89347395165837) q[1];
ry(-2.4493477980681773) q[2];
cx q[1],q[2];
ry(-1.78040974990687) q[1];
ry(-1.1120500656573347) q[2];
cx q[1],q[2];
ry(-1.5060721323290949) q[2];
ry(-3.0503699534640303) q[3];
cx q[2],q[3];
ry(-1.8763628691380076) q[2];
ry(-2.501743903840481) q[3];
cx q[2],q[3];
ry(2.8328288326839797) q[3];
ry(-2.6354182572762026) q[4];
cx q[3],q[4];
ry(-0.5014184038642479) q[3];
ry(-3.028766353195116) q[4];
cx q[3],q[4];
ry(0.259275457475589) q[4];
ry(-2.476237301611484) q[5];
cx q[4],q[5];
ry(-0.20067791134277613) q[4];
ry(0.32486782425947264) q[5];
cx q[4],q[5];
ry(-2.5325660309832374) q[5];
ry(-1.0551597409310205) q[6];
cx q[5],q[6];
ry(2.9673928442178874) q[5];
ry(-1.6329902509382577) q[6];
cx q[5],q[6];
ry(-0.04735603116188416) q[6];
ry(3.0257140286725313) q[7];
cx q[6],q[7];
ry(3.0718768782278265) q[6];
ry(1.5861185851821524) q[7];
cx q[6],q[7];
ry(1.8599906110934306) q[0];
ry(1.5942079833481873) q[1];
cx q[0],q[1];
ry(-1.954102002821114) q[0];
ry(0.42452751895054386) q[1];
cx q[0],q[1];
ry(-3.0323631897785215) q[1];
ry(-1.268889020256531) q[2];
cx q[1],q[2];
ry(1.7548867807624422) q[1];
ry(2.2014186427921656) q[2];
cx q[1],q[2];
ry(2.572317282552228) q[2];
ry(1.7112022375726283) q[3];
cx q[2],q[3];
ry(-3.119475994213203) q[2];
ry(2.6355779258767136) q[3];
cx q[2],q[3];
ry(-2.5627033845507716) q[3];
ry(0.807966949154394) q[4];
cx q[3],q[4];
ry(-0.4865208679115707) q[3];
ry(-2.78528943912605) q[4];
cx q[3],q[4];
ry(1.0951262003182585) q[4];
ry(-1.6930573089941083) q[5];
cx q[4],q[5];
ry(-0.8660188765687629) q[4];
ry(-0.37794644154367496) q[5];
cx q[4],q[5];
ry(-2.5148251635589616) q[5];
ry(-2.7244825996342383) q[6];
cx q[5],q[6];
ry(-1.672235682882291) q[5];
ry(3.109219174115991) q[6];
cx q[5],q[6];
ry(1.980400120661975) q[6];
ry(-0.5980312655361391) q[7];
cx q[6],q[7];
ry(-0.9583890716393011) q[6];
ry(0.2154902757374888) q[7];
cx q[6],q[7];
ry(-1.74255425945027) q[0];
ry(-1.5391035544388254) q[1];
cx q[0],q[1];
ry(2.0828430110268443) q[0];
ry(-2.903781463748396) q[1];
cx q[0],q[1];
ry(-2.136107679659222) q[1];
ry(-0.40767943678065915) q[2];
cx q[1],q[2];
ry(-0.9255270662042168) q[1];
ry(0.6099076478757617) q[2];
cx q[1],q[2];
ry(-3.0009905541465263) q[2];
ry(-1.9045603889830862) q[3];
cx q[2],q[3];
ry(2.3520279895588594) q[2];
ry(-2.3857266791366594) q[3];
cx q[2],q[3];
ry(-2.56325667220414) q[3];
ry(0.5990147033777529) q[4];
cx q[3],q[4];
ry(2.9438600029073507) q[3];
ry(-1.5637846808947773) q[4];
cx q[3],q[4];
ry(2.9715772777665297) q[4];
ry(-0.785309573659676) q[5];
cx q[4],q[5];
ry(2.400482742212227) q[4];
ry(0.8247081751371796) q[5];
cx q[4],q[5];
ry(-1.4276524385615943) q[5];
ry(-2.3510126859738807) q[6];
cx q[5],q[6];
ry(2.7034574684310027) q[5];
ry(0.1581385309109029) q[6];
cx q[5],q[6];
ry(-1.6139774639953561) q[6];
ry(-2.1264528308316564) q[7];
cx q[6],q[7];
ry(2.5360551838284757) q[6];
ry(-0.6988904859491798) q[7];
cx q[6],q[7];
ry(-0.3231594966414433) q[0];
ry(3.1069956451298553) q[1];
cx q[0],q[1];
ry(0.5436141479043898) q[0];
ry(-0.7334543173334609) q[1];
cx q[0],q[1];
ry(-1.5174723830374983) q[1];
ry(-0.07636014131582236) q[2];
cx q[1],q[2];
ry(-1.8689470688625576) q[1];
ry(0.7660346194802354) q[2];
cx q[1],q[2];
ry(0.544987064116333) q[2];
ry(-0.23585808145929213) q[3];
cx q[2],q[3];
ry(0.2517230066077607) q[2];
ry(-2.116004429816838) q[3];
cx q[2],q[3];
ry(0.763181995705379) q[3];
ry(-3.015519857826098) q[4];
cx q[3],q[4];
ry(-2.8373132114515447) q[3];
ry(-1.6671455103985693) q[4];
cx q[3],q[4];
ry(1.3272699666263348) q[4];
ry(2.4157187684109327) q[5];
cx q[4],q[5];
ry(0.9074116721091396) q[4];
ry(2.5747639417417996) q[5];
cx q[4],q[5];
ry(1.7129388840713387) q[5];
ry(2.140635208846983) q[6];
cx q[5],q[6];
ry(3.078402212660037) q[5];
ry(2.806919676918963) q[6];
cx q[5],q[6];
ry(-2.6143411612221126) q[6];
ry(1.6820691335117433) q[7];
cx q[6],q[7];
ry(0.07638592026167698) q[6];
ry(-2.3603543010096835) q[7];
cx q[6],q[7];
ry(1.1693031906827012) q[0];
ry(-1.095711075590034) q[1];
cx q[0],q[1];
ry(2.517147242234199) q[0];
ry(1.0788234992431507) q[1];
cx q[0],q[1];
ry(-1.14333192934092) q[1];
ry(0.8984681208807547) q[2];
cx q[1],q[2];
ry(-2.757917850559925) q[1];
ry(-2.9239174825008134) q[2];
cx q[1],q[2];
ry(2.434605958092588) q[2];
ry(2.6499980901799742) q[3];
cx q[2],q[3];
ry(2.827426792643077) q[2];
ry(-2.4543486024247483) q[3];
cx q[2],q[3];
ry(-2.015045880610101) q[3];
ry(-0.5078553848506369) q[4];
cx q[3],q[4];
ry(-0.4607238818481617) q[3];
ry(0.7391737905354192) q[4];
cx q[3],q[4];
ry(2.0978628603851046) q[4];
ry(2.7325006305441724) q[5];
cx q[4],q[5];
ry(1.2210219538264768) q[4];
ry(1.1262636726773991) q[5];
cx q[4],q[5];
ry(1.9421939082163513) q[5];
ry(0.2933987661929516) q[6];
cx q[5],q[6];
ry(3.0999168392111196) q[5];
ry(-2.5046727438654868) q[6];
cx q[5],q[6];
ry(-1.3841236286539838) q[6];
ry(0.6289053045344728) q[7];
cx q[6],q[7];
ry(2.512246643379275) q[6];
ry(-1.2350245840453384) q[7];
cx q[6],q[7];
ry(0.06690139847796939) q[0];
ry(-0.7553170873781286) q[1];
cx q[0],q[1];
ry(1.141517300347779) q[0];
ry(-1.8213755708286334) q[1];
cx q[0],q[1];
ry(1.7625499008338648) q[1];
ry(-2.1304566653005352) q[2];
cx q[1],q[2];
ry(-0.6016221465954863) q[1];
ry(-0.07724845023528329) q[2];
cx q[1],q[2];
ry(2.130986829767363) q[2];
ry(-0.5607942459132047) q[3];
cx q[2],q[3];
ry(0.6099809757119435) q[2];
ry(-0.026595214773406803) q[3];
cx q[2],q[3];
ry(2.2131366867881117) q[3];
ry(1.7300095306970622) q[4];
cx q[3],q[4];
ry(1.8302996428614833) q[3];
ry(1.6337381068044676) q[4];
cx q[3],q[4];
ry(2.4724180047651423) q[4];
ry(0.7636007997900923) q[5];
cx q[4],q[5];
ry(0.8606615214230904) q[4];
ry(1.8481918026495077) q[5];
cx q[4],q[5];
ry(0.22699173241858972) q[5];
ry(0.455518201484815) q[6];
cx q[5],q[6];
ry(2.915030395125756) q[5];
ry(2.8772855427692403) q[6];
cx q[5],q[6];
ry(2.1550183424857257) q[6];
ry(-0.4600066486025965) q[7];
cx q[6],q[7];
ry(1.0039179575247505) q[6];
ry(-2.516230777622364) q[7];
cx q[6],q[7];
ry(-1.4210386783443902) q[0];
ry(1.7133575737769071) q[1];
cx q[0],q[1];
ry(0.6235296834217188) q[0];
ry(2.478914478055052) q[1];
cx q[0],q[1];
ry(-0.9443957738146894) q[1];
ry(1.997249628821305) q[2];
cx q[1],q[2];
ry(2.8354915800457245) q[1];
ry(-0.5241407509069864) q[2];
cx q[1],q[2];
ry(-1.9136629161108667) q[2];
ry(-2.4516868357281525) q[3];
cx q[2],q[3];
ry(-1.2841793203018783) q[2];
ry(-1.8193016611298365) q[3];
cx q[2],q[3];
ry(-0.032815690252495415) q[3];
ry(-0.23365783303966528) q[4];
cx q[3],q[4];
ry(1.5253495854849437) q[3];
ry(-2.1182079768496007) q[4];
cx q[3],q[4];
ry(-2.4879538496810847) q[4];
ry(0.2519162051036723) q[5];
cx q[4],q[5];
ry(0.8483195083871086) q[4];
ry(0.9186487189666394) q[5];
cx q[4],q[5];
ry(0.008634821430181145) q[5];
ry(2.3540278262665595) q[6];
cx q[5],q[6];
ry(-0.6417467575680735) q[5];
ry(-3.0436718852435782) q[6];
cx q[5],q[6];
ry(-1.6582771655896542) q[6];
ry(2.576304276357899) q[7];
cx q[6],q[7];
ry(-3.1275733684131137) q[6];
ry(0.5457165187019326) q[7];
cx q[6],q[7];
ry(-2.3682900650288325) q[0];
ry(2.468286883723544) q[1];
cx q[0],q[1];
ry(1.6326477511677404) q[0];
ry(1.437371701752924) q[1];
cx q[0],q[1];
ry(-2.46824262268637) q[1];
ry(-0.07047345764902618) q[2];
cx q[1],q[2];
ry(0.4081605887492644) q[1];
ry(1.2086617105713549) q[2];
cx q[1],q[2];
ry(-2.9583974815296172) q[2];
ry(2.9659436108266237) q[3];
cx q[2],q[3];
ry(-2.5619449651723145) q[2];
ry(1.1641789366570894) q[3];
cx q[2],q[3];
ry(-0.22415741675288794) q[3];
ry(-0.3802537476021653) q[4];
cx q[3],q[4];
ry(2.1683627943914687) q[3];
ry(-0.801019710269231) q[4];
cx q[3],q[4];
ry(0.07650904905891842) q[4];
ry(-1.222917808568786) q[5];
cx q[4],q[5];
ry(-0.6633238429910419) q[4];
ry(1.976132924983434) q[5];
cx q[4],q[5];
ry(-0.07777620986076617) q[5];
ry(1.281234842417513) q[6];
cx q[5],q[6];
ry(-1.3949363581120027) q[5];
ry(1.523669909173744) q[6];
cx q[5],q[6];
ry(-1.5790466009596544) q[6];
ry(2.6002236180758125) q[7];
cx q[6],q[7];
ry(2.2040917416780617) q[6];
ry(0.8239809750398638) q[7];
cx q[6],q[7];
ry(-1.5898726290377692) q[0];
ry(-2.80632369319845) q[1];
cx q[0],q[1];
ry(-1.995156771642466) q[0];
ry(0.24924114083796708) q[1];
cx q[0],q[1];
ry(-1.544861960800672) q[1];
ry(-1.7729492052576654) q[2];
cx q[1],q[2];
ry(0.45129943702990766) q[1];
ry(-1.827147462360621) q[2];
cx q[1],q[2];
ry(-1.3627923292386246) q[2];
ry(-0.39190873799566367) q[3];
cx q[2],q[3];
ry(1.7354639991138772) q[2];
ry(1.2051470626392762) q[3];
cx q[2],q[3];
ry(1.5663208645844593) q[3];
ry(1.52519032200996) q[4];
cx q[3],q[4];
ry(-0.168568233240954) q[3];
ry(3.136951205823683) q[4];
cx q[3],q[4];
ry(0.46798357564811616) q[4];
ry(-2.4697212532893427) q[5];
cx q[4],q[5];
ry(-2.916277791999426) q[4];
ry(1.246658610494089) q[5];
cx q[4],q[5];
ry(1.8535807850666555) q[5];
ry(2.5098784833504357) q[6];
cx q[5],q[6];
ry(-1.0639222700364375) q[5];
ry(-1.4457597689363684) q[6];
cx q[5],q[6];
ry(-1.993244052395534) q[6];
ry(-1.4152360762846254) q[7];
cx q[6],q[7];
ry(-2.1325217016009983) q[6];
ry(-2.851622746235012) q[7];
cx q[6],q[7];
ry(0.22021219171606296) q[0];
ry(0.1974415357995936) q[1];
cx q[0],q[1];
ry(-0.3754723099588529) q[0];
ry(2.681492603164772) q[1];
cx q[0],q[1];
ry(-2.4995323966149496) q[1];
ry(-1.386582173455082) q[2];
cx q[1],q[2];
ry(-0.16382027775494645) q[1];
ry(-0.9992959824302643) q[2];
cx q[1],q[2];
ry(-0.44926964175655976) q[2];
ry(0.6356498574678797) q[3];
cx q[2],q[3];
ry(1.4429628616914247) q[2];
ry(2.958163062160968) q[3];
cx q[2],q[3];
ry(1.703104793972046) q[3];
ry(3.0385437593618985) q[4];
cx q[3],q[4];
ry(1.9412206231504612) q[3];
ry(2.3255879734313862) q[4];
cx q[3],q[4];
ry(-1.3426327847734525) q[4];
ry(-0.6227436300130638) q[5];
cx q[4],q[5];
ry(1.826585524293952) q[4];
ry(-2.5448217175574124) q[5];
cx q[4],q[5];
ry(-2.0360224425130093) q[5];
ry(3.04867483874742) q[6];
cx q[5],q[6];
ry(-2.829809303748837) q[5];
ry(2.0488117830431927) q[6];
cx q[5],q[6];
ry(1.540136504862321) q[6];
ry(2.420797203514385) q[7];
cx q[6],q[7];
ry(-0.14138219135119268) q[6];
ry(2.8231180916137295) q[7];
cx q[6],q[7];
ry(-1.8409501512316213) q[0];
ry(3.074278508664458) q[1];
cx q[0],q[1];
ry(-1.5120528934817754) q[0];
ry(2.7242161983547506) q[1];
cx q[0],q[1];
ry(-2.4512322881034936) q[1];
ry(2.4956401302462465) q[2];
cx q[1],q[2];
ry(-0.26011537899954007) q[1];
ry(1.3903941583248396) q[2];
cx q[1],q[2];
ry(-0.2643442447130404) q[2];
ry(-3.07912085334025) q[3];
cx q[2],q[3];
ry(-1.9295830165720524) q[2];
ry(0.5039057586371021) q[3];
cx q[2],q[3];
ry(2.261053317148061) q[3];
ry(2.506108744994429) q[4];
cx q[3],q[4];
ry(-1.2649594157708226) q[3];
ry(-1.9117370883380345) q[4];
cx q[3],q[4];
ry(2.18910553404909) q[4];
ry(-0.40759290720098984) q[5];
cx q[4],q[5];
ry(-0.4703434669014408) q[4];
ry(-2.933580072733699) q[5];
cx q[4],q[5];
ry(-2.9799716785952204) q[5];
ry(0.44598908639970514) q[6];
cx q[5],q[6];
ry(-2.199727857169681) q[5];
ry(1.7644585883569592) q[6];
cx q[5],q[6];
ry(-0.06146813752345501) q[6];
ry(3.1061028386811884) q[7];
cx q[6],q[7];
ry(-2.611965337576772) q[6];
ry(-0.7378668287674142) q[7];
cx q[6],q[7];
ry(-2.695833450539068) q[0];
ry(1.5740302426916595) q[1];
cx q[0],q[1];
ry(-2.8644795646864694) q[0];
ry(1.4106443315306505) q[1];
cx q[0],q[1];
ry(0.8241562710641783) q[1];
ry(2.3564093471669367) q[2];
cx q[1],q[2];
ry(1.1102788414267355) q[1];
ry(-0.9154083150907896) q[2];
cx q[1],q[2];
ry(-0.05020148946119719) q[2];
ry(-1.1523357805406216) q[3];
cx q[2],q[3];
ry(0.17853782840174134) q[2];
ry(-1.405645295680115) q[3];
cx q[2],q[3];
ry(-1.628469686119913) q[3];
ry(0.10405312623903684) q[4];
cx q[3],q[4];
ry(2.5074793135990485) q[3];
ry(-1.3786546788465799) q[4];
cx q[3],q[4];
ry(2.0524255410312877) q[4];
ry(1.798498712953997) q[5];
cx q[4],q[5];
ry(0.6742519810114462) q[4];
ry(1.7265174154153726) q[5];
cx q[4],q[5];
ry(0.17447878989324678) q[5];
ry(0.31403957342233335) q[6];
cx q[5],q[6];
ry(1.7708836255564826) q[5];
ry(-2.2201184545328125) q[6];
cx q[5],q[6];
ry(-0.40740095555642203) q[6];
ry(2.2893103736793705) q[7];
cx q[6],q[7];
ry(1.1314665034629543) q[6];
ry(-2.205189628631473) q[7];
cx q[6],q[7];
ry(-0.7020348496548303) q[0];
ry(-0.9320114240650327) q[1];
cx q[0],q[1];
ry(3.1131172943694736) q[0];
ry(-1.6735012823848017) q[1];
cx q[0],q[1];
ry(-2.4740844912369657) q[1];
ry(-0.08585913620899384) q[2];
cx q[1],q[2];
ry(2.188425779191522) q[1];
ry(0.34150599031935336) q[2];
cx q[1],q[2];
ry(1.405668564515778) q[2];
ry(-2.8553934204814877) q[3];
cx q[2],q[3];
ry(-2.0063822140979735) q[2];
ry(2.46581046244517) q[3];
cx q[2],q[3];
ry(0.8321314445269054) q[3];
ry(-0.7333306683718703) q[4];
cx q[3],q[4];
ry(0.5950329144645904) q[3];
ry(-1.5228309787870282) q[4];
cx q[3],q[4];
ry(1.5569092522192431) q[4];
ry(-0.7954697571486875) q[5];
cx q[4],q[5];
ry(-0.8755655374085353) q[4];
ry(-0.40750924164971064) q[5];
cx q[4],q[5];
ry(-2.3696808006070165) q[5];
ry(0.7746296774305348) q[6];
cx q[5],q[6];
ry(-0.2840733987346553) q[5];
ry(3.1324572763598413) q[6];
cx q[5],q[6];
ry(1.9436513447871304) q[6];
ry(0.46281215140601706) q[7];
cx q[6],q[7];
ry(-0.9901263321244868) q[6];
ry(1.1119591729263893) q[7];
cx q[6],q[7];
ry(1.8985153318568775) q[0];
ry(2.7174371579399423) q[1];
cx q[0],q[1];
ry(3.0448535029063835) q[0];
ry(-2.830646284499358) q[1];
cx q[0],q[1];
ry(-1.3903089973435865) q[1];
ry(-0.48073794842683604) q[2];
cx q[1],q[2];
ry(2.5280676764162453) q[1];
ry(1.455230475988516) q[2];
cx q[1],q[2];
ry(0.6101148758570666) q[2];
ry(-3.0990356692542433) q[3];
cx q[2],q[3];
ry(2.0281063298328528) q[2];
ry(2.8980136679103716) q[3];
cx q[2],q[3];
ry(-2.954941142904528) q[3];
ry(2.8754349509083386) q[4];
cx q[3],q[4];
ry(-1.7223964062038277) q[3];
ry(-1.1779021698516696) q[4];
cx q[3],q[4];
ry(-1.0294032636431216) q[4];
ry(1.3325433262066273) q[5];
cx q[4],q[5];
ry(-0.45723023113124617) q[4];
ry(-1.359178580589509) q[5];
cx q[4],q[5];
ry(-2.468217622974636) q[5];
ry(1.0077063735400285) q[6];
cx q[5],q[6];
ry(-1.1991708712135256) q[5];
ry(2.9902437494711873) q[6];
cx q[5],q[6];
ry(-0.012438586729773604) q[6];
ry(1.3648347433449208) q[7];
cx q[6],q[7];
ry(2.521286342546264) q[6];
ry(-2.6216727620683407) q[7];
cx q[6],q[7];
ry(-2.3200395290395597) q[0];
ry(-1.2533985392205003) q[1];
cx q[0],q[1];
ry(-2.1318148784345325) q[0];
ry(2.7091283214354536) q[1];
cx q[0],q[1];
ry(0.15575909475152247) q[1];
ry(0.07328159141428969) q[2];
cx q[1],q[2];
ry(-0.36219941772712533) q[1];
ry(0.11459274196075868) q[2];
cx q[1],q[2];
ry(-1.548570053389896) q[2];
ry(-1.5648744304043403) q[3];
cx q[2],q[3];
ry(0.1751203002966646) q[2];
ry(0.8525058104106124) q[3];
cx q[2],q[3];
ry(-2.1469129425277713) q[3];
ry(-1.0800656625775376) q[4];
cx q[3],q[4];
ry(2.5030380678171045) q[3];
ry(-2.9978357674733163) q[4];
cx q[3],q[4];
ry(0.7387844585469736) q[4];
ry(0.8222807261466345) q[5];
cx q[4],q[5];
ry(-0.474168526782182) q[4];
ry(-2.1606438731613524) q[5];
cx q[4],q[5];
ry(-0.4042834514864903) q[5];
ry(2.758679813987838) q[6];
cx q[5],q[6];
ry(-1.6664411465135505) q[5];
ry(0.09216859273426348) q[6];
cx q[5],q[6];
ry(-0.819836908071716) q[6];
ry(1.8459179569295894) q[7];
cx q[6],q[7];
ry(-2.3883240114311852) q[6];
ry(1.6364273000993619) q[7];
cx q[6],q[7];
ry(-2.2444145113047544) q[0];
ry(1.0145933333602892) q[1];
cx q[0],q[1];
ry(0.2651605382682159) q[0];
ry(-2.0338457314125833) q[1];
cx q[0],q[1];
ry(2.3867880854790386) q[1];
ry(-0.09933556306070024) q[2];
cx q[1],q[2];
ry(0.8303113419777022) q[1];
ry(-0.9517356842269433) q[2];
cx q[1],q[2];
ry(-2.3974821104049826) q[2];
ry(-0.24565603881495338) q[3];
cx q[2],q[3];
ry(0.149130738854572) q[2];
ry(1.2965238318099708) q[3];
cx q[2],q[3];
ry(1.1258812621090293) q[3];
ry(-2.6578821153184644) q[4];
cx q[3],q[4];
ry(1.2867004017748416) q[3];
ry(-2.3254277924110687) q[4];
cx q[3],q[4];
ry(-0.7627758493784143) q[4];
ry(2.5862368840324885) q[5];
cx q[4],q[5];
ry(-3.086902278755119) q[4];
ry(-1.5271474710448034) q[5];
cx q[4],q[5];
ry(1.4818157777403527) q[5];
ry(2.8767406654010266) q[6];
cx q[5],q[6];
ry(1.8592133713887986) q[5];
ry(2.705502720666403) q[6];
cx q[5],q[6];
ry(-2.9224908246613857) q[6];
ry(0.9371888123883583) q[7];
cx q[6],q[7];
ry(-0.2663679096825172) q[6];
ry(2.3535229528194512) q[7];
cx q[6],q[7];
ry(0.6533602022355286) q[0];
ry(-1.9636406123747925) q[1];
cx q[0],q[1];
ry(2.6449574741662514) q[0];
ry(1.9456516352696764) q[1];
cx q[0],q[1];
ry(2.056736737659099) q[1];
ry(-0.7490364639138001) q[2];
cx q[1],q[2];
ry(-2.4769409459425122) q[1];
ry(1.4676736362962188) q[2];
cx q[1],q[2];
ry(2.676553519224986) q[2];
ry(-2.3514626819883557) q[3];
cx q[2],q[3];
ry(2.887949331929149) q[2];
ry(1.4201156066903424) q[3];
cx q[2],q[3];
ry(-1.8147393336439934) q[3];
ry(-2.419670578377018) q[4];
cx q[3],q[4];
ry(2.143745288523295) q[3];
ry(0.649582308358899) q[4];
cx q[3],q[4];
ry(-2.183356963634631) q[4];
ry(-0.5494509728860607) q[5];
cx q[4],q[5];
ry(-0.2398976702032773) q[4];
ry(-2.708547607899447) q[5];
cx q[4],q[5];
ry(-0.31503221338557985) q[5];
ry(-0.03658784807837457) q[6];
cx q[5],q[6];
ry(-0.055788962170392274) q[5];
ry(2.745318815745574) q[6];
cx q[5],q[6];
ry(1.304635654487413) q[6];
ry(-1.8443120821212882) q[7];
cx q[6],q[7];
ry(2.774558358435686) q[6];
ry(-2.9471458700177524) q[7];
cx q[6],q[7];
ry(1.9876857112634778) q[0];
ry(-1.173686962325002) q[1];
cx q[0],q[1];
ry(0.7746630894752906) q[0];
ry(-3.085571692936309) q[1];
cx q[0],q[1];
ry(2.564155778297599) q[1];
ry(-1.6863060075549) q[2];
cx q[1],q[2];
ry(2.9474892928168934) q[1];
ry(2.247097640662637) q[2];
cx q[1],q[2];
ry(-2.1583720276561644) q[2];
ry(0.5304777757294671) q[3];
cx q[2],q[3];
ry(2.8947135794647525) q[2];
ry(1.9962321091014958) q[3];
cx q[2],q[3];
ry(0.8601540391962678) q[3];
ry(1.9080049258241) q[4];
cx q[3],q[4];
ry(0.11368553730352478) q[3];
ry(-2.925582277967002) q[4];
cx q[3],q[4];
ry(0.7647890089786289) q[4];
ry(1.2860720064728541) q[5];
cx q[4],q[5];
ry(0.14326834747933176) q[4];
ry(-0.5447297416660142) q[5];
cx q[4],q[5];
ry(1.796570075561423) q[5];
ry(-0.37168337559588677) q[6];
cx q[5],q[6];
ry(0.2702863653444271) q[5];
ry(-0.06385113691510647) q[6];
cx q[5],q[6];
ry(-0.9502899092169964) q[6];
ry(1.1422611197535926) q[7];
cx q[6],q[7];
ry(-1.945858704427434) q[6];
ry(-2.9771527231139987) q[7];
cx q[6],q[7];
ry(-1.6280306878995967) q[0];
ry(-2.666604738617412) q[1];
cx q[0],q[1];
ry(-0.5529087201232397) q[0];
ry(1.6708518316649084) q[1];
cx q[0],q[1];
ry(1.3494460913063648) q[1];
ry(-0.5362149018587927) q[2];
cx q[1],q[2];
ry(-0.4656402893271796) q[1];
ry(2.8982576353702503) q[2];
cx q[1],q[2];
ry(2.8300260121034024) q[2];
ry(0.7747339504443403) q[3];
cx q[2],q[3];
ry(-1.9035627416495795) q[2];
ry(1.5857449233847323) q[3];
cx q[2],q[3];
ry(1.5968108406615644) q[3];
ry(3.141508252632971) q[4];
cx q[3],q[4];
ry(-0.22636368396955425) q[3];
ry(-0.6927952429701196) q[4];
cx q[3],q[4];
ry(-1.47598782176975) q[4];
ry(3.0197288772435478) q[5];
cx q[4],q[5];
ry(2.6425012251050646) q[4];
ry(-3.1375587318630203) q[5];
cx q[4],q[5];
ry(-2.203980666834892) q[5];
ry(-3.060171477091881) q[6];
cx q[5],q[6];
ry(-0.9474893163412883) q[5];
ry(1.2032100589366612) q[6];
cx q[5],q[6];
ry(-1.1897969344644348) q[6];
ry(-1.0650952162772542) q[7];
cx q[6],q[7];
ry(2.878842978617009) q[6];
ry(0.9937457890288606) q[7];
cx q[6],q[7];
ry(-1.0516380813382855) q[0];
ry(-2.7431319528718334) q[1];
cx q[0],q[1];
ry(-1.319495939577676) q[0];
ry(-2.63702661411876) q[1];
cx q[0],q[1];
ry(2.175947454559119) q[1];
ry(-1.8667849709530344) q[2];
cx q[1],q[2];
ry(-2.0509814648556484) q[1];
ry(-0.2618184525358479) q[2];
cx q[1],q[2];
ry(0.2267524578776721) q[2];
ry(0.7257111369118217) q[3];
cx q[2],q[3];
ry(-2.2196469234273515) q[2];
ry(0.7850415795056147) q[3];
cx q[2],q[3];
ry(-0.42616686237445117) q[3];
ry(1.565790583332676) q[4];
cx q[3],q[4];
ry(0.1983040884158811) q[3];
ry(1.488085789536446) q[4];
cx q[3],q[4];
ry(-2.8946389703941318) q[4];
ry(1.407297664423664) q[5];
cx q[4],q[5];
ry(2.09578358703028) q[4];
ry(-0.18545295605346823) q[5];
cx q[4],q[5];
ry(2.4552375197059737) q[5];
ry(-1.9070001880790635) q[6];
cx q[5],q[6];
ry(-1.867906783605866) q[5];
ry(0.3375286706088569) q[6];
cx q[5],q[6];
ry(2.5114356179192634) q[6];
ry(-1.9808774201237869) q[7];
cx q[6],q[7];
ry(-0.889415573764439) q[6];
ry(-1.790094101524452) q[7];
cx q[6],q[7];
ry(-2.5134128630640444) q[0];
ry(-0.270273270529547) q[1];
ry(0.12923635842411496) q[2];
ry(-0.980009771837845) q[3];
ry(-1.4206348014349646) q[4];
ry(-2.4208272738231322) q[5];
ry(-0.8112313349632246) q[6];
ry(-0.1557856741320379) q[7];