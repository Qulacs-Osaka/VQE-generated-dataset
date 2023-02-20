OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.7965990246836014) q[0];
rz(-2.726451777272767) q[0];
ry(-2.922191270963307) q[1];
rz(0.16878089559084808) q[1];
ry(2.6304401458744895) q[2];
rz(-1.5815161858540903) q[2];
ry(1.5181932734205352) q[3];
rz(1.0218052593659914) q[3];
ry(-0.000606377960489901) q[4];
rz(0.4106128670154812) q[4];
ry(3.1304373279227384) q[5];
rz(-2.204501182602712) q[5];
ry(1.9114518258648565) q[6];
rz(1.2632972059459568) q[6];
ry(0.4070669711485338) q[7];
rz(-0.1075861481021887) q[7];
ry(1.9828340168439) q[8];
rz(-0.3330998282944009) q[8];
ry(0.47686387713988626) q[9];
rz(-2.0319800058837068) q[9];
ry(-0.565071147233874) q[10];
rz(-2.437932178818569) q[10];
ry(2.7304256336784256) q[11];
rz(2.590074424961453) q[11];
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
ry(2.7391952743171335) q[0];
rz(0.04201398776405263) q[0];
ry(1.551731365024203) q[1];
rz(-2.5735178854327523) q[1];
ry(0.0032573926848584725) q[2];
rz(2.44352737664375) q[2];
ry(-0.2764666497613234) q[3];
rz(-1.5494828068610706) q[3];
ry(-0.00045994686726036344) q[4];
rz(-0.3895306697045226) q[4];
ry(-0.002642513027412363) q[5];
rz(1.6437490991640509) q[5];
ry(2.591702971363464) q[6];
rz(1.6146438909366636) q[6];
ry(0.7993726661439791) q[7];
rz(0.16283669193392714) q[7];
ry(0.5358302183313368) q[8];
rz(-1.4029428969389452) q[8];
ry(-2.3335375482866656) q[9];
rz(0.9976424006671739) q[9];
ry(2.5063509785302287) q[10];
rz(-0.4713967644639494) q[10];
ry(2.2685518930815) q[11];
rz(-2.943049108036936) q[11];
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
ry(1.514002095927243) q[0];
rz(-2.8087796144173205) q[0];
ry(1.244704492172393) q[1];
rz(-1.9767559180803909) q[1];
ry(-2.7442273995813196) q[2];
rz(-3.1179399096476645) q[2];
ry(-1.4715742590854588) q[3];
rz(3.0854188720655062) q[3];
ry(3.140354915593231) q[4];
rz(-1.0822673957264122) q[4];
ry(-1.54366620054917) q[5];
rz(-2.3450590269619878) q[5];
ry(1.5565541397670142) q[6];
rz(1.2551814616989452) q[6];
ry(-2.562349239411412) q[7];
rz(0.8066549912207588) q[7];
ry(-2.767378818811748) q[8];
rz(-0.11194621809711958) q[8];
ry(-2.2432518254966105) q[9];
rz(-2.1569671490555873) q[9];
ry(1.8583520757340288) q[10];
rz(2.904997504313218) q[10];
ry(-0.6524074891581169) q[11];
rz(3.10198176371932) q[11];
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
ry(0.431042394843165) q[0];
rz(3.0229917002031397) q[0];
ry(2.888375213102881) q[1];
rz(-2.7007447044140784) q[1];
ry(-0.4852352096771124) q[2];
rz(-2.701603751314272) q[2];
ry(-0.7409028980168063) q[3];
rz(-2.1409276021770784) q[3];
ry(-0.0025872002850855225) q[4];
rz(2.0072756259145237) q[4];
ry(-3.1392055382481776) q[5];
rz(2.4767894356399767) q[5];
ry(0.9737247239606791) q[6];
rz(-0.28831362709445507) q[6];
ry(-0.015047340845763145) q[7];
rz(-2.6335160226647716) q[7];
ry(2.398622716701953) q[8];
rz(2.618678983586799) q[8];
ry(2.2629712924496124) q[9];
rz(-2.548123953851555) q[9];
ry(-1.181455401658474) q[10];
rz(1.3682410366686126) q[10];
ry(-1.5779325167769462) q[11];
rz(1.1550860589848142) q[11];
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
ry(1.426343731978438) q[0];
rz(-0.21108864898448182) q[0];
ry(-2.1874679920464697) q[1];
rz(0.20943164684709273) q[1];
ry(-2.2938282124282967) q[2];
rz(-3.0829147580037133) q[2];
ry(2.2213726420783075) q[3];
rz(-0.32526976662003293) q[3];
ry(-0.0008101325878786858) q[4];
rz(-1.181007868053336) q[4];
ry(-3.0416645016426873) q[5];
rz(1.0326335271821474) q[5];
ry(2.642879808507504) q[6];
rz(-1.9808695741482778) q[6];
ry(-1.760765780293998) q[7];
rz(1.6788735554773424) q[7];
ry(1.4728970407505209) q[8];
rz(-2.7251182036655055) q[8];
ry(1.2729153747310873) q[9];
rz(0.7475241925948523) q[9];
ry(1.6847868873322644) q[10];
rz(-1.465085502258928) q[10];
ry(1.6649362250173707) q[11];
rz(-1.7197146627572595) q[11];
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
ry(-1.670116578373678) q[0];
rz(-3.1072760955631695) q[0];
ry(-0.015515760572704096) q[1];
rz(-1.5268649732511594) q[1];
ry(-0.034389916572097334) q[2];
rz(1.5150804464220762) q[2];
ry(-0.1437479560186823) q[3];
rz(3.0146466895825577) q[3];
ry(-1.355307034668355) q[4];
rz(2.6395078560937906) q[4];
ry(-3.1400329111243495) q[5];
rz(-2.917573187324284) q[5];
ry(2.036131054752076) q[6];
rz(2.0017535033493594) q[6];
ry(0.009495271421761764) q[7];
rz(-1.3126484424215468) q[7];
ry(-0.015220523377699418) q[8];
rz(-0.990604547627119) q[8];
ry(2.4519116408838335) q[9];
rz(1.1570413707545626) q[9];
ry(0.47626302298395634) q[10];
rz(-3.07320297667978) q[10];
ry(-2.1493055191102624) q[11];
rz(1.1295552894036485) q[11];
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
ry(-1.6080451873864816) q[0];
rz(-1.3743665451543583) q[0];
ry(0.48473043413109085) q[1];
rz(3.109359641070698) q[1];
ry(0.49103502405009214) q[2];
rz(-0.19787493585578633) q[2];
ry(0.786976077781322) q[3];
rz(1.5730716463690735) q[3];
ry(-3.1400537681959837) q[4];
rz(-0.12479595603074055) q[4];
ry(-2.6862517758818267) q[5];
rz(0.7999969881115714) q[5];
ry(-3.139523102839481) q[6];
rz(-3.0568604433377122) q[6];
ry(1.960056751493557) q[7];
rz(-1.447433770769238) q[7];
ry(-2.1469366187289864) q[8];
rz(-0.7642144708248394) q[8];
ry(1.6977506233096005) q[9];
rz(-1.653421492471538) q[9];
ry(-2.374986305929538) q[10];
rz(-2.8112582986839287) q[10];
ry(-0.7953624513411208) q[11];
rz(-0.09459722076055649) q[11];
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
ry(1.2943259603549793) q[0];
rz(-0.7761452777756808) q[0];
ry(-1.7706052146805578) q[1];
rz(-0.02508212286168998) q[1];
ry(0.0011289980537201316) q[2];
rz(-2.8145081252881132) q[2];
ry(-1.4069500788630682) q[3];
rz(-2.571370373593478) q[3];
ry(-2.8740857339247228) q[4];
rz(-0.03869774459785022) q[4];
ry(-0.001702660144051454) q[5];
rz(-1.1865293161685866) q[5];
ry(-0.8433378920876429) q[6];
rz(1.550347912074236) q[6];
ry(0.0077319425919126165) q[7];
rz(-1.1021094496456012) q[7];
ry(-0.3493156037783596) q[8];
rz(-2.7049794872063617) q[8];
ry(2.256765673777529) q[9];
rz(1.3290941585846616) q[9];
ry(-0.3318111372805145) q[10];
rz(-0.9379409652060007) q[10];
ry(-2.470546117017903) q[11];
rz(0.5299104047384261) q[11];
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
ry(0.008391610119363272) q[0];
rz(-0.9165379088990839) q[0];
ry(-1.3992502737870915) q[1];
rz(-2.474522565182592) q[1];
ry(-3.079545999388327) q[2];
rz(0.4966108832558651) q[2];
ry(2.4698800170640016) q[3];
rz(2.7050752440644823) q[3];
ry(0.002851207017947388) q[4];
rz(2.978806941121) q[4];
ry(-1.4104133806248473) q[5];
rz(0.641374967700507) q[5];
ry(0.00042989748138273814) q[6];
rz(-2.719134016768486) q[6];
ry(1.1822740022612692) q[7];
rz(-0.720360805479462) q[7];
ry(-0.3045432734003013) q[8];
rz(-3.0416852318162593) q[8];
ry(1.9513139779032453) q[9];
rz(-1.8656225716082624) q[9];
ry(2.273994465154629) q[10];
rz(-2.0177359319744976) q[10];
ry(2.3495433030993125) q[11];
rz(-1.1963754081164317) q[11];
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
ry(3.029157215512493) q[0];
rz(-1.5783611725431208) q[0];
ry(2.6223025127019604) q[1];
rz(2.972580488285926) q[1];
ry(0.04819224059359023) q[2];
rz(3.1213220488282216) q[2];
ry(-1.8822597654881985) q[3];
rz(-2.6004697551190334) q[3];
ry(1.796719836280979) q[4];
rz(2.0905452466842016) q[4];
ry(-3.14016331786726) q[5];
rz(-0.25703960874867876) q[5];
ry(0.35619463228680054) q[6];
rz(0.12150911021957861) q[6];
ry(-1.5711858660893911) q[7];
rz(-3.0646626906123178) q[7];
ry(-2.9035204525495297) q[8];
rz(0.6747884573354089) q[8];
ry(-1.8519952458991482) q[9];
rz(3.02381064813413) q[9];
ry(-2.0633809439013966) q[10];
rz(-1.1912812663045884) q[10];
ry(-0.44018575833416396) q[11];
rz(0.20053110287093645) q[11];
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
ry(-0.04102499559483075) q[0];
rz(-0.5364735532710342) q[0];
ry(2.057850356645273) q[1];
rz(-2.629789159773951) q[1];
ry(3.123291571436107) q[2];
rz(2.0383669872956487) q[2];
ry(0.5208081525924879) q[3];
rz(-0.79698269415283) q[3];
ry(-3.1337995723787735) q[4];
rz(-0.0881643660208562) q[4];
ry(0.0003173516297320313) q[5];
rz(-0.18364319349692418) q[5];
ry(3.1380692283128657) q[6];
rz(-2.2058537822063093) q[6];
ry(0.03502585123019841) q[7];
rz(-0.07955582811190515) q[7];
ry(-1.4074216926833607) q[8];
rz(1.069828217567549) q[8];
ry(-1.5701207641830668) q[9];
rz(1.569975616427486) q[9];
ry(0.4139896672152039) q[10];
rz(0.3273315761861894) q[10];
ry(0.5923324652591315) q[11];
rz(-2.2377978071713334) q[11];
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
ry(2.508145620043122) q[0];
rz(-2.0990955459968275) q[0];
ry(-0.865717432878866) q[1];
rz(2.0063748929140757) q[1];
ry(-2.696950296120363) q[2];
rz(-0.08025730731851223) q[2];
ry(2.393473170096512) q[3];
rz(-1.15566411976517) q[3];
ry(0.3710056819580298) q[4];
rz(-1.1717161688739228) q[4];
ry(-3.139121875353761) q[5];
rz(0.5595824616103222) q[5];
ry(-1.8409429160311648) q[6];
rz(2.5218666518409747) q[6];
ry(2.9642989148915837) q[7];
rz(3.137219346224329) q[7];
ry(0.004817651949961643) q[8];
rz(-1.0097483399674339) q[8];
ry(-1.5705370733537576) q[9];
rz(1.383277727179442) q[9];
ry(-0.22200432283535074) q[10];
rz(-1.6929477905754837) q[10];
ry(-3.141264784083309) q[11];
rz(1.6392025174839402) q[11];
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
ry(-0.013482998223688838) q[0];
rz(2.4057835141729864) q[0];
ry(-1.2130917810187931) q[1];
rz(0.046052962622936455) q[1];
ry(2.847078236433479) q[2];
rz(-0.4954549564599452) q[2];
ry(-2.4836580103627335) q[3];
rz(2.3247194781177316) q[3];
ry(1.533982888244863) q[4];
rz(-1.5049765084371607) q[4];
ry(-1.571050045459726) q[5];
rz(2.0783159988796296) q[5];
ry(-3.13771574773589) q[6];
rz(1.3180331047322813) q[6];
ry(1.570662617992733) q[7];
rz(-0.13307320436526648) q[7];
ry(0.0001312203342559469) q[8];
rz(-0.21655679573256947) q[8];
ry(3.060548875753711) q[9];
rz(0.09822876531591174) q[9];
ry(2.8871998504554917) q[10];
rz(0.5272872390973754) q[10];
ry(-1.570619836529847) q[11];
rz(-1.5681597463577348) q[11];
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
ry(-0.756339298606596) q[0];
rz(-1.0237425980371189) q[0];
ry(2.5190916050224055) q[1];
rz(-2.0545796411964536) q[1];
ry(-3.1406423175358817) q[2];
rz(2.868354674543555) q[2];
ry(1.5712214834202527) q[3];
rz(-3.1403626659081803) q[3];
ry(0.05848926429643342) q[4];
rz(-1.4508941018306392) q[4];
ry(-3.141056972109976) q[5];
rz(-1.7630706415219226) q[5];
ry(0.0064344407419714145) q[6];
rz(1.478807364309837) q[6];
ry(-0.0004377417759494355) q[7];
rz(-2.6597426856382334) q[7];
ry(-1.568732469274213) q[8];
rz(-3.1366509984244018) q[8];
ry(-1.571417336923378) q[9];
rz(-3.140335726593857) q[9];
ry(0.0019987264346276916) q[10];
rz(-1.7292476864359247) q[10];
ry(-2.6620525926061935) q[11];
rz(-1.5675097119918937) q[11];
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
ry(-0.009255912280291767) q[0];
rz(2.0850829541458076) q[0];
ry(1.5714504325579952) q[1];
rz(-0.02987416691113598) q[1];
ry(1.5682801251484504) q[2];
rz(0.016161902956087104) q[2];
ry(2.1631380280313355) q[3];
rz(-3.1265113059435388) q[3];
ry(3.0749484685225803) q[4];
rz(3.065536146478523) q[4];
ry(1.9797242238176709) q[5];
rz(0.4196378702844656) q[5];
ry(-1.579460697809559) q[6];
rz(-0.2742664875633185) q[6];
ry(1.6853318923111296) q[7];
rz(1.1658133164957538) q[7];
ry(2.1258056863379173) q[8];
rz(3.1391576870499547) q[8];
ry(1.5706835184113963) q[9];
rz(0.4438837654563657) q[9];
ry(-1.5768953338248657) q[10];
rz(0.0025764489650903404) q[10];
ry(0.9557438602767839) q[11];
rz(1.5671299523093802) q[11];
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
ry(-0.353173254494519) q[0];
rz(-1.5581045919691872) q[0];
ry(-1.4205561163760043) q[1];
rz(-0.2783036918940232) q[1];
ry(0.8627436811332254) q[2];
rz(2.2886291541162236) q[2];
ry(1.987954356251885) q[3];
rz(-0.6155475295750011) q[3];
ry(-3.1370004986487223) q[4];
rz(-3.0153824328623404) q[4];
ry(0.002152757767809084) q[5];
rz(-2.3123392333031156) q[5];
ry(-0.027085406115094912) q[6];
rz(-1.2815146597188027) q[6];
ry(3.1409485504343793) q[7];
rz(-2.9940057361239076) q[7];
ry(1.5563126231240865) q[8];
rz(3.1380108913727804) q[8];
ry(-1.5710160674522111) q[9];
rz(-0.21060333633318237) q[9];
ry(-1.569401477302439) q[10];
rz(0.9508280567103785) q[10];
ry(2.3925336748791177) q[11];
rz(3.1397201875100413) q[11];
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
ry(3.1399572035054186) q[0];
rz(-1.7456786768486483) q[0];
ry(-3.1372014184558763) q[1];
rz(3.045281629694293) q[1];
ry(-0.010640126068850009) q[2];
rz(-2.313994470815647) q[2];
ry(0.001185898649396222) q[3];
rz(-2.4904926847829087) q[3];
ry(0.03151973519702178) q[4];
rz(-0.4405207997503111) q[4];
ry(1.8605261164076596) q[5];
rz(-0.27606812368914097) q[5];
ry(3.1387474416514514) q[6];
rz(1.5631447971225612) q[6];
ry(0.0014434358290793818) q[7];
rz(-1.504503063973619) q[7];
ry(-1.2414637646007871) q[8];
rz(-1.5691529548137035) q[8];
ry(-3.1408863784325245) q[9];
rz(2.9284928020561427) q[9];
ry(-1.571883948344576) q[10];
rz(3.0925429282304266) q[10];
ry(1.5683889530210227) q[11];
rz(-0.5297872892232975) q[11];
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
ry(-0.7339514430475628) q[0];
rz(1.608744668831589) q[0];
ry(-2.990106602900493) q[1];
rz(-1.3817746443166088) q[1];
ry(2.432810149760612) q[2];
rz(-0.0187170036211332) q[2];
ry(-2.1394430023428095) q[3];
rz(2.444597013022196) q[3];
ry(3.1354710040163964) q[4];
rz(-0.5272061706948294) q[4];
ry(3.1394293348965094) q[5];
rz(0.7180867396466634) q[5];
ry(-0.007232032287570078) q[6];
rz(1.4465769194044364) q[6];
ry(0.0013893410729837186) q[7];
rz(0.6809675683930089) q[7];
ry(1.581284450206196) q[8];
rz(1.5738608625114203) q[8];
ry(0.47553066480552464) q[9];
rz(-1.5597827555402983) q[9];
ry(3.0322076068391812) q[10];
rz(-1.6201497438237167) q[10];
ry(3.1407150411698526) q[11];
rz(2.4328700655008046) q[11];
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
ry(-1.5757455526180284) q[0];
rz(-1.497251352730047) q[0];
ry(-1.5165425680441005) q[1];
rz(3.0840217115095094) q[1];
ry(1.5678571280970282) q[2];
rz(-1.5733522961548203) q[2];
ry(-3.1407847823573767) q[3];
rz(-2.286820203228826) q[3];
ry(-0.015464830273256146) q[4];
rz(-0.4316389255269976) q[4];
ry(-0.7904878657949955) q[5];
rz(-0.35700584323374474) q[5];
ry(-2.99059969557473) q[6];
rz(0.5810413163654429) q[6];
ry(2.874692029365363) q[7];
rz(0.13796932979701512) q[7];
ry(-1.5817075107685683) q[8];
rz(-0.08603703173158728) q[8];
ry(0.23098597491047965) q[9];
rz(0.4811742756171302) q[9];
ry(-1.470862475777895) q[10];
rz(1.5698822742064484) q[10];
ry(-1.570195760527417) q[11];
rz(-1.5694115903322292) q[11];
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
ry(0.07308618093320265) q[0];
rz(2.930956072108044) q[0];
ry(-0.008683319613323803) q[1];
rz(0.18517433716158926) q[1];
ry(-1.570210194109351) q[2];
rz(0.25219068470306394) q[2];
ry(-1.5713510042826704) q[3];
rz(-2.9359122573208527) q[3];
ry(3.140119850383842) q[4];
rz(-1.0452270220638806) q[4];
ry(0.0022387549426969855) q[5];
rz(0.9745861145149249) q[5];
ry(-0.0009997676868866991) q[6];
rz(2.1178583773396005) q[6];
ry(-3.1407760659051727) q[7];
rz(2.865810133471742) q[7];
ry(-0.004765049361460605) q[8];
rz(1.679993305162661) q[8];
ry(-0.004518221334853578) q[9];
rz(-0.5383199181730474) q[9];
ry(1.563597060246714) q[10];
rz(-2.812561982157078) q[10];
ry(-2.2114676231596695) q[11];
rz(-1.4736861407958786) q[11];
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
ry(1.600122992155451) q[0];
rz(-2.948783527584309) q[0];
ry(-1.5510006103402814) q[1];
rz(-2.929674116620669) q[1];
ry(-0.14014616051371842) q[2];
rz(1.524414995224222) q[2];
ry(3.0138296887618075) q[3];
rz(1.6227078360398792) q[3];
ry(1.6057046605828198) q[4];
rz(-2.9443511235425954) q[4];
ry(1.552779409438216) q[5];
rz(3.0776470622390177) q[5];
ry(-0.14010908883479978) q[6];
rz(1.8333086710825894) q[6];
ry(-1.8013463812401436) q[7];
rz(-0.16695180708035068) q[7];
ry(-1.2368002000181244) q[8];
rz(-1.617007517881233) q[8];
ry(0.39766790009637903) q[9];
rz(-1.5840147391618042) q[9];
ry(-1.587055056316326) q[10];
rz(3.1050469092289643) q[10];
ry(-2.9366644829278075) q[11];
rz(-1.533061845789887) q[11];