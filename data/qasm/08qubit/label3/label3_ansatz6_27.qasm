OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-3.0949394409254016) q[0];
ry(-2.5899836048057376) q[1];
cx q[0],q[1];
ry(-2.5909186856306636) q[0];
ry(-1.6436061435813132) q[1];
cx q[0],q[1];
ry(-2.534298990549515) q[1];
ry(-2.677703560729015) q[2];
cx q[1],q[2];
ry(-0.5554428697280089) q[1];
ry(1.8860067990427298) q[2];
cx q[1],q[2];
ry(0.0679614976843359) q[2];
ry(-2.640063753897333) q[3];
cx q[2],q[3];
ry(0.27425680153789583) q[2];
ry(0.8045335703407407) q[3];
cx q[2],q[3];
ry(0.28642425709369357) q[3];
ry(0.9299745507730144) q[4];
cx q[3],q[4];
ry(0.39543133803684327) q[3];
ry(2.8277296683170547) q[4];
cx q[3],q[4];
ry(-2.835693073156342) q[4];
ry(-1.6341191731730729) q[5];
cx q[4],q[5];
ry(1.0971510960595363) q[4];
ry(1.8998056916055919) q[5];
cx q[4],q[5];
ry(-0.327377244053773) q[5];
ry(-1.6710978034678394) q[6];
cx q[5],q[6];
ry(-2.6591140706977336) q[5];
ry(0.848074460432013) q[6];
cx q[5],q[6];
ry(-1.207003882577386) q[6];
ry(-0.48932682387923443) q[7];
cx q[6],q[7];
ry(-1.7494126531042369) q[6];
ry(-1.5764374129207304) q[7];
cx q[6],q[7];
ry(2.3020569230501104) q[0];
ry(-1.104832424373186) q[1];
cx q[0],q[1];
ry(-2.0932525232989247) q[0];
ry(1.6735788244151237) q[1];
cx q[0],q[1];
ry(-3.119584266023707) q[1];
ry(1.2744671667034977) q[2];
cx q[1],q[2];
ry(-1.3863792216278439) q[1];
ry(-1.2754403378278258) q[2];
cx q[1],q[2];
ry(-2.3929000684739594) q[2];
ry(0.6547048176719299) q[3];
cx q[2],q[3];
ry(-2.1314213868699885) q[2];
ry(2.981526874697184) q[3];
cx q[2],q[3];
ry(0.6687239629084564) q[3];
ry(3.085240922670485) q[4];
cx q[3],q[4];
ry(-2.6928922718037756) q[3];
ry(-1.263860493407437) q[4];
cx q[3],q[4];
ry(-2.2338724310795266) q[4];
ry(0.03392303729984025) q[5];
cx q[4],q[5];
ry(0.6496072882878362) q[4];
ry(-1.6116831535144236) q[5];
cx q[4],q[5];
ry(-2.6248400477045055) q[5];
ry(-2.7674188092072134) q[6];
cx q[5],q[6];
ry(2.8563629880533936) q[5];
ry(-2.7494392547715707) q[6];
cx q[5],q[6];
ry(1.0187251841262808) q[6];
ry(-0.1500022594215092) q[7];
cx q[6],q[7];
ry(0.19857750695806775) q[6];
ry(-1.2839864706095083) q[7];
cx q[6],q[7];
ry(0.9507420202440137) q[0];
ry(1.1350477759215805) q[1];
cx q[0],q[1];
ry(0.08623509002561214) q[0];
ry(2.795923582348011) q[1];
cx q[0],q[1];
ry(2.88756560519231) q[1];
ry(2.9916352091825775) q[2];
cx q[1],q[2];
ry(-0.19473242797354118) q[1];
ry(-2.5730813375681123) q[2];
cx q[1],q[2];
ry(-0.6039278631096897) q[2];
ry(-1.3950384891153522) q[3];
cx q[2],q[3];
ry(-0.5142968495358863) q[2];
ry(1.6841935612266596) q[3];
cx q[2],q[3];
ry(-0.8550762734799031) q[3];
ry(-1.7260704066253947) q[4];
cx q[3],q[4];
ry(-2.6790393349685395) q[3];
ry(-2.2779942770593364) q[4];
cx q[3],q[4];
ry(-2.1360885431892127) q[4];
ry(-2.109403889735172) q[5];
cx q[4],q[5];
ry(-1.235797475593631) q[4];
ry(-0.7659027866561345) q[5];
cx q[4],q[5];
ry(0.9711288315171971) q[5];
ry(-0.459735874353913) q[6];
cx q[5],q[6];
ry(0.02437647488408068) q[5];
ry(1.618115341508559) q[6];
cx q[5],q[6];
ry(2.033624564776364) q[6];
ry(3.013778201755045) q[7];
cx q[6],q[7];
ry(-2.939845780260824) q[6];
ry(-0.6136560597954555) q[7];
cx q[6],q[7];
ry(-0.4605092885779909) q[0];
ry(-2.3941568403611093) q[1];
cx q[0],q[1];
ry(1.7369070444861063) q[0];
ry(2.8400921894630264) q[1];
cx q[0],q[1];
ry(-2.058047480526225) q[1];
ry(-2.2067448502566283) q[2];
cx q[1],q[2];
ry(-2.3055308063750006) q[1];
ry(1.640664706071431) q[2];
cx q[1],q[2];
ry(-1.5186469663885598) q[2];
ry(0.5656720023003022) q[3];
cx q[2],q[3];
ry(1.6264126690410337) q[2];
ry(-2.6807944979067515) q[3];
cx q[2],q[3];
ry(0.4093581790990234) q[3];
ry(1.3193095696210984) q[4];
cx q[3],q[4];
ry(1.6251795983813946) q[3];
ry(-0.7730453613764166) q[4];
cx q[3],q[4];
ry(-0.7752094701828788) q[4];
ry(-0.6352023670562366) q[5];
cx q[4],q[5];
ry(1.4669846133286688) q[4];
ry(0.9138390139156671) q[5];
cx q[4],q[5];
ry(0.9374031640683032) q[5];
ry(-2.868848378260416) q[6];
cx q[5],q[6];
ry(0.2776873832587303) q[5];
ry(-2.440953396758352) q[6];
cx q[5],q[6];
ry(0.19642584894977555) q[6];
ry(2.6784328775553155) q[7];
cx q[6],q[7];
ry(1.4004039561219876) q[6];
ry(-1.1226358424424578) q[7];
cx q[6],q[7];
ry(0.18921719525816091) q[0];
ry(1.8945869693515516) q[1];
cx q[0],q[1];
ry(0.4879509226173449) q[0];
ry(-2.5903628054846988) q[1];
cx q[0],q[1];
ry(-0.5131636177713776) q[1];
ry(3.0227524047458147) q[2];
cx q[1],q[2];
ry(-0.1623791741842401) q[1];
ry(-2.4756015671588134) q[2];
cx q[1],q[2];
ry(-0.9442925564218712) q[2];
ry(2.521512239692752) q[3];
cx q[2],q[3];
ry(-1.513174331362661) q[2];
ry(2.3110178174660416) q[3];
cx q[2],q[3];
ry(-0.3591076049224854) q[3];
ry(2.3656416894627563) q[4];
cx q[3],q[4];
ry(-0.27069104578345016) q[3];
ry(-2.0546984811799707) q[4];
cx q[3],q[4];
ry(-2.283023730880201) q[4];
ry(3.008877479295081) q[5];
cx q[4],q[5];
ry(2.1995544235418585) q[4];
ry(-1.775202526067707) q[5];
cx q[4],q[5];
ry(1.7087066005833895) q[5];
ry(-2.9932434975307394) q[6];
cx q[5],q[6];
ry(-1.230614033644927) q[5];
ry(-1.231785516483806) q[6];
cx q[5],q[6];
ry(-2.2504496406637235) q[6];
ry(0.5018539144998106) q[7];
cx q[6],q[7];
ry(-0.86645875793611) q[6];
ry(-0.04307006919027887) q[7];
cx q[6],q[7];
ry(0.09515067066089973) q[0];
ry(-2.085121615878934) q[1];
cx q[0],q[1];
ry(-1.653419575916747) q[0];
ry(1.169561717285272) q[1];
cx q[0],q[1];
ry(-1.1112648240243006) q[1];
ry(0.1511335526567491) q[2];
cx q[1],q[2];
ry(1.4669467036681108) q[1];
ry(1.0984253962142958) q[2];
cx q[1],q[2];
ry(-0.05918290451777137) q[2];
ry(1.2698014811011484) q[3];
cx q[2],q[3];
ry(0.7049306337068362) q[2];
ry(1.1881043875489459) q[3];
cx q[2],q[3];
ry(0.6081930875440786) q[3];
ry(-0.4731510595608892) q[4];
cx q[3],q[4];
ry(1.6495586563176632) q[3];
ry(0.6212755552042216) q[4];
cx q[3],q[4];
ry(-2.7190157391479977) q[4];
ry(-2.72321465603469) q[5];
cx q[4],q[5];
ry(-1.995425912385226) q[4];
ry(0.725323079996282) q[5];
cx q[4],q[5];
ry(0.8590696321571292) q[5];
ry(-1.7685048784095647) q[6];
cx q[5],q[6];
ry(1.3340857122769219) q[5];
ry(-0.7717107581807037) q[6];
cx q[5],q[6];
ry(2.2072887318618566) q[6];
ry(-2.7597753406695684) q[7];
cx q[6],q[7];
ry(2.774779124565629) q[6];
ry(-2.5052760427840686) q[7];
cx q[6],q[7];
ry(-0.09097087108562718) q[0];
ry(0.6466822247056873) q[1];
cx q[0],q[1];
ry(1.0985612170968475) q[0];
ry(-2.90629129945041) q[1];
cx q[0],q[1];
ry(1.3066007455329034) q[1];
ry(0.6186898822342188) q[2];
cx q[1],q[2];
ry(-1.023059140134718) q[1];
ry(2.841030391184758) q[2];
cx q[1],q[2];
ry(2.5285984059007767) q[2];
ry(0.9555112225684333) q[3];
cx q[2],q[3];
ry(0.41321942719134946) q[2];
ry(0.5845819665439529) q[3];
cx q[2],q[3];
ry(1.237046760408074) q[3];
ry(1.9826253043451976) q[4];
cx q[3],q[4];
ry(0.484550269259139) q[3];
ry(-0.26433015729587866) q[4];
cx q[3],q[4];
ry(2.0408385284866037) q[4];
ry(-2.830327516811951) q[5];
cx q[4],q[5];
ry(-1.561814071096184) q[4];
ry(-0.4216654506784328) q[5];
cx q[4],q[5];
ry(0.2806427061393379) q[5];
ry(-3.1066247367470163) q[6];
cx q[5],q[6];
ry(-1.8971910983560392) q[5];
ry(2.223638373632732) q[6];
cx q[5],q[6];
ry(0.6363285663050453) q[6];
ry(-2.2729959887390487) q[7];
cx q[6],q[7];
ry(0.1307389696756652) q[6];
ry(0.4061847593983172) q[7];
cx q[6],q[7];
ry(-2.140180912808394) q[0];
ry(-0.7175071336394484) q[1];
cx q[0],q[1];
ry(-0.5809783236080326) q[0];
ry(0.0711466381054798) q[1];
cx q[0],q[1];
ry(-2.7560663426645418) q[1];
ry(2.018766817677415) q[2];
cx q[1],q[2];
ry(-0.41839562463221597) q[1];
ry(0.39567450552990274) q[2];
cx q[1],q[2];
ry(-1.0536715503563325) q[2];
ry(-1.5537377343266756) q[3];
cx q[2],q[3];
ry(2.3998724215199427) q[2];
ry(-2.329546107530057) q[3];
cx q[2],q[3];
ry(3.11219275373787) q[3];
ry(-0.29332249692286716) q[4];
cx q[3],q[4];
ry(-2.063882874844981) q[3];
ry(1.3499231076995448) q[4];
cx q[3],q[4];
ry(0.5982119460527185) q[4];
ry(-2.917455603296683) q[5];
cx q[4],q[5];
ry(-1.0065118153455819) q[4];
ry(-1.7466567137840505) q[5];
cx q[4],q[5];
ry(-1.9307964350648095) q[5];
ry(-1.2834616214363455) q[6];
cx q[5],q[6];
ry(-2.2019823308372737) q[5];
ry(-1.2890644872763402) q[6];
cx q[5],q[6];
ry(-1.663005849477966) q[6];
ry(2.079176411725543) q[7];
cx q[6],q[7];
ry(-2.8008554062741324) q[6];
ry(2.3070664266994583) q[7];
cx q[6],q[7];
ry(-2.8111495041254404) q[0];
ry(-1.139768949041403) q[1];
cx q[0],q[1];
ry(0.6181391844558082) q[0];
ry(1.0114891294848247) q[1];
cx q[0],q[1];
ry(1.6938855839217144) q[1];
ry(0.9321278741320416) q[2];
cx q[1],q[2];
ry(-2.9424387512639965) q[1];
ry(1.1223626469452732) q[2];
cx q[1],q[2];
ry(0.703075229125005) q[2];
ry(1.0808542649009867) q[3];
cx q[2],q[3];
ry(1.4472219108914175) q[2];
ry(-0.4760644613842732) q[3];
cx q[2],q[3];
ry(-2.909139582919852) q[3];
ry(-1.8617588796630207) q[4];
cx q[3],q[4];
ry(3.062058891429277) q[3];
ry(0.38962049260826553) q[4];
cx q[3],q[4];
ry(-2.375791298228484) q[4];
ry(0.5451110020778014) q[5];
cx q[4],q[5];
ry(1.9987422177095555) q[4];
ry(2.1350507940842167) q[5];
cx q[4],q[5];
ry(2.9044694099902477) q[5];
ry(1.3557622344856284) q[6];
cx q[5],q[6];
ry(1.125835445575476) q[5];
ry(0.3420587217626808) q[6];
cx q[5],q[6];
ry(2.365684468273892) q[6];
ry(0.9552827142223085) q[7];
cx q[6],q[7];
ry(-1.812432927420284) q[6];
ry(-2.7921361259945554) q[7];
cx q[6],q[7];
ry(1.426179788810693) q[0];
ry(2.5173449272419393) q[1];
cx q[0],q[1];
ry(-1.222607651252396) q[0];
ry(-2.1100846676434246) q[1];
cx q[0],q[1];
ry(0.6773636874506712) q[1];
ry(1.971367372683915) q[2];
cx q[1],q[2];
ry(-1.9667570909818688) q[1];
ry(-2.054828295781216) q[2];
cx q[1],q[2];
ry(2.4453332654588587) q[2];
ry(-0.11908478461668345) q[3];
cx q[2],q[3];
ry(-2.106111498036366) q[2];
ry(1.266293329994184) q[3];
cx q[2],q[3];
ry(3.1359848875008085) q[3];
ry(1.3733068691691592) q[4];
cx q[3],q[4];
ry(0.4109564581352023) q[3];
ry(-2.1314551128945576) q[4];
cx q[3],q[4];
ry(-1.4780059629540796) q[4];
ry(0.26268132263770266) q[5];
cx q[4],q[5];
ry(-2.134544997969335) q[4];
ry(-1.6611322763272094) q[5];
cx q[4],q[5];
ry(0.3088444503065517) q[5];
ry(-1.7131952286984928) q[6];
cx q[5],q[6];
ry(-1.7223920245555435) q[5];
ry(-2.9105509575768322) q[6];
cx q[5],q[6];
ry(2.957462772719423) q[6];
ry(-1.5007421498368685) q[7];
cx q[6],q[7];
ry(2.179553733444723) q[6];
ry(-2.7078304899105086) q[7];
cx q[6],q[7];
ry(-2.9227951804746812) q[0];
ry(1.6726651825238346) q[1];
cx q[0],q[1];
ry(0.09177311412178747) q[0];
ry(-2.1252786896996065) q[1];
cx q[0],q[1];
ry(2.3674156839928178) q[1];
ry(1.3265900219965898) q[2];
cx q[1],q[2];
ry(1.5880426779757568) q[1];
ry(2.1137055624933345) q[2];
cx q[1],q[2];
ry(0.4735190270404068) q[2];
ry(-0.08897370732209353) q[3];
cx q[2],q[3];
ry(1.7986025363665066) q[2];
ry(2.058221213446133) q[3];
cx q[2],q[3];
ry(2.325632588970936) q[3];
ry(-1.3632065494768444) q[4];
cx q[3],q[4];
ry(-0.8923155559590138) q[3];
ry(0.944979809683664) q[4];
cx q[3],q[4];
ry(-3.1395001526690147) q[4];
ry(1.4892800437474927) q[5];
cx q[4],q[5];
ry(1.0083128042472576) q[4];
ry(0.8852282317716993) q[5];
cx q[4],q[5];
ry(2.1780698115444492) q[5];
ry(1.68705951914872) q[6];
cx q[5],q[6];
ry(-2.9526892888251983) q[5];
ry(-2.4542499406045115) q[6];
cx q[5],q[6];
ry(-1.092985541863717) q[6];
ry(-1.9026491270837291) q[7];
cx q[6],q[7];
ry(0.16278808984893622) q[6];
ry(0.9822369711874832) q[7];
cx q[6],q[7];
ry(3.0859433661185705) q[0];
ry(-0.3536298317281568) q[1];
cx q[0],q[1];
ry(0.30161073353397505) q[0];
ry(0.48003050932275215) q[1];
cx q[0],q[1];
ry(1.720470116556939) q[1];
ry(0.03107826647830133) q[2];
cx q[1],q[2];
ry(-0.9937197129977111) q[1];
ry(1.475774950645799) q[2];
cx q[1],q[2];
ry(-2.4433974210617353) q[2];
ry(-0.9022121460125113) q[3];
cx q[2],q[3];
ry(1.2126491981433958) q[2];
ry(-2.1617736416351168) q[3];
cx q[2],q[3];
ry(2.7763436731317492) q[3];
ry(-2.351762551316887) q[4];
cx q[3],q[4];
ry(0.6613328538351997) q[3];
ry(2.885318328537942) q[4];
cx q[3],q[4];
ry(-1.6314396201896182) q[4];
ry(-2.2845772732687433) q[5];
cx q[4],q[5];
ry(-1.8448806023328683) q[4];
ry(3.138068872883308) q[5];
cx q[4],q[5];
ry(1.6906247102322614) q[5];
ry(2.958957432440854) q[6];
cx q[5],q[6];
ry(0.9496172935023129) q[5];
ry(0.10727906606722866) q[6];
cx q[5],q[6];
ry(-1.7644705067510849) q[6];
ry(2.1886100287287062) q[7];
cx q[6],q[7];
ry(-2.352697482351204) q[6];
ry(2.984165414406543) q[7];
cx q[6],q[7];
ry(-0.4706050331569589) q[0];
ry(0.3978337447551202) q[1];
cx q[0],q[1];
ry(-1.0166541915995326) q[0];
ry(-2.497397929109852) q[1];
cx q[0],q[1];
ry(-2.7637265585223174) q[1];
ry(-2.180892991168263) q[2];
cx q[1],q[2];
ry(0.2892237754711502) q[1];
ry(-3.1025083766930983) q[2];
cx q[1],q[2];
ry(-0.15975172945733185) q[2];
ry(2.3619893111704684) q[3];
cx q[2],q[3];
ry(1.932031944259202) q[2];
ry(-2.922798184686005) q[3];
cx q[2],q[3];
ry(0.907289470241186) q[3];
ry(2.834453434656439) q[4];
cx q[3],q[4];
ry(0.2430699562076958) q[3];
ry(-0.008503278603885) q[4];
cx q[3],q[4];
ry(-2.0551786713874325) q[4];
ry(2.1634215784165747) q[5];
cx q[4],q[5];
ry(-2.742623043663855) q[4];
ry(-0.2041737358830593) q[5];
cx q[4],q[5];
ry(-0.6246638326686691) q[5];
ry(2.4302649936644025) q[6];
cx q[5],q[6];
ry(1.9186704373761723) q[5];
ry(1.682444943609615) q[6];
cx q[5],q[6];
ry(0.5671890984358585) q[6];
ry(-0.3680717283861554) q[7];
cx q[6],q[7];
ry(2.985155024978698) q[6];
ry(2.110627714341162) q[7];
cx q[6],q[7];
ry(2.9777622584991508) q[0];
ry(-1.1239399899960487) q[1];
cx q[0],q[1];
ry(-1.3432484050951636) q[0];
ry(2.856785940768079) q[1];
cx q[0],q[1];
ry(-1.8258634001555907) q[1];
ry(-3.080520298651829) q[2];
cx q[1],q[2];
ry(2.2051941554076393) q[1];
ry(2.1834884454784946) q[2];
cx q[1],q[2];
ry(-1.9553827638777959) q[2];
ry(0.617200327709055) q[3];
cx q[2],q[3];
ry(1.7500357050102422) q[2];
ry(-1.4845635982496477) q[3];
cx q[2],q[3];
ry(-0.9105909055000794) q[3];
ry(1.8260045424905993) q[4];
cx q[3],q[4];
ry(2.1298218288694994) q[3];
ry(-0.8903088499497473) q[4];
cx q[3],q[4];
ry(1.730883934346779) q[4];
ry(-2.7754726258389004) q[5];
cx q[4],q[5];
ry(-1.9640480382871128) q[4];
ry(-0.6496916802190666) q[5];
cx q[4],q[5];
ry(-1.348581495145786) q[5];
ry(2.85949680112934) q[6];
cx q[5],q[6];
ry(-2.8726577586109125) q[5];
ry(0.2913078237419926) q[6];
cx q[5],q[6];
ry(2.0671238997268127) q[6];
ry(2.0318308805616696) q[7];
cx q[6],q[7];
ry(1.5370151467249087) q[6];
ry(1.8345720329341146) q[7];
cx q[6],q[7];
ry(-2.5217780212774774) q[0];
ry(-2.7599450863849126) q[1];
cx q[0],q[1];
ry(-1.6249228714609387) q[0];
ry(1.6282457324956632) q[1];
cx q[0],q[1];
ry(2.221040565569969) q[1];
ry(3.1239769621854028) q[2];
cx q[1],q[2];
ry(-0.6561495004963929) q[1];
ry(2.109204802653066) q[2];
cx q[1],q[2];
ry(2.312773929915835) q[2];
ry(-2.9454893605735903) q[3];
cx q[2],q[3];
ry(-1.9478421219226032) q[2];
ry(-1.352216345401657) q[3];
cx q[2],q[3];
ry(-2.022406838896784) q[3];
ry(-1.8742132155307178) q[4];
cx q[3],q[4];
ry(2.6576644690584668) q[3];
ry(1.648208888982441) q[4];
cx q[3],q[4];
ry(1.0967233145517596) q[4];
ry(-3.032305495470024) q[5];
cx q[4],q[5];
ry(2.555462599515279) q[4];
ry(3.0363610870008584) q[5];
cx q[4],q[5];
ry(2.7711178913910905) q[5];
ry(0.43436058560674395) q[6];
cx q[5],q[6];
ry(-1.3702728233285875) q[5];
ry(1.1729887108751784) q[6];
cx q[5],q[6];
ry(0.2381378405853851) q[6];
ry(1.0682906484394337) q[7];
cx q[6],q[7];
ry(-3.008596976433374) q[6];
ry(0.27584951982400213) q[7];
cx q[6],q[7];
ry(1.155817839153988) q[0];
ry(-1.6936668206479863) q[1];
cx q[0],q[1];
ry(-2.2360884543791055) q[0];
ry(-2.2797402079762787) q[1];
cx q[0],q[1];
ry(0.0752182556281884) q[1];
ry(0.7602183205796811) q[2];
cx q[1],q[2];
ry(-1.330631979684938) q[1];
ry(1.5331152737751417) q[2];
cx q[1],q[2];
ry(1.257083647423963) q[2];
ry(-2.3273104637869313) q[3];
cx q[2],q[3];
ry(1.95726414534938) q[2];
ry(2.6290243596838074) q[3];
cx q[2],q[3];
ry(2.9704599269466883) q[3];
ry(-1.0115728823415928) q[4];
cx q[3],q[4];
ry(-2.213421498877605) q[3];
ry(-0.732204059717283) q[4];
cx q[3],q[4];
ry(1.1327419822469604) q[4];
ry(2.0892908051691697) q[5];
cx q[4],q[5];
ry(0.7908074272381111) q[4];
ry(0.5589982585193428) q[5];
cx q[4],q[5];
ry(1.977873123312893) q[5];
ry(-2.888984426449235) q[6];
cx q[5],q[6];
ry(2.807619755114201) q[5];
ry(-2.2756499285206244) q[6];
cx q[5],q[6];
ry(-0.6793250610949743) q[6];
ry(-2.9456849897153257) q[7];
cx q[6],q[7];
ry(1.3934230303213204) q[6];
ry(1.670265071563768) q[7];
cx q[6],q[7];
ry(-2.5147452800342) q[0];
ry(0.2987399312979418) q[1];
cx q[0],q[1];
ry(-1.2203856651801155) q[0];
ry(-0.0961857860547255) q[1];
cx q[0],q[1];
ry(-0.7458860351718385) q[1];
ry(1.3695469413793806) q[2];
cx q[1],q[2];
ry(-0.16349221154608795) q[1];
ry(-0.8117332528806873) q[2];
cx q[1],q[2];
ry(-0.8706814128279565) q[2];
ry(0.5881255693686614) q[3];
cx q[2],q[3];
ry(1.8786722423489914) q[2];
ry(-0.43980299002015927) q[3];
cx q[2],q[3];
ry(3.0350968245662275) q[3];
ry(0.43756665139702733) q[4];
cx q[3],q[4];
ry(2.5286771670171446) q[3];
ry(-1.3730290743842957) q[4];
cx q[3],q[4];
ry(0.7420609099566092) q[4];
ry(-2.2788855087136497) q[5];
cx q[4],q[5];
ry(0.47649601441306383) q[4];
ry(1.3016353423084117) q[5];
cx q[4],q[5];
ry(2.230003517348257) q[5];
ry(1.4331017218880149) q[6];
cx q[5],q[6];
ry(0.966366752209444) q[5];
ry(-0.6873935207869917) q[6];
cx q[5],q[6];
ry(-0.13934200544021103) q[6];
ry(1.8095629633084576) q[7];
cx q[6],q[7];
ry(2.1105246563428883) q[6];
ry(-2.757855920467876) q[7];
cx q[6],q[7];
ry(2.96688613679679) q[0];
ry(-1.8081567876051974) q[1];
cx q[0],q[1];
ry(-3.0784760985714943) q[0];
ry(-1.5951725458017485) q[1];
cx q[0],q[1];
ry(-2.1062695538255296) q[1];
ry(-0.10913689913367283) q[2];
cx q[1],q[2];
ry(2.8168051463699935) q[1];
ry(1.3157170746237965) q[2];
cx q[1],q[2];
ry(2.885383915520177) q[2];
ry(-2.5093955906616636) q[3];
cx q[2],q[3];
ry(2.67523833495956) q[2];
ry(-0.9180533879963956) q[3];
cx q[2],q[3];
ry(-0.48574608039042166) q[3];
ry(-1.050996466484353) q[4];
cx q[3],q[4];
ry(-0.6638512375939748) q[3];
ry(0.9123624000778406) q[4];
cx q[3],q[4];
ry(-1.9457781230988502) q[4];
ry(2.1335201800112813) q[5];
cx q[4],q[5];
ry(2.905854067078653) q[4];
ry(-2.3849438180461444) q[5];
cx q[4],q[5];
ry(0.6558013712874617) q[5];
ry(0.9905013805759273) q[6];
cx q[5],q[6];
ry(2.5125589448074925) q[5];
ry(0.7211605984170147) q[6];
cx q[5],q[6];
ry(-0.8134536180224387) q[6];
ry(3.099479703657205) q[7];
cx q[6],q[7];
ry(-0.07911303327622043) q[6];
ry(-1.8758131145985757) q[7];
cx q[6],q[7];
ry(1.0058106331103422) q[0];
ry(-1.541900953842283) q[1];
cx q[0],q[1];
ry(-0.22689745824358276) q[0];
ry(-3.1262516492244323) q[1];
cx q[0],q[1];
ry(-1.1425293120335134) q[1];
ry(-2.789029613647459) q[2];
cx q[1],q[2];
ry(-1.7572252086991904) q[1];
ry(3.03520581114057) q[2];
cx q[1],q[2];
ry(-2.431971759645175) q[2];
ry(1.7200863699641704) q[3];
cx q[2],q[3];
ry(0.3356940520019931) q[2];
ry(-2.417177645935888) q[3];
cx q[2],q[3];
ry(-2.297438302204587) q[3];
ry(0.27663822932007154) q[4];
cx q[3],q[4];
ry(0.3997368318273953) q[3];
ry(-2.6489037454496347) q[4];
cx q[3],q[4];
ry(-1.863774128609494) q[4];
ry(-0.7282614478266661) q[5];
cx q[4],q[5];
ry(-0.8219821370875681) q[4];
ry(0.8985594238249032) q[5];
cx q[4],q[5];
ry(-1.2509903355384493) q[5];
ry(-0.5610068785135293) q[6];
cx q[5],q[6];
ry(0.8670386169723088) q[5];
ry(-1.3085222362442077) q[6];
cx q[5],q[6];
ry(2.4585474505186204) q[6];
ry(3.138257749770957) q[7];
cx q[6],q[7];
ry(-0.4448503116597351) q[6];
ry(-1.0212969352130534) q[7];
cx q[6],q[7];
ry(-2.8252116257661806) q[0];
ry(-2.2820691496509258) q[1];
cx q[0],q[1];
ry(0.5033656239313924) q[0];
ry(0.6783675468228426) q[1];
cx q[0],q[1];
ry(-2.907535263700132) q[1];
ry(2.841139109254681) q[2];
cx q[1],q[2];
ry(-2.060517063955731) q[1];
ry(0.922649590498045) q[2];
cx q[1],q[2];
ry(3.1286371830247526) q[2];
ry(2.6869859998656813) q[3];
cx q[2],q[3];
ry(2.2231084669992867) q[2];
ry(2.6520228822910616) q[3];
cx q[2],q[3];
ry(2.141585667073759) q[3];
ry(1.9486667171980763) q[4];
cx q[3],q[4];
ry(1.8733787817060543) q[3];
ry(-2.4979356869691096) q[4];
cx q[3],q[4];
ry(2.7957539551028723) q[4];
ry(-2.587660016433764) q[5];
cx q[4],q[5];
ry(-0.3508559833644611) q[4];
ry(1.2477025495372214) q[5];
cx q[4],q[5];
ry(1.1120508215861973) q[5];
ry(-0.5704015233294183) q[6];
cx q[5],q[6];
ry(1.1582114691132954) q[5];
ry(1.1618161551719997) q[6];
cx q[5],q[6];
ry(-1.7265323531700627) q[6];
ry(-2.9957303310053156) q[7];
cx q[6],q[7];
ry(2.8111882646072353) q[6];
ry(1.9783050576963754) q[7];
cx q[6],q[7];
ry(-1.7195230282434357) q[0];
ry(1.7884338299989186) q[1];
cx q[0],q[1];
ry(1.358405439024894) q[0];
ry(-0.4930352595731892) q[1];
cx q[0],q[1];
ry(-1.9945020981114832) q[1];
ry(1.7315104682596605) q[2];
cx q[1],q[2];
ry(1.0186361563882782) q[1];
ry(1.0916735791080647) q[2];
cx q[1],q[2];
ry(2.5573219422905558) q[2];
ry(-2.3341659436675535) q[3];
cx q[2],q[3];
ry(2.942780708336123) q[2];
ry(-1.0043951085254286) q[3];
cx q[2],q[3];
ry(1.4630014145904262) q[3];
ry(2.2873179638690364) q[4];
cx q[3],q[4];
ry(-0.36831791074861986) q[3];
ry(3.1240999368521107) q[4];
cx q[3],q[4];
ry(1.4250102651897303) q[4];
ry(-2.788841738556411) q[5];
cx q[4],q[5];
ry(1.4419079185942738) q[4];
ry(-2.016437010543796) q[5];
cx q[4],q[5];
ry(2.1162596522891732) q[5];
ry(-0.47339588550104844) q[6];
cx q[5],q[6];
ry(0.9995735191206085) q[5];
ry(0.08077703566397076) q[6];
cx q[5],q[6];
ry(-1.6058452994769887) q[6];
ry(0.61299993031525) q[7];
cx q[6],q[7];
ry(2.2719613574481885) q[6];
ry(-1.585306962098434) q[7];
cx q[6],q[7];
ry(-1.498495683595105) q[0];
ry(0.10235563936424531) q[1];
cx q[0],q[1];
ry(-0.9695324363095619) q[0];
ry(0.513646159730472) q[1];
cx q[0],q[1];
ry(-0.2499519064491961) q[1];
ry(-1.6356043251883552) q[2];
cx q[1],q[2];
ry(-0.41027793255896583) q[1];
ry(-0.46808810199896334) q[2];
cx q[1],q[2];
ry(2.38598305790275) q[2];
ry(-1.4094765030126286) q[3];
cx q[2],q[3];
ry(2.001499784017552) q[2];
ry(1.1765577648157446) q[3];
cx q[2],q[3];
ry(-0.24808428520335646) q[3];
ry(0.6527861371732394) q[4];
cx q[3],q[4];
ry(-1.5957945882003606) q[3];
ry(-0.6860526899817652) q[4];
cx q[3],q[4];
ry(-0.5444475043167865) q[4];
ry(-1.5342701386879964) q[5];
cx q[4],q[5];
ry(-0.5428902716645788) q[4];
ry(0.3806293735140649) q[5];
cx q[4],q[5];
ry(-0.6797518821360766) q[5];
ry(2.054550769916722) q[6];
cx q[5],q[6];
ry(-1.6785259170539208) q[5];
ry(-2.792612991834238) q[6];
cx q[5],q[6];
ry(-0.8267938877423753) q[6];
ry(-2.9589611921497996) q[7];
cx q[6],q[7];
ry(-2.5803176988070846) q[6];
ry(3.120758403463731) q[7];
cx q[6],q[7];
ry(-0.6459800954709922) q[0];
ry(-0.7501313159571351) q[1];
cx q[0],q[1];
ry(-0.45444181882749907) q[0];
ry(2.2136179430417027) q[1];
cx q[0],q[1];
ry(-0.7492185137994933) q[1];
ry(-0.7813745500085072) q[2];
cx q[1],q[2];
ry(-0.4716274634937489) q[1];
ry(-1.5462090486427829) q[2];
cx q[1],q[2];
ry(-1.7105753820807423) q[2];
ry(-1.7725610436168129) q[3];
cx q[2],q[3];
ry(0.4383586918641553) q[2];
ry(-1.520906506142885) q[3];
cx q[2],q[3];
ry(-2.6059801750995297) q[3];
ry(-1.2467137678137437) q[4];
cx q[3],q[4];
ry(1.9109048772477264) q[3];
ry(-1.92997796372426) q[4];
cx q[3],q[4];
ry(-1.1945187209907877) q[4];
ry(1.3389932643020361) q[5];
cx q[4],q[5];
ry(2.379411427999052) q[4];
ry(3.0756609498407994) q[5];
cx q[4],q[5];
ry(2.489178267398272) q[5];
ry(-1.8580731593325122) q[6];
cx q[5],q[6];
ry(2.640767377883967) q[5];
ry(0.3110004194100652) q[6];
cx q[5],q[6];
ry(1.8131630141647355) q[6];
ry(-2.0996200461232686) q[7];
cx q[6],q[7];
ry(0.8983037823211779) q[6];
ry(-3.1314174427582566) q[7];
cx q[6],q[7];
ry(-2.4209071877343207) q[0];
ry(0.7524155942199132) q[1];
cx q[0],q[1];
ry(-2.800817939423409) q[0];
ry(-1.5019635762046843) q[1];
cx q[0],q[1];
ry(0.5175921521329646) q[1];
ry(-0.1041019210421048) q[2];
cx q[1],q[2];
ry(1.3852450051253036) q[1];
ry(-1.3925767042228374) q[2];
cx q[1],q[2];
ry(1.4445665012403477) q[2];
ry(-0.3872567115891238) q[3];
cx q[2],q[3];
ry(0.8320074260317293) q[2];
ry(-1.4525452663421703) q[3];
cx q[2],q[3];
ry(-2.793598601872857) q[3];
ry(-1.8150218094270951) q[4];
cx q[3],q[4];
ry(-2.0261560639928518) q[3];
ry(1.4565084198293388) q[4];
cx q[3],q[4];
ry(0.25495067979976593) q[4];
ry(-0.815041818257741) q[5];
cx q[4],q[5];
ry(-1.6488376615168157) q[4];
ry(-0.47951451069374595) q[5];
cx q[4],q[5];
ry(2.204176386266724) q[5];
ry(-0.6769132274510756) q[6];
cx q[5],q[6];
ry(1.6555303537082582) q[5];
ry(-0.19610342773579958) q[6];
cx q[5],q[6];
ry(-1.7226322501532065) q[6];
ry(-0.6061292157930058) q[7];
cx q[6],q[7];
ry(1.491356696544902) q[6];
ry(2.2727929231254604) q[7];
cx q[6],q[7];
ry(-2.845602671746318) q[0];
ry(-1.763513760803372) q[1];
cx q[0],q[1];
ry(-2.866766538750795) q[0];
ry(-2.208173485372065) q[1];
cx q[0],q[1];
ry(0.3295954164682131) q[1];
ry(-2.7529260602487344) q[2];
cx q[1],q[2];
ry(1.911166890333737) q[1];
ry(-2.924551567002009) q[2];
cx q[1],q[2];
ry(-0.9443835461766605) q[2];
ry(0.3395238705756345) q[3];
cx q[2],q[3];
ry(1.682533967366032) q[2];
ry(-0.38134183596537014) q[3];
cx q[2],q[3];
ry(-1.4999600435881693) q[3];
ry(-1.619009583886811) q[4];
cx q[3],q[4];
ry(-1.3420373467694462) q[3];
ry(1.51763482484571) q[4];
cx q[3],q[4];
ry(-2.9688317100139927) q[4];
ry(0.6020524329280985) q[5];
cx q[4],q[5];
ry(2.2811551388325935) q[4];
ry(2.0018454823134997) q[5];
cx q[4],q[5];
ry(-0.7342844038163321) q[5];
ry(-0.8153179840371209) q[6];
cx q[5],q[6];
ry(0.5256347889079409) q[5];
ry(-1.8290850935351102) q[6];
cx q[5],q[6];
ry(-0.7564998789823898) q[6];
ry(-0.19760879901898423) q[7];
cx q[6],q[7];
ry(1.1056955854084953) q[6];
ry(-1.969465496272394) q[7];
cx q[6],q[7];
ry(-2.0717072347474925) q[0];
ry(-1.4610750649410005) q[1];
cx q[0],q[1];
ry(2.250639357799907) q[0];
ry(-3.0834108042187403) q[1];
cx q[0],q[1];
ry(2.212158239384755) q[1];
ry(-0.23040374019799614) q[2];
cx q[1],q[2];
ry(-0.25509394905088717) q[1];
ry(-2.195722058802125) q[2];
cx q[1],q[2];
ry(-1.264713400396257) q[2];
ry(-1.8351612235698411) q[3];
cx q[2],q[3];
ry(0.2026216398457974) q[2];
ry(2.9614404208955345) q[3];
cx q[2],q[3];
ry(0.426645331444913) q[3];
ry(2.8663157517234943) q[4];
cx q[3],q[4];
ry(0.2943324201765094) q[3];
ry(2.919574870852961) q[4];
cx q[3],q[4];
ry(-1.6516656306392976) q[4];
ry(-0.8889930808527726) q[5];
cx q[4],q[5];
ry(2.4003682194696725) q[4];
ry(-0.005015186992644681) q[5];
cx q[4],q[5];
ry(-3.1289700167262904) q[5];
ry(2.2983612667926376) q[6];
cx q[5],q[6];
ry(1.918173343338642) q[5];
ry(-0.8061117782536257) q[6];
cx q[5],q[6];
ry(-1.6466101841645293) q[6];
ry(-0.7521905368863466) q[7];
cx q[6],q[7];
ry(-2.4950193285388385) q[6];
ry(0.13869206576060128) q[7];
cx q[6],q[7];
ry(1.8481206665926342) q[0];
ry(-2.149443592697766) q[1];
cx q[0],q[1];
ry(-1.1300285456809156) q[0];
ry(-0.5810768377422146) q[1];
cx q[0],q[1];
ry(1.4367750788785605) q[1];
ry(0.6142786038137081) q[2];
cx q[1],q[2];
ry(0.2525876051058033) q[1];
ry(1.0910624354070404) q[2];
cx q[1],q[2];
ry(-2.939299022470201) q[2];
ry(2.0373894930034475) q[3];
cx q[2],q[3];
ry(-0.9782443648212382) q[2];
ry(0.24214602967774915) q[3];
cx q[2],q[3];
ry(2.8154022922594164) q[3];
ry(1.9850948267661586) q[4];
cx q[3],q[4];
ry(-3.0204319741000334) q[3];
ry(-2.2101118932946218) q[4];
cx q[3],q[4];
ry(-0.822541522951346) q[4];
ry(-0.43008898190828404) q[5];
cx q[4],q[5];
ry(1.5966241972049673) q[4];
ry(-2.9804107301447087) q[5];
cx q[4],q[5];
ry(-1.0325149504190172) q[5];
ry(-0.9556060267378745) q[6];
cx q[5],q[6];
ry(1.531729161769377) q[5];
ry(0.4845614563230054) q[6];
cx q[5],q[6];
ry(0.7222547677057696) q[6];
ry(-2.7792537767419745) q[7];
cx q[6],q[7];
ry(1.9849463642017688) q[6];
ry(-0.09893491826639611) q[7];
cx q[6],q[7];
ry(-0.5680088140625985) q[0];
ry(2.6848279311234013) q[1];
cx q[0],q[1];
ry(-0.035340633448068814) q[0];
ry(1.5162569434971072) q[1];
cx q[0],q[1];
ry(0.7137340213193237) q[1];
ry(3.0987355987181786) q[2];
cx q[1],q[2];
ry(2.164550249872501) q[1];
ry(0.062407149586094716) q[2];
cx q[1],q[2];
ry(-2.3014128683603152) q[2];
ry(0.23951872879361247) q[3];
cx q[2],q[3];
ry(-2.3772067894473334) q[2];
ry(0.6546720705230892) q[3];
cx q[2],q[3];
ry(3.0287403151435806) q[3];
ry(-0.4184589814036843) q[4];
cx q[3],q[4];
ry(-1.2749439618747473) q[3];
ry(-3.1403160873594715) q[4];
cx q[3],q[4];
ry(-3.0950557716921243) q[4];
ry(-1.4159188477030877) q[5];
cx q[4],q[5];
ry(-0.9407187273297188) q[4];
ry(-2.2393319528460127) q[5];
cx q[4],q[5];
ry(-1.546157507659979) q[5];
ry(2.137983514038269) q[6];
cx q[5],q[6];
ry(-2.8889918696626524) q[5];
ry(0.5731430003669535) q[6];
cx q[5],q[6];
ry(-2.6456088960809465) q[6];
ry(-0.04175889077395101) q[7];
cx q[6],q[7];
ry(2.4602290203500736) q[6];
ry(-0.996990053720595) q[7];
cx q[6],q[7];
ry(-0.4214863979593648) q[0];
ry(1.528340490753854) q[1];
cx q[0],q[1];
ry(2.790871240857927) q[0];
ry(-0.784181378792911) q[1];
cx q[0],q[1];
ry(-1.7906942326181587) q[1];
ry(2.235736070333072) q[2];
cx q[1],q[2];
ry(0.2878103725124501) q[1];
ry(1.9784416592422311) q[2];
cx q[1],q[2];
ry(-0.5493501486569814) q[2];
ry(1.221852003511886) q[3];
cx q[2],q[3];
ry(0.7517090631905374) q[2];
ry(-2.526382107569339) q[3];
cx q[2],q[3];
ry(-1.1383353107688352) q[3];
ry(-2.386923668865312) q[4];
cx q[3],q[4];
ry(1.250890855549349) q[3];
ry(0.032220437387163514) q[4];
cx q[3],q[4];
ry(-2.5707510513117855) q[4];
ry(-1.0306470672920276) q[5];
cx q[4],q[5];
ry(2.5203447363307117) q[4];
ry(1.8615400337119725) q[5];
cx q[4],q[5];
ry(-2.1357986605534185) q[5];
ry(0.9503190979047721) q[6];
cx q[5],q[6];
ry(-0.6985955446363814) q[5];
ry(-2.851657369153556) q[6];
cx q[5],q[6];
ry(-2.2348824413712842) q[6];
ry(-2.7477134915700496) q[7];
cx q[6],q[7];
ry(3.1203703642529588) q[6];
ry(-2.031696204231914) q[7];
cx q[6],q[7];
ry(-0.6323827347059349) q[0];
ry(-1.9000509910921268) q[1];
cx q[0],q[1];
ry(0.8330696943771239) q[0];
ry(0.5745519258796152) q[1];
cx q[0],q[1];
ry(1.4546892567098335) q[1];
ry(1.20071851971134) q[2];
cx q[1],q[2];
ry(0.9818498854935862) q[1];
ry(-0.8624528659299803) q[2];
cx q[1],q[2];
ry(-1.1335300865119424) q[2];
ry(0.04898186783505106) q[3];
cx q[2],q[3];
ry(-1.1743990575957968) q[2];
ry(2.14141951910027) q[3];
cx q[2],q[3];
ry(-2.5623169208269165) q[3];
ry(-2.9786032941851825) q[4];
cx q[3],q[4];
ry(-2.7579837555710554) q[3];
ry(-1.758851743301331) q[4];
cx q[3],q[4];
ry(-1.3165226403052586) q[4];
ry(-1.3707567023088494) q[5];
cx q[4],q[5];
ry(2.1910713283746706) q[4];
ry(-0.008978520743596096) q[5];
cx q[4],q[5];
ry(1.9988900640461367) q[5];
ry(1.2131256509318584) q[6];
cx q[5],q[6];
ry(1.1883038969717417) q[5];
ry(-0.7786543294785799) q[6];
cx q[5],q[6];
ry(-0.6797821667279829) q[6];
ry(0.4286010970852976) q[7];
cx q[6],q[7];
ry(2.9514907697466923) q[6];
ry(-3.0559706417795813) q[7];
cx q[6],q[7];
ry(-1.830893918315297) q[0];
ry(2.0075862003847584) q[1];
ry(2.7479749550575776) q[2];
ry(0.7642700053989507) q[3];
ry(0.6297730814658427) q[4];
ry(1.6986953678672885) q[5];
ry(-0.6087420233133499) q[6];
ry(-1.8433690720392883) q[7];